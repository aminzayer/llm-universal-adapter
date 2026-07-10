"""
Dynamic multi-agent orchestration for the LLM Universal Adapter.

The :class:`SwarmOrchestrator` performs lightweight intent classification and routes
a shared execution context to specialized worker agents. All inter-agent
communication is enforced as pydantic ``BaseModel`` payloads that travel
through the existing :class:`utils.structured.StructuredGenerator`.

Built-in workers:
    * :class:`SearchAgent`    - Uses an ``ElasticsearchDiscoveryTool``.
    * :class:`SummaryAgent`   - Uses the underlying LLM to summarize text.

Custom workers can be added to the orchestrator via :meth:`SwarmOrchestrator.register_worker`.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field, ValidationError

from adapter.base import BaseLLMAdapter
from prompts import PromptRegistry
from tools.es_discovery import ElasticsearchDiscoveryTool
from utils.structured import StructuredGenerator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schemas (inter-agent payloads)
# ---------------------------------------------------------------------------


class ClassificationDecision(BaseModel):
    """The output of :class:`ClassifierAgent`."""

    intent: str = Field(..., description="The target worker identifier, e.g. 'search' or 'summary'.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model's confidence in the routing decision.")
    reasoning: str = Field("", description="Short natural-language explanation of the choice.")


class SwarmTask(BaseModel):
    """The shared execution context passed between agents."""

    request_id: str = Field(..., description="A unique correlation id for tracing.")
    user_input: str = Field(..., description="The original user prompt or query.")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata propagated to workers (e.g. session, locale).",
    )


class SearchAgentInput(BaseModel):
    """Input contract for :class:`SearchAgent`."""

    query: str
    index_name: str = "documents"
    top_k: int = 5
    max_count_per_domain: int = 2


class SearchAgentOutput(BaseModel):
    """Output contract for :class:`SearchAgent`."""

    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    raw_response: str = ""


class SummaryAgentInput(BaseModel):
    """Input contract for :class:`SummaryAgent`."""

    text: str
    max_points: int = 3


class SummaryAgentOutput(BaseModel):
    """Output contract for :class:`SummaryAgent`."""

    summary: str
    key_points: List[str] = Field(default_factory=list)


class SwarmResult(BaseModel):
    """The orchestrator's final, structured response."""

    request_id: str
    intent: str
    worker: str
    output: BaseModel
    classification: ClassificationDecision


# ---------------------------------------------------------------------------
# Worker base + built-in workers
# ---------------------------------------------------------------------------


class BaseWorker(ABC):
    """
    Abstract base class for swarm workers.

    Workers accept a :class:`SwarmTask` plus the raw :class:`ClassificationDecision`
    that triggered them, and return a pydantic ``BaseModel`` describing their
    output. They must not raise on expected, recoverable errors — instead they
    should encode the failure in their output schema.
    """

    #: Identifier that matches ``ClassificationDecision.intent`` (e.g. ``"search"``).
    name: str = ""

    #: Pydantic model class the worker is expected to return.
    output_model: Type[BaseModel] = BaseModel

    @abstractmethod
    async def run(self, task: SwarmTask, decision: ClassificationDecision) -> BaseModel:
        """Execute the worker and return a structured result."""
        raise NotImplementedError


class SearchAgent(BaseWorker):
    """
    Worker that queries Elasticsearch via :class:`ElasticsearchDiscoveryTool`.

    The tool returns a JSON string; we parse it and surface a structured
    :class:`SearchAgentOutput` so downstream consumers (and the orchestrator)
    receive a typed payload.
    """

    name = "search"
    output_model = SearchAgentOutput

    def __init__(
        self,
        es_tool: ElasticsearchDiscoveryTool,
        index_name: str = "documents",
        top_k: int = 5,
        max_count_per_domain: int = 2,
    ) -> None:
        self.es_tool = es_tool
        self.index_name = index_name
        self.top_k = top_k
        self.max_count_per_domain = max_count_per_domain

    async def run(self, task: SwarmTask, decision: ClassificationDecision) -> BaseModel:
        params = SearchAgentInput(
            query=task.user_input,
            index_name=self.index_name,
            top_k=self.top_k,
            max_count_per_domain=self.max_count_per_domain,
        )
        raw = await self.es_tool.search(
            query=params.query,
            index_name=params.index_name,
            top_k=params.top_k,
            max_count_per_domain=params.max_count_per_domain,
        )
        results: List[Dict[str, Any]] = []
        try:
            payload = json.loads(raw) if raw else {}
            if isinstance(payload, dict) and "error" in payload:
                logger.warning("SearchAgent received an error payload: %s", payload["error"])
            else:
                results = list(payload.get("results", [])) if isinstance(payload, dict) else []
        except json.JSONDecodeError as exc:
            logger.warning("SearchAgent could not decode Elasticsearch response: %s", exc)

        return SearchAgentOutput(
            query=params.query,
            results=results,
            raw_response=raw,
        )


class SummaryAgent(BaseWorker):
    """
    Worker that summarizes text using the LLM, producing a structured
    :class:`SummaryAgentOutput` via :class:`StructuredGenerator`.
    """

    name = "summary"
    output_model = SummaryAgentOutput

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        *,
        prompt_registry: Optional[PromptRegistry] = None,
    ) -> None:
        self.adapter = adapter
        self.structured = StructuredGenerator(adapter)
        self._registry: Optional[PromptRegistry] = prompt_registry

    def _get_registry(self) -> PromptRegistry:
        """
        Return the configured registry, lazily constructing a no-DB registry
        (backed by the module-level :data:`DEFAULT_PROMPTS`) on first use.
        """
        if self._registry is None:
            self._registry = PromptRegistry(db_pool=None)
        return self._registry

    async def run(self, task: SwarmTask, decision: ClassificationDecision) -> BaseModel:
        params = SummaryAgentInput(text=task.user_input)
        template = await self._get_registry().get_prompt("summary.system")
        prompt = template.format(text=params.text, max_points=params.max_points)
        return await self.structured.generate(prompt, SummaryAgentOutput)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class ClassifierAgent:
    """
    A lightweight agent that classifies incoming user intent into one of the
    registered worker identifiers. It always returns a :class:`ClassificationDecision`.
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        candidate_intents: List[str],
        confidence_threshold: float = 0.5,
        *,
        prompt_registry: Optional[PromptRegistry] = None,
    ) -> None:
        if not candidate_intents:
            raise ValueError("ClassifierAgent requires at least one candidate intent.")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0.")
        self.adapter = adapter
        self.candidate_intents = list(candidate_intents)
        self.confidence_threshold = confidence_threshold
        self.structured = StructuredGenerator(adapter)
        self._registry: Optional[PromptRegistry] = prompt_registry

    def _get_registry(self) -> PromptRegistry:
        """
        Return the configured registry, lazily constructing a no-DB registry
        (backed by the module-level :data:`DEFAULT_PROMPTS`) on first use.
        """
        if self._registry is None:
            self._registry = PromptRegistry(db_pool=None)
        return self._registry

    async def classify(self, task: SwarmTask) -> ClassificationDecision:
        intent_list = ", ".join(f"'{i}'" for i in self.candidate_intents)
        template = await self._get_registry().get_prompt("classifier.system")
        prompt = template.format(intent_list=intent_list, user_input=task.user_input)
        decision = await self.structured.generate(prompt, ClassificationDecision)

        if decision.intent not in self.candidate_intents:
            raise ValueError(
                f"Classifier returned intent '{decision.intent}' which is not in "
                f"the candidate set {self.candidate_intents}."
            )
        return decision

    def is_confident(self, decision: ClassificationDecision) -> bool:
        return decision.confidence >= self.confidence_threshold


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class UnknownIntentError(KeyError):
    """Raised when the classifier produces an intent with no registered worker."""


class SwarmOrchestrator:
    """
    Coordinates a :class:`ClassifierAgent` with a registry of :class:`BaseWorker`
    implementations. All messages are pydantic models produced through
    :class:`StructuredGenerator`, ensuring strict inter-agent contracts.
    """

    def __init__(
        self,
        classifier: ClassifierAgent,
        default_intent: Optional[str] = None,
    ) -> None:
        self.classifier = classifier
        self.default_intent = default_intent
        self._workers: Dict[str, BaseWorker] = {}

    def register_worker(self, worker: BaseWorker) -> None:
        if not worker.name:
            raise ValueError("Worker.name must be a non-empty identifier.")
        if worker.name in self._workers:
            raise ValueError(f"Worker '{worker.name}' is already registered.")
        self._workers[worker.name] = worker

    @property
    def registered_intents(self) -> List[str]:
        return list(self._workers.keys())

    def get_worker(self, intent: str) -> BaseWorker:
        try:
            return self._workers[intent]
        except KeyError as exc:
            raise UnknownIntentError(
                f"No worker registered for intent '{intent}'. "
                f"Known intents: {sorted(self._workers)}"
            ) from exc

    async def dispatch(self, task: SwarmTask) -> SwarmResult:
        decision = await self.classifier.classify(task)

        # Low-confidence fallback: prefer the configured default, else the first worker.
        if not self.classifier.is_confident(decision):
            fallback = self.default_intent or (
                next(iter(self._workers)) if self._workers else None
            )
            if fallback is None:
                raise RuntimeError("No workers registered and classifier is not confident.")
            if fallback != decision.intent:
                logger.info(
                    "Classifier confidence %.2f below threshold %.2f; "
                    "falling back from '%s' to '%s'.",
                    decision.confidence,
                    self.classifier.confidence_threshold,
                    decision.intent,
                    fallback,
                )
                decision = ClassificationDecision(
                    intent=fallback,
                    confidence=decision.confidence,
                    reasoning=decision.reasoning + " [low-confidence fallback applied]",
                )

        worker = self.get_worker(decision.intent)
        try:
            output = await worker.run(task, decision)
        except Exception as exc:
            from orchestration.hitl import ApprovalRequiredError, HITLStateManager
            if isinstance(exc, ApprovalRequiredError):
                adapter = self.classifier.adapter
                redis_client = getattr(adapter, "redis_client", None)
                if redis_client:
                    manager = HITLStateManager(redis_client)
                    state = await manager.get_state(exc.state_id)
                    if state:
                        state.task = task.model_dump()
                        state.worker_name = worker.name
                        await manager.save_state(state)
            raise exc

        return SwarmResult(
            request_id=task.request_id,
            intent=decision.intent,
            worker=worker.name,
            output=output,
            classification=decision,
        )


__all__ = [
    "BaseWorker",
    "ClassificationDecision",
    "ClassifierAgent",
    "SearchAgent",
    "SearchAgentInput",
    "SearchAgentOutput",
    "SummaryAgent",
    "SummaryAgentInput",
    "SummaryAgentOutput",
    "SwarmOrchestrator",
    "SwarmResult",
    "SwarmTask",
    "UnknownIntentError",
]


# Re-export ValidationError so callers can import a swarm-related error surface
# without reaching into pydantic directly.
_ = ValidationError  # pragma: no cover - silence unused-import warnings
