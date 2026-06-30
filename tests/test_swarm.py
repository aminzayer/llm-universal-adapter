"""
Tests for the SwarmOrchestrator. Sub-agent responses are mocked at the
StructuredGenerator boundary, the Elasticsearch tool boundary, and the
worker boundary so the test never touches a real LLM or ES cluster.
"""

import json
from typing import Any, AsyncGenerator, List

import pytest
from pydantic import BaseModel

from adapter.base import BaseLLMAdapter
from orchestration.swarm import (
    BaseWorker,
    ClassificationDecision,
    ClassifierAgent,
    SearchAgent,
    SearchAgentOutput,
    SummaryAgent,
    SummaryAgentOutput,
    SwarmOrchestrator,
    SwarmResult,
    SwarmTask,
    UnknownIntentError,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class SequenceAdapter(BaseLLMAdapter):
    """
    Adapter that returns a fixed sequence of strings, regardless of prompt.
    Mirrors ``MockStructuredAdapter`` in test_structured.py but kept local
    to avoid coupling the swarm tests to that file.
    """

    def __init__(self, responses: List[str]) -> None:
        super().__init__()
        self.responses = responses
        self.call_count = 0
        self.prompts: List[str] = []

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        res = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return res

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        yield ""

    async def get_token_count(self, text: str) -> int:
        return len(text.split())

    async def generate_with_tools(self, prompt: str) -> str:
        return ""


class FakeESTool:
    """Stand-in for ElasticsearchDiscoveryTool that returns canned JSON."""

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: List[dict] = []

    def search(self, query: str, index_name: str, top_k: int = 5, max_count_per_domain: int = 2) -> str:
        self.calls.append(
            {
                "query": query,
                "index_name": index_name,
                "top_k": top_k,
                "max_count_per_domain": max_count_per_domain,
            }
        )
        return json.dumps(self.payload)


class StubWorker(BaseWorker):
    """Worker that returns a pre-canned pydantic model and records its calls."""

    name = "stub"
    output_model = SearchAgentOutput

    # pyrefly: ignore [bad-function-definition]
    def __init__(self, name: str = "stub", output: BaseModel = None) -> None:
        self.name = name
        self.output = output
        self.calls: List[Any] = []

    async def run(self, task: SwarmTask, decision: ClassificationDecision) -> BaseModel:
        self.calls.append((task, decision))
        return self.output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def search_task() -> SwarmTask:
    return SwarmTask(
        request_id="req-1",
        user_input="Find recent articles about vector databases",
    )


@pytest.fixture
def summary_task() -> SwarmTask:
    return SwarmTask(
        request_id="req-2",
        user_input="The quick brown fox jumps over the lazy dog. " * 20,
        context={"lang": "en"},
    )


# ---------------------------------------------------------------------------
# ClassifierAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classifier_returns_typed_decision() -> None:
    adapter = SequenceAdapter(
        [json.dumps({"intent": "search", "confidence": 0.9, "reasoning": "user asked to find"})]
    )
    classifier = ClassifierAgent(adapter, candidate_intents=["search", "summary"])

    decision = await classifier.classify(SwarmTask(request_id="r", user_input="find x"))

    assert isinstance(decision, ClassificationDecision)
    assert decision.intent == "search"
    assert decision.confidence == pytest.approx(0.9)
    assert classifier.is_confident(decision) is True


@pytest.mark.asyncio
async def test_classifier_rejects_unknown_intent() -> None:
    adapter = SequenceAdapter(
        [json.dumps({"intent": "mystery", "confidence": 0.8, "reasoning": "?"})]
    )
    classifier = ClassifierAgent(adapter, candidate_intents=["search", "summary"])

    with pytest.raises(ValueError, match="not in the candidate set"):
        await classifier.classify(SwarmTask(request_id="r", user_input="x"))


def test_classifier_validates_threshold() -> None:
    adapter = SequenceAdapter(["{}"])
    with pytest.raises(ValueError):
        ClassifierAgent(adapter, candidate_intents=["a"], confidence_threshold=1.5)
    with pytest.raises(ValueError):
        ClassifierAgent(adapter, candidate_intents=[])


# ---------------------------------------------------------------------------
# SearchAgent (mocked Elasticsearch tool)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_agent_uses_tool_and_returns_typed_output(search_task: SwarmTask) -> None:
    es_payload = {
        "results": [
            {"title": "A", "url": "https://a.example.com/a", "content": "..."},
            {"title": "B", "url": "https://b.example.com/b", "content": "..."},
        ]
    }
    fake_es = FakeESTool(es_payload)
    # pyrefly: ignore [bad-argument-type]
    agent = SearchAgent(fake_es, index_name="docs", top_k=4, max_count_per_domain=1)

    result = await agent.run(
        search_task,
        ClassificationDecision(intent="search", confidence=0.9, reasoning="r"),
    )

    assert isinstance(result, SearchAgentOutput)
    assert result.query == search_task.user_input
    assert len(result.results) == 2
    assert result.results[0]["title"] == "A"
    assert result.raw_response == json.dumps(es_payload)
    # Tool called with propagated parameters
    assert fake_es.calls[0]["index_name"] == "docs"
    assert fake_es.calls[0]["top_k"] == 4


@pytest.mark.asyncio
async def test_search_agent_handles_error_payload() -> None:
    fake_es = FakeESTool({"error": "Search failed: backend down"})
    # pyrefly: ignore [bad-argument-type]
    agent = SearchAgent(fake_es)

    result = await agent.run(
        SwarmTask(request_id="r", user_input="x"),
        ClassificationDecision(intent="search", confidence=0.5),
    )

    # pyrefly: ignore [missing-attribute]
    assert result.results == []
    # pyrefly: ignore [missing-attribute]
    assert "error" in result.raw_response


# ---------------------------------------------------------------------------
# SummaryAgent (mocked LLM via SequenceAdapter)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_summary_agent_returns_typed_output(summary_task: SwarmTask) -> None:
    summary_payload = {
        "summary": "A fox jumps over a dog.",
        "key_points": ["Foxes are quick", "Dogs are lazy", "This happens often"],
    }
    adapter = SequenceAdapter([json.dumps(summary_payload)])
    agent = SummaryAgent(adapter)

    result = await agent.run(
        summary_task,
        ClassificationDecision(intent="summary", confidence=0.7),
    )

    assert isinstance(result, SummaryAgentOutput)
    assert result.summary == "A fox jumps over a dog."
    assert len(result.key_points) == 3


# ---------------------------------------------------------------------------
# SwarmOrchestrator: dispatch
# ---------------------------------------------------------------------------


def _build_orchestrator_with_stubs(decision: ClassificationDecision, worker: BaseWorker) -> SwarmOrchestrator:
    # The classifier uses its own LLM mock to emit the decision JSON.
    adapter = SequenceAdapter([decision.model_dump_json()])
    classifier = ClassifierAgent(adapter, candidate_intents=[worker.name, "summary"])
    orch = SwarmOrchestrator(classifier=classifier, default_intent="summary")
    orch.register_worker(worker)
    return orch


@pytest.mark.asyncio
async def test_dispatch_routes_to_search_worker(search_task: SwarmTask) -> None:
    es_payload = {"results": [{"title": "hit", "url": "https://x", "content": "c"}]}
    fake_es = FakeESTool(es_payload)
    # pyrefly: ignore [bad-argument-type]
    search_worker = SearchAgent(fake_es)
    decision = ClassificationDecision(intent="search", confidence=0.95, reasoning="r")
    orch = _build_orchestrator_with_stubs(decision, search_worker)

    result = await orch.dispatch(search_task)

    assert isinstance(result, SwarmResult)
    assert result.intent == "search"
    assert result.worker == "search"
    assert isinstance(result.output, SearchAgentOutput)
    assert result.output.results[0]["title"] == "hit"
    assert result.classification.intent == "search"


@pytest.mark.asyncio
async def test_dispatch_routes_to_summary_worker(summary_task: SwarmTask) -> None:
    summary_payload = {"summary": "short", "key_points": ["p1"]}
    adapter = SequenceAdapter([json.dumps(summary_payload)])
    summary_worker = SummaryAgent(adapter)
    decision = ClassificationDecision(intent="summary", confidence=0.8, reasoning="r")
    orch = _build_orchestrator_with_stubs(decision, summary_worker)

    result = await orch.dispatch(summary_task)

    assert result.intent == "summary"
    assert result.worker == "summary"
    assert isinstance(result.output, SummaryAgentOutput)
    assert result.output.summary == "short"


@pytest.mark.asyncio
async def test_dispatch_low_confidence_falls_back_to_default() -> None:
    adapter = SequenceAdapter([json.dumps({"intent": "search", "confidence": 0.2, "reasoning": "unsure"})])
    classifier = ClassifierAgent(adapter, candidate_intents=["search", "summary"], confidence_threshold=0.5)
    summary_payload = {"summary": "fallback", "key_points": []}
    summary_adapter = SequenceAdapter([json.dumps(summary_payload)])
    summary_worker = SummaryAgent(summary_adapter)
    orch = SwarmOrchestrator(classifier=classifier, default_intent="summary")
    orch.register_worker(summary_worker)
    # Also register a stub search worker so 'search' is a valid intent.
    orch.register_worker(StubWorker(name="search", output=SearchAgentOutput(query="x")))

    task = SwarmTask(request_id="r", user_input="maybe find things")
    result = await orch.dispatch(task)

    assert result.intent == "summary"
    assert result.worker == "summary"
    assert "low-confidence fallback applied" in result.classification.reasoning


@pytest.mark.asyncio
async def test_dispatch_unknown_intent_raises(search_task: SwarmTask) -> None:
    # Classifier emits a valid intent string, but no worker is registered for it.
    decision = ClassificationDecision(intent="summary", confidence=0.9, reasoning="r")
    adapter = SequenceAdapter([decision.model_dump_json()])
    classifier = ClassifierAgent(adapter, candidate_intents=["summary"])
    orch = SwarmOrchestrator(classifier=classifier)
    # No workers registered.

    with pytest.raises(UnknownIntentError):
        await orch.dispatch(search_task)


def test_register_worker_rejects_duplicate() -> None:
    orch = SwarmOrchestrator(classifier=ClassifierAgent(SequenceAdapter(["{}"]), candidate_intents=["x"]))
    w1 = StubWorker(name="x", output=SearchAgentOutput(query="q"))
    w2 = StubWorker(name="x", output=SearchAgentOutput(query="q"))
    orch.register_worker(w1)
    with pytest.raises(ValueError, match="already registered"):
        orch.register_worker(w2)


def test_register_worker_rejects_empty_name() -> None:
    orch = SwarmOrchestrator(classifier=ClassifierAgent(SequenceAdapter(["{}"]), candidate_intents=["x"]))
    bad = StubWorker(name="", output=SearchAgentOutput(query="q"))
    with pytest.raises(ValueError, match="non-empty"):
        orch.register_worker(bad)


def test_registered_intents_lists_workers() -> None:
    orch = SwarmOrchestrator(classifier=ClassifierAgent(SequenceAdapter(["{}"]), candidate_intents=["a", "b"]))
    orch.register_worker(StubWorker(name="a", output=SearchAgentOutput(query="q")))
    orch.register_worker(StubWorker(name="b", output=SearchAgentOutput(query="q")))
    assert sorted(orch.registered_intents) == ["a", "b"]


# ---------------------------------------------------------------------------
# Inter-agent contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inter_agent_messages_are_pydantic_models() -> None:
    """
    Verify that the orchestrator never returns a raw dict/string — every
    inter-agent message is a pydantic BaseModel instance.
    """
    decision = ClassificationDecision(intent="summary", confidence=0.9)
    summary_payload = {"summary": "s", "key_points": []}
    adapter = SequenceAdapter([decision.model_dump_json(), json.dumps(summary_payload)])
    classifier = ClassifierAgent(adapter, candidate_intents=["summary"])
    summary_worker = SummaryAgent(adapter)
    orch = SwarmOrchestrator(classifier=classifier)
    orch.register_worker(summary_worker)

    result = await orch.dispatch(SwarmTask(request_id="r", user_input="text"))

    assert isinstance(result, SwarmResult)
    assert isinstance(result.classification, ClassificationDecision)
    assert isinstance(result.output, BaseModel)
    assert isinstance(result.output, SummaryAgentOutput)
