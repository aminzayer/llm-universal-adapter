"""
Input security layer for the LLM Universal Adapter.

The :class:`InputGuardrailMiddleware` sits in front of every LLM call and runs
two lightweight checks on the raw user prompt:

    1. **PII detection and masking** — emails, phone numbers, API keys, credit
       card-like digit runs, SSN-like numbers, and IPv4 addresses are replaced
       with type-tagged placeholders before the prompt leaves the process.
    2. **Prompt injection detection** — known attack phrases (instruction
       override, role hijack, system-prompt extraction, jailbreak prefixes,
       delimiter injection) raise :class:`SecurityViolationError` and halt
       execution before the prompt reaches the cache or the adapter.

The middleware is a regular :class:`adapter.base.BaseLLMAdapter`, so it slots
into the existing pipeline (factory -> ObservabilityMiddleware -> raw
adapter) without any protocol changes. Streaming endpoints are passed
through unchanged — the prompt itself is intercepted before the call.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, List, Optional, Tuple

from adapter.base import BaseLLMAdapter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SecurityViolationError(RuntimeError):
    """
    Raised when a prompt is refused by the security layer.

    Carries the matched pattern so callers can log or surface the reason
    without having to re-parse the message.
    """

    def __init__(self, message: str, *, matched_pattern: str = "") -> None:
        super().__init__(message)
        self.matched_pattern = matched_pattern


# ---------------------------------------------------------------------------
# PII masker
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIIMatch:
    """A single PII occurrence found in the input text."""

    pii_type: str
    start: int
    end: int
    placeholder: str


@dataclass
class PIIMaskResult:
    """The outcome of running :class:`PIIMasker` over a prompt."""

    sanitized_text: str
    matches: List[PIIMatch] = field(default_factory=list)

    @property
    def has_matches(self) -> bool:
        return bool(self.matches)


# Each entry is (pii_type, compiled_pattern, placeholder_template).
# Order matters: longer / more specific patterns come first so they are not
# shadowed by shorter fallbacks (e.g. credit-card before generic digit runs).
_PII_PATTERNS: List[Tuple[str, "re.Pattern[str]", str]] = [
    (
        "email",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "[REDACTED_EMAIL]",
    ),
    (
        "credit_card",
        # 13-19 digit runs, optionally separated by spaces or hyphens
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        "[REDACTED_CC]",
    ),
    (
        "ssn",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "[REDACTED_SSN]",
    ),
    (
        "ipv4",
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        "[REDACTED_IPV4]",
    ),
    (
        "phone_intl",
        # E.164-style: optional +, 8-15 digits, allowing common separators
        re.compile(r"(?<!\d)\+?\d{1,3}[ .-]?\(?\d{1,4}\)?[ .-]?\d{2,4}[ .-]?\d{2,4}(?!\d)"),
        "[REDACTED_PHONE]",
    ),
    (
        "openai_key",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "[REDACTED_OPENAI_KEY]",
    ),
    (
        "github_token",
        re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        "aws_access_key",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        "[REDACTED_AWS_KEY]",
    ),
    (
        "bearer_token",
        re.compile(r"(?i)\b(?:Bearer|Authorization)\s+[A-Za-z0-9._\-+/=]{20,}\b"),
        "[REDACTED_BEARER_TOKEN]",
    ),
    (
        "generic_api_key",
        # Catches `api_key=...`, `apikey: ...`, `token=...` assignments with a
        # long opaque value. Generous lower bound (16+ chars) to avoid false
        # positives on short prose.
        re.compile(r"(?i)\b(?:api[_-]?key|apikey|access[_-]?token|secret[_-]?key)\s*[=:]\s*['\"]?([A-Za-z0-9._\-+/=]{16,})"),
        "[REDACTED_API_KEY]",
    ),
]


class PIIMasker:
    """
    Regex-based PII detector and masker.

    The detector is intentionally conservative — it favors high precision
    (avoiding mangling innocent prose) over high recall. Callers who need
    stronger guarantees should layer an additional ML-based detector on top.
    """

    def __init__(self, patterns: Optional[List[Tuple[str, "re.Pattern[str]", str]]] = None) -> None:
        # Allow tests to inject custom patterns; default to the curated set.
        self._patterns = list(patterns) if patterns is not None else list(_PII_PATTERNS)

    def scan(self, text: str) -> PIIMaskResult:
        """Return a :class:`PIIMaskResult` with matches and the sanitized text."""
        if not text:
            return PIIMaskResult(sanitized_text=text)

        matches: List[PIIMatch] = []
        sanitized = text

        for pii_type, pattern, placeholder in self._patterns:
            sanitized = pattern.sub(self._record(pii_type, placeholder, matches), sanitized)

        return PIIMaskResult(sanitized_text=sanitized, matches=matches)

    @staticmethod
    def _record(pii_type: str, placeholder: str, matches: List[PIIMatch]) -> Callable[["re.Match[str]"], str]:
        """Return a :func:`re.sub` replacement callback that records each match."""

        def _replace(match: "re.Match[str]") -> str:
            matches.append(
                PIIMatch(
                    pii_type=pii_type,
                    start=match.start(),
                    end=match.end(),
                    placeholder=placeholder,
                )
            )
            return placeholder

        return _replace


# ---------------------------------------------------------------------------
# Prompt injection detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InjectionVerdict:
    """Outcome of :class:`PromptInjectionDetector.evaluate`."""

    is_malicious: bool
    reason: str = ""
    matched_pattern: str = ""

    @classmethod
    def safe(cls) -> "InjectionVerdict":
        return cls(is_malicious=False)


# Each entry is (compiled_pattern, human-readable label).
# Patterns are intentionally narrow: a single suspicious phrase is enough to
# trip the verdict. If a downstream caller needs stricter detection they can
# add additional patterns without changing call sites.
_INJECTION_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    # Instruction-override / ignore-previous
    (re.compile(r"(?i)\b(?:ignore|disregard|forget)\b[^\n.!?]{0,40}\b(?:previous|prior|all|above|system)\b[^\n.!?]{0,40}\b(?:instructions?|prompts?|rules?)\b"), "instruction_override"),
    (re.compile(r"(?i)\bforget\s+everything\s+(?:above|prior|previous)\b"), "instruction_override"),
    (re.compile(r"(?i)\bdo\s+not\s+follow\b[^\n.!?]{0,40}\b(?:rules|instructions|guidelines)\b"), "instruction_override"),
    # Role / persona hijack
    (re.compile(r"(?i)^\s*you\s+are\s+now\s+(?:a|an|the)\b"), "role_hijack"),
    (re.compile(r"(?i)\bact\s+as\s+(?:a|an|the)\s+(?:dan|jailbreak|unrestricted|evil|malicious)\b"), "role_hijack"),
    (re.compile(r"(?i)\bpretend\s+(?:to\s+be|you\s+are)\b[^\n.!?]{0,40}\b(?:evil|unrestricted|uncensored)\b"), "role_hijack"),
    # System prompt extraction
    (re.compile(r"(?i)\b(?:reveal|show|print|dump|leak|repeat)\b[^\n.!?]{0,40}\b(?:system|hidden|original)\b[^\n.!?]{0,40}\b(?:prompt|instructions?)\b"), "system_prompt_extraction"),
    (re.compile(r"(?i)\bwhat\s+(?:is|are)\s+your\s+(?:system|hidden|initial)\s+(?:prompt|instructions?)\b"), "system_prompt_extraction"),
    # Delimiter / token-injection (e.g. fake chat-template markers)
    (re.compile(r"<\|im_start\|>"), "delimiter_injection"),
    (re.compile(r"<\|im_end\|>"), "delimiter_injection"),
    (re.compile(r"<\/?system\b[^>]*>"), "delimiter_injection"),
    # Markdown-wrapped instructions trying to impersonate the system channel
    (re.compile(r"(?im)^\s*system\s*:\s*\S"), "delimiter_injection"),
    # Confirmed jailbreak prefixes
    (re.compile(r"(?i)\bDAN\s+mode\b"), "jailbreak_prefix"),
    (re.compile(r"(?i)\bjailbreak\s+prompt\b"), "jailbreak_prefix"),
    # Generic bypass phrasing
    (re.compile(r"(?i)\bbypass\b[^\n.!?]{0,30}\b(?:safety|filter|moderation|guardrails?)\b"), "jailbreak_prefix"),
]


class PromptInjectionDetector:
    """Pattern-based detector for known prompt-injection attacks."""

    def __init__(self, patterns: Optional[List[Tuple["re.Pattern[str]", str]]] = None) -> None:
        self._patterns = list(patterns) if patterns is not None else list(_INJECTION_PATTERNS)

    def evaluate(self, text: str) -> InjectionVerdict:
        """Return :class:`InjectionVerdict`. ``is_malicious=True`` on first hit."""
        if not text:
            return InjectionVerdict.safe()

        for pattern, label in self._patterns:
            match = pattern.search(text)
            if match:
                snippet = match.group(0).strip()
                # Cap snippet to keep error messages readable.
                snippet = snippet if len(snippet) <= 80 else snippet[:77] + "..."
                return InjectionVerdict(
                    is_malicious=True,
                    reason=f"Matched injection category '{label}' near: {snippet!r}",
                    matched_pattern=label,
                )
        return InjectionVerdict.safe()


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


@dataclass
class GuardrailReport:
    """Diagnostic record of what the guardrail did to a single prompt."""

    original_length: int
    sanitized_length: int
    pii_types: List[str] = field(default_factory=list)
    pii_count: int = 0
    injection_blocked: bool = False
    injection_label: str = ""


class InputGuardrailMiddleware(BaseLLMAdapter):
    """
    Middleware that screens every inbound prompt before it reaches the
    inner adapter (and therefore the cache, the LLM provider, and any
    downstream tools).

    On a clean prompt:
        * PII is masked in-place and the sanitized prompt is forwarded.
    On a malicious prompt:
        * :class:`SecurityViolationError` is raised and execution halts.

    The middleware implements the full :class:`BaseLLMAdapter` surface so it
    is drop-in compatible with anything that consumes an adapter.
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        pii_masker: Optional[PIIMasker] = None,
        injection_detector: Optional[PromptInjectionDetector] = None,
        *,
        block_on_pii: bool = False,
        on_violation: Optional[Callable[[str, InjectionVerdict], None]] = None,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.pii_masker = pii_masker or PIIMasker()
        self.injection_detector = injection_detector or PromptInjectionDetector()
        # Default behavior: mask PII, do not block on it. Strict callers can flip this.
        self.block_on_pii = block_on_pii
        self.on_violation = on_violation
        # Keep tools in sync with the inner adapter (matches ObservabilityMiddleware).
        self.tools = self.adapter.tools
        # Last report, exposed for telemetry / debugging.
        self.last_report: Optional[GuardrailReport] = None

    def register_tool(self, name: str, func: Callable[..., Any], description: str, requires_approval: bool = False) -> None:
        """Delegate tool registration to the inner adapter."""
        self.adapter.register_tool(name, func, description, requires_approval=requires_approval)
        self.tools = self.adapter.tools

    def set_redis_client(self, redis_client: Any) -> None:
        """Propagates the Redis client to the underlying adapter."""
        super().set_redis_client(redis_client)
        self.adapter.set_redis_client(redis_client)

    async def resume_with_tools(self, state: Any) -> str:
        """Resumes suspended tool execution on the underlying adapter."""
        return await self.adapter.resume_with_tools(state)

    def _screen(self, prompt: str) -> Tuple[str, GuardrailReport]:
        """Run PII masking + injection detection; return (sanitized_prompt, report)."""
        report = GuardrailReport(
            original_length=len(prompt or ""),
            sanitized_length=len(prompt or ""),
        )

        # 1) Injection detection runs on the original text so attackers can't
        #    bypass it by embedding PII-like characters in their injection.
        verdict = self.injection_detector.evaluate(prompt or "")
        if verdict.is_malicious:
            report.injection_blocked = True
            report.injection_label = verdict.matched_pattern
            self.last_report = report
            logger.warning(
                "Prompt injection blocked: pattern=%s reason=%s",
                verdict.matched_pattern,
                verdict.reason,
            )
            if self.on_violation is not None:
                try:
                    self.on_violation(prompt or "", verdict)
                except Exception:  # pragma: no cover - observer must not break the flow
                    logger.exception("on_violation callback raised; continuing.")
            raise SecurityViolationError(verdict.reason, matched_pattern=verdict.matched_pattern)

        # 2) PII masking runs on the (clean) prompt.
        pii_result = self.pii_masker.scan(prompt or "")
        if pii_result.has_matches:
            if self.block_on_pii:
                report.injection_blocked = True
                report.injection_label = "pii_blocked"
                report.pii_count = len(pii_result.matches)
                report.pii_types = sorted({m.pii_type for m in pii_result.matches})
                self.last_report = report
                raise SecurityViolationError(
                    "Prompt contains sensitive PII and blocking is enabled.",
                    matched_pattern="pii_blocked",
                )
            logger.info(
                "Masked %d PII token(s): %s",
                len(pii_result.matches),
                sorted({m.pii_type for m in pii_result.matches}),
            )
            report.pii_count = len(pii_result.matches)
            report.pii_types = sorted({m.pii_type for m in pii_result.matches})
            prompt = pii_result.sanitized_text
            report.sanitized_length = len(prompt)

        self.last_report = report
        return prompt, report

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """Screen the prompt, then forward the sanitized version to the inner adapter."""
        sanitized, _ = self._screen(prompt)
        return await self.adapter.generate_response(sanitized, **kwargs)

    async def generate_with_tools(self, prompt: str) -> str:
        """Screen the prompt, then forward the sanitized version to the inner adapter."""
        sanitized, _ = self._screen(prompt)
        return await self.adapter.generate_with_tools(sanitized)

    async def agenerate_stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Stream from the inner adapter.

        Streaming prompts are screened before the call (the same as
        ``generate_response``); individual chunks are forwarded verbatim
        because they originate from the LLM, not the user.
        """
        sanitized, _ = self._screen(prompt)
        async for chunk in self.adapter.agenerate_stream(sanitized, **kwargs):
            yield chunk

    async def get_token_count(self, text: str) -> int:
        """Delegate to the inner adapter — token counting is not security-relevant."""
        return await self.adapter.get_token_count(text)


__all__ = [
    "GuardrailReport",
    "InputGuardrailMiddleware",
    "InjectionVerdict",
    "PIIMasker",
    "PIIMaskResult",
    "PIIMatch",
    "PromptInjectionDetector",
    "SecurityViolationError",
]
