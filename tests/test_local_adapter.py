import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from adapter.local_adapter import LocalModelAdapter, LocalModelError


@pytest.fixture(autouse=True)
def fast_retries():
    """
    Fixture to mock ``time.sleep`` used by ``tenacity`` during backoff.
    This ensures tests involving retries run instantaneously.
    """
    with patch("tenacity.nap.time.sleep"), patch("asyncio.sleep"):
        yield


def _build_post_mock(responses: list[dict[str, Any]]) -> MagicMock:
    """
    Build a mock for ``aiohttp.ClientSession.post`` that returns each entry
    of ``responses`` on successive calls. Each entry is a dict with ``status``
    and either ``json_body``, ``text_body``, or ``sse_lines``.

    Returns a tuple of (post_mock, response_iter) where response_iter is an
    iterator yielding the pre-built response mocks. Callers should iterate
    manually via the side_effect on ``__aenter__``.
    """
    response_iter = iter(_make_responses(responses))

    def next_resp() -> MagicMock:
        try:
            return next(response_iter)
        except StopIteration as exc:  # pragma: no cover - test bug
            raise AssertionError("post() called more times than expected") from exc

    cm = MagicMock()
    # AsyncMock wraps side_effect calls in a coroutine, so this returns the
    # pre-built response mock from the iterator each time __aenter__ is awaited.
    cm.__aenter__ = AsyncMock(side_effect=next_resp)
    cm.__aexit__ = AsyncMock(return_value=False)

    def call_post(*args: Any, **kwargs: Any) -> MagicMock:
        return cm

    post_mock = MagicMock(side_effect=call_post)
    return post_mock


def _make_responses(specs: list[dict[str, Any]]) -> list[MagicMock]:
    """Build a list of pre-made response mocks from the test specs.

    The adapter does ``body = await resp.text()`` and then ``json.loads(body)``,
    so the default text body is the JSON serialisation of ``json_body`` (or
    whatever ``text_body`` overrides it with).
    """
    out: list[MagicMock] = []
    for spec in specs:
        resp = MagicMock()
        resp.status = spec["status"]
        json_body = spec.get("json_body", {})
        default_text = json.dumps(json_body) if "text_body" not in spec else spec["text_body"]
        resp.text = AsyncMock(return_value=default_text)
        resp.json = AsyncMock(return_value=json_body)
        resp.content = MagicMock()
        resp.content.__aiter__ = lambda self_, s=spec: _async_iter(s.get("sse_lines", []))
        out.append(resp)
    return out


async def _async_iter(lines: list[bytes]):
    for line in lines:
        yield line


def _patch_session(post_mock: MagicMock) -> Any:
    """
    Patches ``LocalModelAdapter._get_session`` so it returns a mock whose
    ``post`` is the supplied ``post_mock``. The async-context-manager
    contract on ``post`` is supplied by ``_build_post_mock``.
    """
    session = MagicMock()
    session.post = post_mock
    session.closed = False
    session.close = AsyncMock()
    return patch.object(LocalModelAdapter, "_get_session", AsyncMock(return_value=session))


# =============================================================================
# Constructor validation
# =============================================================================


def test_local_adapter_requires_base_url() -> None:
    """Constructing without a base_url must raise ValueError."""
    with pytest.raises(ValueError, match="base_url is required"):
        LocalModelAdapter(base_url="", model="llama3.1")


def test_local_adapter_requires_model() -> None:
    """Constructing without a model must raise ValueError."""
    with pytest.raises(ValueError, match="model is required"):
        LocalModelAdapter(base_url="http://localhost:11434/v1", model="")


def test_local_adapter_strips_trailing_slash() -> None:
    """Trailing slashes on base_url must be normalised."""
    adapter = LocalModelAdapter(base_url="http://localhost:11434/v1/", model="llama3.1")
    assert adapter.base_url == "http://localhost:11434/v1"


# =============================================================================
# generate_response
# =============================================================================


@pytest.mark.asyncio
async def test_local_adapter_generate_response_success() -> None:
    """A 200 response with OpenAI-shaped body yields the assistant content."""
    post_mock = _build_post_mock([
        {"status": 200, "json_body": {"choices": [{"message": {"content": "hi from local"}}]}},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        response = await adapter.generate_response("hello")

    assert response == "hi from local"
    assert post_mock.call_count == 1
    # Verify URL hit and that the body was shaped like OpenAI's chat completion call.
    called_url = post_mock.call_args.args[0]
    assert called_url == "http://localhost:11434/v1/chat/completions"
    body = post_mock.call_args.kwargs["json"]
    assert body["model"] == "llama3.1"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["stream"] is False


@pytest.mark.asyncio
async def test_local_adapter_generate_response_sends_bearer_when_key_set() -> None:
    """An api_key must surface as an Authorization header."""
    post_mock = _build_post_mock([
        {"status": 200, "json_body": {"choices": [{"message": {"content": "ok"}}]}},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(
            base_url="http://localhost:8000/v1", model="qwen2.5", api_key="secret"
        )
        await adapter.generate_response("hi")

    headers = post_mock.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret"
    assert headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_local_adapter_generate_response_omits_auth_header_without_key() -> None:
    """Without an api_key, no Authorization header should be set."""
    post_mock = _build_post_mock([
        {"status": 200, "json_body": {"choices": [{"message": {"content": "ok"}}]}},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        await adapter.generate_response("hi")

    headers = post_mock.call_args.kwargs["headers"]
    assert "Authorization" not in headers


@pytest.mark.asyncio
async def test_local_adapter_generate_response_retries_on_5xx() -> None:
    """Two 500s then a 200 should result in 3 calls and the final answer."""
    post_mock = _build_post_mock([
        {"status": 500, "text_body": "boom"},
        {"status": 500, "text_body": "boom"},
        {"status": 200, "json_body": {"choices": [{"message": {"content": "recovered"}}]}},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        response = await adapter.generate_response("hi")

    assert response == "recovered"
    assert post_mock.call_count == 3


@pytest.mark.asyncio
async def test_local_adapter_generate_response_does_not_retry_on_4xx() -> None:
    """A 4xx error must surface immediately — it is a caller bug."""
    post_mock = _build_post_mock([
        {"status": 400, "text_body": "bad request"},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        with pytest.raises(LocalModelError) as exc_info:
            await adapter.generate_response("hi")

    assert exc_info.value.status == 400
    assert post_mock.call_count == 1


@pytest.mark.asyncio
async def test_local_adapter_generate_response_raises_on_empty_choices() -> None:
    """An empty ``choices`` array must raise LocalModelError, not silently return empty."""
    post_mock = _build_post_mock([
        {"status": 200, "json_body": {"choices": []}},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        with pytest.raises(LocalModelError, match="empty choices"):
            await adapter.generate_response("hi")


@pytest.mark.asyncio
async def test_local_adapter_generate_response_forwards_kwargs() -> None:
    """Extra kwargs (temperature, max_tokens) must reach the request body."""
    post_mock = _build_post_mock([
        {"status": 200, "json_body": {"choices": [{"message": {"content": "ok"}}]}},
    ])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        await adapter.generate_response("hi", temperature=0.2, max_tokens=64)

    body = post_mock.call_args.kwargs["json"]
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 64


# =============================================================================
# agenerate_stream
# =============================================================================


def _sse_line(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


@pytest.mark.asyncio
async def test_local_adapter_agenerate_stream_yields_chunks() -> None:
    """Streaming responses must yield concatenated delta.content strings."""
    sse_payloads = [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " "}}]},
        {"choices": [{"delta": {"content": "world"}}]},
        {"choices": [{"delta": {}}]},
    ]
    sse_lines = [_sse_line(p) for p in sse_payloads] + [b"data: [DONE]\n\n"]

    post_mock = _build_post_mock([{"status": 200, "sse_lines": sse_lines}])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        chunks = []
        async for chunk in adapter.agenerate_stream("hi"):
            chunks.append(chunk)

    assert "".join(chunks) == "Hello world"
    # The body must advertise streaming.
    body = post_mock.call_args.kwargs["json"]
    assert body["stream"] is True


@pytest.mark.asyncio
async def test_local_adapter_agenerate_stream_skips_blank_lines() -> None:
    """Blank lines and lines without a ``data:`` prefix must be ignored."""
    sse_lines = [
        b"",
        b": ping",
        _sse_line({"choices": [{"delta": {"content": "ok"}}]}),
        b"",
    ]
    post_mock = _build_post_mock([{"status": 200, "sse_lines": sse_lines}])

    with _patch_session(post_mock):
        adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
        chunks = []
        async for chunk in adapter.agenerate_stream("hi"):
            chunks.append(chunk)

    assert "".join(chunks) == "ok"


# =============================================================================
# get_token_count
# =============================================================================


@pytest.mark.asyncio
async def test_local_adapter_get_token_count_approximates() -> None:
    """Token count is a local heuristic — chars // 4, min 1 for non-empty."""
    adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
    assert await adapter.get_token_count("") == 0
    assert await adapter.get_token_count("hi") == 1
    # 16 characters // 4 == 4 tokens
    assert await adapter.get_token_count("hello world!!!!!") == 4


# =============================================================================
# generate_with_tools
# =============================================================================


@pytest.mark.asyncio
async def test_local_adapter_generate_with_tools_raises() -> None:
    """Tool calling is intentionally not supported on the local adapter."""
    adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
    with pytest.raises(NotImplementedError, match="does not support tool calling"):
        await adapter.generate_with_tools("anything")


# =============================================================================
# aclose
# =============================================================================


@pytest.mark.asyncio
async def test_local_adapter_aclose_is_idempotent() -> None:
    """Calling aclose() twice must not raise."""
    adapter = LocalModelAdapter(base_url="http://localhost:11434/v1", model="llama3.1")
    # Force a session to be created.
    await adapter._get_session()
    await adapter.aclose()
    # Second call should be a no-op even though the session is already gone.
    await adapter.aclose()
    assert adapter._session is None


def test_local_model_error_carries_status_and_body() -> None:
    """The error type must expose the HTTP status and response body."""
    err = LocalModelError(status=503, body="oops")
    assert err.status == 503
    assert err.body == "oops"
    assert "503" in str(err)
    assert "oops" in str(err)


def test_local_model_error_is_runtime_error() -> None:
    """LocalModelError must subclass RuntimeError so generic exception handlers catch it."""
    assert issubclass(LocalModelError, RuntimeError)
    # aiohttp is imported and available for the adapter
    assert aiohttp is not None