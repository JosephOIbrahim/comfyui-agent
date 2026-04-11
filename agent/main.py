"""Agent loop -- the core of the ComfyUI Agent.

Uses a pluggable LLM provider (Anthropic, OpenAI, Gemini, Ollama) with
streaming tool-use to run an interactive agent. The agent decides which
tools to call; we execute them and feed results back.

Includes streaming responses, context management, retry logic, and logging.
"""

import concurrent.futures
import contextvars
import logging
import threading
import time

from .config import (
    AGENT_MODEL, MAX_TOKENS, MAX_AGENT_TURNS,
    COMPACT_THRESHOLD, API_MAX_RETRIES, API_RETRY_DELAY,
)
from .context import compact, mask_processed_results
from .llm import (
    get_provider,
    LLMProvider,
    LLMRateLimitError,
    LLMConnectionError,
    LLMServerError,
    LLMError,
    ToolUseBlock,
    ToolResultBlock,
)
from .logging_config import set_correlation_id
from .streaming import StreamHandler, NullHandler
from .system_prompt import build_system_prompt
from . import tools as _tools
from .tools import ALL_TOOLS

log = logging.getLogger(__name__)


def _wrap_safe(callback):
    """Wrap a StreamHandler callback so its exceptions don't crash the agent loop.

    Custom handlers can raise (KeyError, IOError, AttributeError, etc.). Without
    this guard, a single bad handler would kill the entire conversation. We log
    the exception at warning level and continue -- the conversation survives a
    misbehaving renderer.
    """
    def safe_callback(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception:
            log.warning(
                "Stream handler %s raised; continuing",
                getattr(callback, "__name__", "<callback>"),
                exc_info=True,
            )
            return None
    return safe_callback


# Graceful shutdown flag -- checked at top of each agent turn
_shutdown = threading.Event()


def request_shutdown() -> None:
    """Signal the agent loop to exit gracefully."""
    _shutdown.set()


def create_client():
    """Create LLM provider (reads LLM_PROVIDER from env, default: anthropic).

    Backward-compatible: returns an LLMProvider instance.
    """
    return get_provider()


# ---------------------------------------------------------------------------
# Streaming with retry
# ---------------------------------------------------------------------------


def _stream_with_retry(
    provider: LLMProvider,
    *,
    model,
    max_tokens,
    system,
    tools,
    messages,
    handler: StreamHandler | None = None,
):
    """Stream a message with retry on transient failures.

    Returns an LLMResponse with common content types.
    """
    h = handler or NullHandler()
    last_error = None

    for attempt in range(API_MAX_RETRIES + 1):
        try:
            return provider.stream(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
                on_text=_wrap_safe(h.on_text),
                on_thinking=_wrap_safe(h.on_thinking),
            )

        except LLMRateLimitError as e:
            last_error = e
            if attempt < API_MAX_RETRIES:
                delay = API_RETRY_DELAY * (2 ** attempt)
                log.warning(
                    "Rate limited, retrying in %.1fs (attempt %d)", delay, attempt + 1
                )
                time.sleep(delay)
            else:
                raise

        except LLMConnectionError as e:
            last_error = e
            if attempt < API_MAX_RETRIES:
                delay = API_RETRY_DELAY * (2 ** attempt)
                log.warning("Connection error, retrying in %.1fs: %s", delay, e)
                time.sleep(delay)
            else:
                raise

        except LLMServerError as e:
            last_error = e
            if e.status_code >= 500 and attempt < API_MAX_RETRIES:
                delay = API_RETRY_DELAY * (2 ** attempt)
                log.warning(
                    "Server error %d, retrying in %.1fs", e.status_code, delay
                )
                time.sleep(delay)
            else:
                raise

    raise last_error  # pragma: no cover


# ---------------------------------------------------------------------------
# Agent turn
# ---------------------------------------------------------------------------


def run_agent_turn(
    client,
    messages: list[dict],
    system: str,
    *,
    handler: StreamHandler | None = None,
    progress=None,
) -> tuple[list[dict], bool]:
    """Run one agent turn with streaming.

    Args:
        progress: Optional progress reporter forwarded to every tool call.
                  Eliminates the need for global monkey-patching of handle().

    Returns (updated_messages, done) where done=True means the agent
    produced a final text response with no tool calls.
    """
    h = handler or NullHandler()

    # Check for graceful shutdown request
    if _shutdown.is_set():
        log.info("Shutdown requested -- exiting agent turn")
        return messages, True

    # Mask processed tool results, then compact if still over budget
    messages = mask_processed_results(messages)
    messages = compact(messages, COMPACT_THRESHOLD)

    t0 = time.monotonic()

    response = _stream_with_retry(
        client,
        model=AGENT_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=ALL_TOOLS,
        messages=messages,
        handler=handler,
    )

    elapsed = time.monotonic() - t0
    log.debug("API response in %.1fs, stop_reason=%s", elapsed, response.stop_reason)

    # Signal stream end (for newline handling in CLI)
    _wrap_safe(h.on_stream_end)()

    # Process tool calls from the complete response
    assistant_content = response.content
    has_tool_use = False
    tool_results = []

    # Collect tool calls
    tool_calls = [block for block in assistant_content if isinstance(block, ToolUseBlock)]
    has_tool_use = len(tool_calls) > 0

    if len(tool_calls) > 1:
        # Execute multiple tool calls in parallel
        for tc in tool_calls:
            _wrap_safe(h.on_tool_call)(tc.name, tc.input)

        def _run_tool(tc):
            t_tool = time.monotonic()
            result = _tools.handle(tc.name, tc.input, progress=progress)
            log.debug(
                "Tool %s completed in %.2fs", tc.name, time.monotonic() - t_tool
            )
            return tc.id, result

        # Copy the parent thread's context (which includes _conn_session from
        # routes.py / mcp_server.py) into each worker so every parallel tool
        # call lands in the right WorkflowSession.  Without this, the workers
        # inherit an empty context and _get_state() falls back to "default".
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for tc in tool_calls:
                worker_ctx = contextvars.copy_context()
                futures[executor.submit(worker_ctx.run, _run_tool, tc)] = tc
            results_map = {}
            for future in concurrent.futures.as_completed(futures):
                tc_for_future = futures[future]
                try:
                    tool_id, result = future.result()
                except Exception as e:
                    # Tool raised — return error string as the tool result so the
                    # agent turn continues rather than crashing the entire turn.
                    log.error("Tool %s raised during parallel execution: %s", tc_for_future.name, e, exc_info=True)
                    import json as _json
                    tool_id, result = tc_for_future.id, _json.dumps({"error": str(e)}, allow_nan=False)  # Cycle 61
                results_map[tool_id] = result

        # Preserve original order, notify on results
        for tc in tool_calls:
            tool_results.append(
                ToolResultBlock(tool_use_id=tc.id, content=results_map[tc.id])
            )
            _wrap_safe(h.on_tool_result)(tc.name, tc.input, results_map[tc.id])
    elif len(tool_calls) == 1:
        # Single tool call -- run directly (no thread overhead)
        tc = tool_calls[0]
        _wrap_safe(h.on_tool_call)(tc.name, tc.input)
        t_tool = time.monotonic()
        result = _tools.handle(tc.name, tc.input, progress=progress)
        log.debug(
            "Tool %s completed in %.2fs", tc.name, time.monotonic() - t_tool
        )
        tool_results.append(
            ToolResultBlock(tool_use_id=tc.id, content=result)
        )
        _wrap_safe(h.on_tool_result)(tc.name, tc.input, result)

    # Append assistant message
    messages.append({"role": "assistant", "content": assistant_content})

    if has_tool_use:
        # Feed tool results back
        messages.append({"role": "user", "content": tool_results})
        return messages, False
    else:
        # No tool calls -- agent is done with this turn
        return messages, True


def run_interactive(
    client,
    *,
    session_context: dict | None = None,
    handler: StreamHandler | None = None,
) -> None:
    """Run the full interactive agent loop with streaming.

    Args:
        session_context: Optional session data for context-aware system prompt.
        handler: StreamHandler for events (text, tool calls, input). Uses NullHandler
                 if not provided.
    """
    h = handler or NullHandler()
    # Cycle 15: only generate a fresh correlation ID if no upstream caller
    # already set one. cli.py:run sets it from the --session flag, so
    # respect that. Programmatic callers (tests, embeddings) get a UUID.
    from .logging_config import get_correlation_id
    if get_correlation_id() is None:
        set_correlation_id()  # Unique ID for this session's log entries
    system = build_system_prompt(session_context=session_context)
    messages = []

    while True:
        # Get user input
        user_text = _wrap_safe(h.on_input)()

        if user_text is None:
            break

        user_text = user_text.strip()
        if not user_text:
            continue

        if user_text.lower() in ("quit", "exit", "q"):
            break

        # Add user message
        messages.append({"role": "user", "content": user_text})

        # Run agent turns until it produces a final response
        turns = 0
        while turns < MAX_AGENT_TURNS:
            turns += 1
            try:
                messages, done = run_agent_turn(
                    client,
                    messages,
                    system,
                    handler=handler,
                )
            except LLMError as e:
                log.error("API error after retries: %s", e)
                _wrap_safe(h.on_text)(f"\n[API error: {e}. Try again.]\n")
                _wrap_safe(h.on_stream_end)()
                break

            if done:
                break

        if turns >= MAX_AGENT_TURNS:
            _wrap_safe(h.on_text)(
                f"\n[Agent hit max turns ({MAX_AGENT_TURNS}). "
                f"Ready for next input.]\n"
            )
            _wrap_safe(h.on_stream_end)()
