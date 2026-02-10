"""Agent loop — the core of the ComfyUI Agent.

Uses the Anthropic API with streaming tool-use to run an interactive agent.
The agent decides which tools to call; we execute them and feed results back.

Includes streaming responses, context management, retry logic, and logging.
"""

import logging
import time

import anthropic

from .config import (
    AGENT_MODEL, MAX_TOKENS, MAX_AGENT_TURNS,
    COMPACT_THRESHOLD, API_MAX_RETRIES, API_RETRY_DELAY,
)
from .system_prompt import build_system_prompt
from .tools import ALL_TOOLS, handle as handle_tool

log = logging.getLogger(__name__)


def create_client() -> anthropic.Anthropic:
    """Create Anthropic client (reads ANTHROPIC_API_KEY from env)."""
    return anthropic.Anthropic()


# ---------------------------------------------------------------------------
# Context management
# ---------------------------------------------------------------------------

def _estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (~4 chars per token)."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += len(str(item.get("content", ""))) // 4
                elif hasattr(item, "text"):
                    total += len(item.text) // 4
                elif hasattr(item, "input"):
                    total += len(str(item.input)) // 4
                else:
                    total += len(str(item)) // 4
    return total


def _compact_messages(messages: list[dict], threshold: int) -> list[dict]:
    """Compact messages to stay within context budget.

    Strategy:
      1. Truncate tool results > 2000 chars
      2. If still over, drop oldest exchanges keeping recent 6 messages
    """
    estimated = _estimate_tokens(messages)
    if estimated <= threshold:
        return messages

    log.info("Context at ~%d tokens, compacting (threshold: %d)", estimated, threshold)

    # Pass 1: Truncate large tool results
    compacted = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            new_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str) and len(content) > 2000:
                        block = {**block, "content": content[:2000] + "\n[...truncated]"}
                new_content.append(block)
            compacted.append({**msg, "content": new_content})
        else:
            compacted.append(msg)

    estimated = _estimate_tokens(compacted)
    if estimated <= threshold:
        log.info("Compacted to ~%d tokens via tool result truncation", estimated)
        return compacted

    # Pass 2: Drop oldest exchanges, keep recent 6 messages (3 exchanges)
    keep_recent = 6
    if len(compacted) > keep_recent:
        dropped_count = len(compacted) - keep_recent
        summary_msg = {
            "role": "user",
            "content": (
                f"[Earlier context: {dropped_count} messages omitted to manage "
                f"context window. Recent conversation follows.]"
            ),
        }
        compacted = [summary_msg] + compacted[-keep_recent:]
        log.info("Dropped %d older messages, keeping %d recent", dropped_count, keep_recent)

    return compacted


# ---------------------------------------------------------------------------
# Streaming with retry
# ---------------------------------------------------------------------------

def _stream_with_retry(
    client,
    *,
    model,
    max_tokens,
    system,
    tools,
    messages,
    on_text_delta=None,
    on_thinking_delta=None,
):
    """Stream a message with retry on transient failures.

    Returns the final Message object.
    """
    last_error = None

    for attempt in range(API_MAX_RETRIES + 1):
        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=system,
                tools=tools,
                messages=messages,
            ) as stream:
                for event in stream:
                    if event.type == "content_block_delta":
                        delta = event.delta
                        if hasattr(delta, "text") and on_text_delta:
                            on_text_delta(delta.text)
                        elif hasattr(delta, "thinking") and on_thinking_delta:
                            on_thinking_delta(delta.thinking)
                return stream.get_final_message()

        except anthropic.RateLimitError as e:
            last_error = e
            if attempt < API_MAX_RETRIES:
                delay = API_RETRY_DELAY * (2 ** attempt)
                log.warning("Rate limited, retrying in %.1fs (attempt %d)", delay, attempt + 1)
                time.sleep(delay)
            else:
                raise

        except anthropic.APIConnectionError as e:
            last_error = e
            if attempt < API_MAX_RETRIES:
                delay = API_RETRY_DELAY * (2 ** attempt)
                log.warning("Connection error, retrying in %.1fs: %s", delay, e)
                time.sleep(delay)
            else:
                raise

        except anthropic.APIStatusError as e:
            last_error = e
            if e.status_code >= 500 and attempt < API_MAX_RETRIES:
                delay = API_RETRY_DELAY * (2 ** attempt)
                log.warning("Server error %d, retrying in %.1fs", e.status_code, delay)
                time.sleep(delay)
            else:
                raise

    raise last_error  # pragma: no cover


# ---------------------------------------------------------------------------
# Agent turn
# ---------------------------------------------------------------------------

def run_agent_turn(
    client: anthropic.Anthropic,
    messages: list[dict],
    system: str,
    *,
    on_text_delta: callable = None,
    on_tool_call: callable = None,
    on_thinking_delta: callable = None,
    on_stream_end: callable = None,
) -> tuple[list[dict], bool]:
    """
    Run one agent turn with streaming.

    Returns (updated_messages, done) where done=True means the agent
    produced a final text response with no tool calls.
    """
    # Compact context if needed
    messages = _compact_messages(messages, COMPACT_THRESHOLD)

    t0 = time.monotonic()

    response = _stream_with_retry(
        client,
        model=AGENT_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=ALL_TOOLS,
        messages=messages,
        on_text_delta=on_text_delta,
        on_thinking_delta=on_thinking_delta,
    )

    elapsed = time.monotonic() - t0
    log.debug("API response in %.1fs, stop_reason=%s", elapsed, response.stop_reason)

    # Signal stream end (for newline handling in CLI)
    if on_stream_end:
        on_stream_end()

    # Process tool calls from the complete response
    assistant_content = response.content
    has_tool_use = False
    tool_results = []

    for block in assistant_content:
        if block.type == "tool_use":
            has_tool_use = True
            if on_tool_call:
                on_tool_call(block.name, block.input)

            t_tool = time.monotonic()
            result = handle_tool(block.name, block.input)
            log.debug("Tool %s completed in %.2fs", block.name, time.monotonic() - t_tool)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

    # Append assistant message
    messages.append({"role": "assistant", "content": assistant_content})

    if has_tool_use:
        # Feed tool results back
        messages.append({"role": "user", "content": tool_results})
        return messages, False
    else:
        # No tool calls — agent is done with this turn
        return messages, True


def run_interactive(
    client: anthropic.Anthropic,
    *,
    on_text_delta: callable = None,
    on_tool_call: callable = None,
    on_thinking_delta: callable = None,
    on_stream_end: callable = None,
    on_user_input: callable = None,
) -> None:
    """
    Run the full interactive agent loop with streaming.

    Callbacks:
        on_text_delta(text)      — incremental text chunk (streaming)
        on_tool_call(name, inp)  — agent is calling a tool
        on_thinking_delta(text)  — thinking text chunk (streaming)
        on_stream_end()          — stream finished (for newline handling)
        on_user_input()          — should return user input string, or None to quit
    """
    system = build_system_prompt()
    messages = []

    while True:
        # Get user input
        if on_user_input:
            user_text = on_user_input()
        else:
            user_text = input("> ")

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
                    on_text_delta=on_text_delta,
                    on_tool_call=on_tool_call,
                    on_thinking_delta=on_thinking_delta,
                    on_stream_end=on_stream_end,
                )
            except anthropic.APIError as e:
                log.error("API error after retries: %s", e)
                if on_text_delta:
                    on_text_delta(f"\n[API error: {e}. Try again.]\n")
                if on_stream_end:
                    on_stream_end()
                break

            if done:
                break

        if turns >= MAX_AGENT_TURNS:
            if on_text_delta:
                on_text_delta(
                    f"\n[Agent hit max turns ({MAX_AGENT_TURNS}). Ready for next input.]\n"
                )
            if on_stream_end:
                on_stream_end()
