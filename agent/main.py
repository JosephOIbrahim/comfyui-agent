"""Agent loop — the core of the ComfyUI Agent.

Uses the Anthropic API with streaming tool-use to run an interactive agent.
The agent decides which tools to call; we execute them and feed results back.

Includes streaming responses, context management, retry logic, and logging.
"""

import concurrent.futures
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


def _summarize_dropped(messages: list[dict]) -> str:
    """Build a structured summary of dropped messages for context continuity.

    Extracts: user requests, tool calls made, key decisions/results, and workflow state.
    """
    sections = {
        "user_requests": [],
        "tools_called": [],
        "key_results": [],
        "workflow_info": None,
    }

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and isinstance(content, str):
            # Capture user requests (first 100 chars each)
            text = content.strip()
            if text and not text.startswith("["):  # Skip system summaries
                sections["user_requests"].append(text[:100])

        elif role == "user" and isinstance(content, list):
            # Extract tool results summary
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                result_text = block.get("content", "")
                if not isinstance(result_text, str):
                    continue
                # Extract key info from tool results
                if '"loaded_path"' in result_text or '"saved"' in result_text:
                    try:
                        import json
                        data = json.loads(result_text)
                        path = data.get("loaded_path") or data.get("saved") or data.get("file")
                        if path:
                            sections["workflow_info"] = path
                    except Exception:
                        pass

        elif role == "assistant" and isinstance(content, list):
            # Extract tool call names from assistant content blocks
            for block in content:
                if hasattr(block, "type") and block.type == "tool_use":
                    sections["tools_called"].append(block.name)
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    sections["tools_called"].append(block.get("name", "?"))

    # Build structured summary
    lines = ["[Context Summary - earlier messages compacted]"]

    if sections["user_requests"]:
        lines.append("Topics discussed: " + "; ".join(sections["user_requests"][:5]))

    if sections["tools_called"]:
        # Deduplicate, preserve order
        seen = set()
        unique = []
        for t in sections["tools_called"]:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        lines.append("Tools used: " + ", ".join(unique))

    if sections["workflow_info"]:
        lines.append(f"Workflow context: {sections['workflow_info']}")

    lines.append("Recent conversation follows.")
    return "\n".join(lines)


def _compact_messages(messages: list[dict], threshold: int) -> list[dict]:
    """Compact messages to stay within context budget.

    Strategy:
      1. Mask old tool results (replace processed results with compact references)
      2. Truncate remaining large tool results > 2000 chars
      3. If still over, drop oldest exchanges with structured summary
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

    # Pass 2: Drop oldest exchanges with structured summary
    keep_recent = 6
    if len(compacted) > keep_recent:
        dropped = compacted[:-keep_recent]
        dropped_count = len(dropped)
        summary_text = _summarize_dropped(dropped)
        summary_msg = {"role": "user", "content": summary_text}
        compacted = [summary_msg] + compacted[-keep_recent:]
        log.info("Dropped %d older messages, keeping %d recent", dropped_count, keep_recent)

    return compacted


# ---------------------------------------------------------------------------
# Observation masking
# ---------------------------------------------------------------------------

_MASK_THRESHOLD = 1500  # chars — tool results larger than this get masked after processing


def _mask_processed_results(messages: list[dict]) -> list[dict]:
    """Replace large tool results in older turns with compact summaries.

    Only masks results from turns that have already been processed
    (i.e., there's a subsequent assistant message). The most recent
    tool results are kept intact since the model hasn't responded to them yet.
    """
    if len(messages) < 3:
        return messages

    # Find the last user message with tool_results (the one the model hasn't processed yet)
    last_tool_result_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg["role"] == "user" and isinstance(msg.get("content"), list):
            has_tool_result = any(
                isinstance(b, dict) and b.get("type") == "tool_result"
                for b in msg["content"]
            )
            if has_tool_result:
                last_tool_result_idx = i
                break

    masked = []
    for i, msg in enumerate(messages):
        # Only mask tool results that are NOT the most recent batch
        if (
            i < last_tool_result_idx
            and msg["role"] == "user"
            and isinstance(msg.get("content"), list)
        ):
            new_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = block.get("content", "")
                    if isinstance(content, str) and len(content) > _MASK_THRESHOLD:
                        # Build compact reference
                        preview = content[:200]
                        char_count = len(content)
                        block = {
                            **block,
                            "content": (
                                f"[Processed result ({char_count} chars)]\n"
                                f"{preview}...\n"
                                f"[Full output was processed in prior turn]"
                            ),
                        }
                new_content.append(block)
            masked.append({**msg, "content": new_content})
        else:
            masked.append(msg)

    return masked


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
    # Mask processed tool results, then compact if still over budget
    messages = _mask_processed_results(messages)
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

    # Collect tool calls
    tool_calls = [block for block in assistant_content if block.type == "tool_use"]
    has_tool_use = len(tool_calls) > 0

    if len(tool_calls) > 1:
        # Execute multiple tool calls in parallel
        for tc in tool_calls:
            if on_tool_call:
                on_tool_call(tc.name, tc.input)

        def _run_tool(tc):
            t_tool = time.monotonic()
            result = handle_tool(tc.name, tc.input)
            log.debug("Tool %s completed in %.2fs", tc.name, time.monotonic() - t_tool)
            return tc.id, result

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(_run_tool, tc): tc for tc in tool_calls}
            results_map = {}
            for future in concurrent.futures.as_completed(futures):
                tool_id, result = future.result()
                results_map[tool_id] = result

        # Preserve original order
        for tc in tool_calls:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tc.id,
                "content": results_map[tc.id],
            })
    elif len(tool_calls) == 1:
        # Single tool call — run directly (no thread overhead)
        tc = tool_calls[0]
        if on_tool_call:
            on_tool_call(tc.name, tc.input)
        t_tool = time.monotonic()
        result = handle_tool(tc.name, tc.input)
        log.debug("Tool %s completed in %.2fs", tc.name, time.monotonic() - t_tool)
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": tc.id,
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
    session_context: dict | None = None,
    on_text_delta: callable = None,
    on_tool_call: callable = None,
    on_thinking_delta: callable = None,
    on_stream_end: callable = None,
    on_user_input: callable = None,
) -> None:
    """
    Run the full interactive agent loop with streaming.

    Args:
        session_context: Optional session data for context-aware system prompt.

    Callbacks:
        on_text_delta(text)      — incremental text chunk (streaming)
        on_tool_call(name, inp)  — agent is calling a tool
        on_thinking_delta(text)  — thinking text chunk (streaming)
        on_stream_end()          — stream finished (for newline handling)
        on_user_input()          — should return user input string, or None to quit
    """
    system = build_system_prompt(session_context=session_context)
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
