"""Agent loop — the core of the ComfyUI Agent.

Uses the Anthropic API with tool-use to run an interactive agent.
The agent decides which tools to call; we execute them and feed results back.
"""

import anthropic
from .config import AGENT_MODEL, MAX_TOKENS, MAX_AGENT_TURNS
from .system_prompt import build_system_prompt
from .tools import ALL_TOOLS, handle as handle_tool


def create_client() -> anthropic.Anthropic:
    """Create Anthropic client (reads ANTHROPIC_API_KEY from env)."""
    return anthropic.Anthropic()


def run_agent_turn(
    client: anthropic.Anthropic,
    messages: list[dict],
    system: str,
    *,
    on_text: callable = None,
    on_tool_call: callable = None,
    on_thinking: callable = None,
) -> tuple[list[dict], bool]:
    """
    Run one agent turn: send messages, process response, handle tool calls.

    Returns (updated_messages, done) where done=True means the agent
    produced a final text response with no tool calls.
    """
    response = client.messages.create(
        model=AGENT_MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=ALL_TOOLS,
        messages=messages,
    )

    # Process response content blocks
    assistant_content = response.content
    has_tool_use = False
    tool_results = []

    for block in assistant_content:
        if block.type == "thinking" and on_thinking:
            on_thinking(block.thinking)

        elif block.type == "text":
            if on_text:
                on_text(block.text)

        elif block.type == "tool_use":
            has_tool_use = True
            if on_tool_call:
                on_tool_call(block.name, block.input)

            # Execute the tool
            result = handle_tool(block.name, block.input)
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
    on_text: callable = None,
    on_tool_call: callable = None,
    on_thinking: callable = None,
    on_user_input: callable = None,
) -> None:
    """
    Run the full interactive agent loop.

    Callbacks:
        on_text(text)           — agent produced text output
        on_tool_call(name, inp) — agent is calling a tool
        on_thinking(text)       — agent thinking (extended thinking)
        on_user_input()         — should return user's input string, or None to quit
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
            messages, done = run_agent_turn(
                client,
                messages,
                system,
                on_text=on_text,
                on_tool_call=on_tool_call,
                on_thinking=on_thinking,
            )
            if done:
                break

        if turns >= MAX_AGENT_TURNS:
            if on_text:
                on_text(f"\n[Agent hit max turns ({MAX_AGENT_TURNS}). Ready for next input.]")
