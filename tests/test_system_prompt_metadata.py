"""Tests for metadata injection into system prompt during session resume."""

import json
from unittest.mock import patch

from agent.system_prompt import build_system_prompt


# ---------------------------------------------------------------------------
# TestSystemPromptMetadata
# ---------------------------------------------------------------------------

class TestSystemPromptMetadata:
    def test_prompt_includes_last_output_context(self):
        """session_context with last_output_path injects 'Last Output Context'."""
        reconstruct_result = json.dumps({
            "has_context": True,
            "summary": "Generated with dreamshaper at 512x512.",
            "context": {
                "schema_version": 1,
                "intent": {
                    "what_artist_wanted": "A dreamy landscape",
                    "how_agent_interpreted": "Lower CFG, softer sampler",
                },
                "session": {
                    "key_params": {"model": "dreamshaper_8", "cfg": 5.0},
                },
            },
        })

        session_ctx = {
            "last_output_path": "G:/COMFYUI_Database/output/test.png",
        }

        with patch("agent.tools.handle", return_value=reconstruct_result):
            prompt = build_system_prompt(session_context=session_ctx)

        assert "Last Output Context" in prompt

    def test_prompt_no_metadata_when_no_path(self):
        """session_context without last_output_path has no 'Last Output Context'."""
        session_ctx = {"name": "test_session"}

        with patch("agent.tools.handle"):
            prompt = build_system_prompt(session_context=session_ctx)

        assert "Last Output Context" not in prompt

    def test_prompt_metadata_failure_silent(self):
        """If reconstruct_context raises, prompt builds normally."""
        session_ctx = {
            "last_output_path": "G:/COMFYUI_Database/output/test.png",
        }

        with patch("agent.tools.handle", side_effect=Exception("Disk error")):
            prompt = build_system_prompt(session_context=session_ctx)

        # Should build without crashing
        assert "ComfyUI co-pilot" in prompt
        assert "Last Output Context" not in prompt

    def test_prompt_metadata_shows_intent(self):
        """When metadata has intent, prompt includes 'Artist wanted' text."""
        reconstruct_result = json.dumps({
            "has_context": True,
            "summary": "Generated with model at 512x512.",
            "context": {
                "schema_version": 1,
                "intent": {
                    "what_artist_wanted": "Dramatic cinematic lighting",
                    "how_agent_interpreted": "Add rim light, increase contrast",
                },
                "session": {
                    "key_params": {"model": "sd15", "cfg": 7.0},
                },
            },
        })

        session_ctx = {
            "last_output_path": "G:/COMFYUI_Database/output/test.png",
        }

        with patch("agent.tools.handle", return_value=reconstruct_result):
            prompt = build_system_prompt(session_context=session_ctx)

        assert "Artist wanted" in prompt
        assert "Dramatic cinematic lighting" in prompt


# ---------------------------------------------------------------------------
# Cycle 62: knowledge file read failure must log at DEBUG
# ---------------------------------------------------------------------------

class TestKnowledgeFileReadLogging:
    """Knowledge file read failure → log.debug (Cycle 62)."""

    def test_knowledge_file_read_failure_logs_debug(self, caplog, tmp_path):
        """When a knowledge file read raises, a debug message must appear."""
        import logging
        from unittest.mock import patch, MagicMock
        from pathlib import Path

        # Create a fake .md file that raises on read_text
        bad_path = MagicMock(spec=Path)
        bad_path.stem = "comfyui_core"
        bad_path.name = "comfyui_core.md"
        bad_path.read_text.side_effect = PermissionError("no read access")

        with patch("agent.system_prompt.KNOWLEDGE_DIR", tmp_path), \
             patch("pathlib.Path.glob", return_value=iter([bad_path])), \
             caplog.at_level(logging.DEBUG, logger="agent.system_prompt"):
            from agent.system_prompt import build_system_prompt
            build_system_prompt()

        assert any("comfyui_core" in r.message or "knowledge" in r.message.lower()
                   for r in caplog.records), "Expected debug log on knowledge file read failure"

    def test_knowledge_file_read_failure_does_not_crash_prompt(self):
        """build_system_prompt must succeed even when all knowledge files fail to read."""
        from unittest.mock import patch, MagicMock
        from pathlib import Path

        bad_path = MagicMock(spec=Path)
        bad_path.stem = "comfyui_core"
        bad_path.name = "comfyui_core.md"
        bad_path.read_text.side_effect = OSError("disk error")

        with patch("pathlib.Path.glob", return_value=iter([bad_path])):
            from agent.system_prompt import build_system_prompt
            prompt = build_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
