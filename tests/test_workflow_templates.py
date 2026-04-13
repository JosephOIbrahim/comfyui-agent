"""Tests for workflow_templates — path traversal security and basic functionality."""

import json


from agent.tools import workflow_templates


# ---------------------------------------------------------------------------
# Cycle 29: template name path traversal security tests
# ---------------------------------------------------------------------------

class TestTemplateNameSecurity:
    """_resolve_template_path must reject names that could escape the templates dir."""

    def test_path_traversal_dot_dot_slash_rejected(self):
        """Template name with ../ must return error (not find arbitrary files)."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "../../etc/passwd",
        }))
        assert "error" in result
        # Must NOT have loaded file content
        assert "class_type" not in str(result)

    def test_path_traversal_backslash_rejected(self):
        """Template name with backslash traversal must return error."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "..\\..\\Windows\\system32\\config\\SAM",
        }))
        assert "error" in result

    def test_path_traversal_null_byte_rejected(self):
        """Template name with null byte must return error."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "valid\x00../../etc/passwd",
        }))
        assert "error" in result

    def test_simple_template_name_accepted(self):
        """A simple template name does not trip the security check."""
        # txt2img_sd15 is a known built-in template
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "txt2img_sd15",
        }))
        # Either succeeds (returns workflow) or not-found error — NOT a traversal block
        if "error" in result:
            # Error must be about the template not being found, not a traversal block
            assert "not found" in result["error"].lower() or "template" in result["error"].lower()

    def test_list_templates_returns_names(self):
        """list_workflow_templates must return a list without path traversal."""
        result = json.loads(workflow_templates.handle("list_workflow_templates", {}))
        # Should have either templates or an error (if templates dir missing)
        assert "templates" in result or "error" in result
        if "templates" in result:
            # All template names must be simple strings (no slashes)
            for t in result["templates"]:
                name = t.get("name", "")
                assert "/" not in name and "\\" not in name, (
                    f"Template name contains path separator: {name!r}"
                )


# ---------------------------------------------------------------------------
# Cycle 45 — get_workflow_template required field guard
# ---------------------------------------------------------------------------

class TestGetTemplateRequiredField:
    """get_workflow_template must return a structured error when 'template' is
    missing, empty, or not a string — never KeyError or TypeError.
    """

    def test_missing_template_returns_error(self):
        """Omitting 'template' must return an error dict, not raise."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {}))
        assert "error" in result
        assert "template" in result["error"].lower()

    def test_empty_string_template_returns_error(self):
        """Empty string 'template' must return an error dict."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "",
        }))
        assert "error" in result

    def test_none_template_returns_error(self):
        """None 'template' must return an error dict, not AttributeError."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": None,
        }))
        assert "error" in result

    def test_integer_template_returns_error(self):
        """Non-string 'template' (int) must return an error dict."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": 123,
        }))
        assert "error" in result

    def test_valid_template_name_not_blocked_by_guard(self):
        """Guard must not break the normal not-found error path for an unknown name."""
        result = json.loads(workflow_templates.handle("get_workflow_template", {
            "template": "nonexistent_template_xyz",
        }))
        # Should return not-found error (not the required-field error)
        assert "error" in result
        assert "nonexistent_template_xyz" in result["error"]
