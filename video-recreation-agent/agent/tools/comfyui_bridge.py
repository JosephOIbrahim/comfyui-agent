"""
ComfyUI Bridge — Connects the comfyui-agent repo to a ComfyUI installation.

This is the critical piece that lets your agent talk to ComfyUI even though
they live in different directories. It handles:
  1. API communication (queue, monitor, fetch)
  2. Node schema discovery via /object_info
  3. File transfer (upload inputs, download outputs)
  4. Model/node inventory
"""

import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path
from typing import Optional


class ComfyUIBridge:
    """Bridge between comfyui-agent and a running ComfyUI instance."""

    def __init__(self, host: str = None, port: int = None, comfyui_path: str = None):
        self.host = host or os.environ.get("COMFYUI_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("COMFYUI_PORT", "8188"))
        self.comfyui_path = Path(comfyui_path or os.environ.get("COMFYUI_PATH", ""))
        self.base_url = f"http://{self.host}:{self.port}"
        self._node_schemas = None  # cached after first fetch

    # ── Connection ──────────────────────────────────────────

    def is_connected(self) -> bool:
        """Check if ComfyUI is running and reachable."""
        try:
            req = urllib.request.Request(f"{self.base_url}/system_stats")
            resp = urllib.request.urlopen(req, timeout=5)
            return resp.status == 200
        except Exception:
            return False

    def get_system_stats(self) -> dict:
        """Get ComfyUI system info (GPU, VRAM, etc.)."""
        return self._get("/system_stats")

    # ── Node Schemas (the key to building workflows) ───────

    def get_node_schemas(self, force_refresh: bool = False) -> dict:
        """
        Fetch ALL available node types with their full input/output specs.
        This is what Jo Zhang means by 'reads ComfyUI source code' —
        the /object_info endpoint gives you everything.
        """
        if self._node_schemas and not force_refresh:
            return self._node_schemas
        self._node_schemas = self._get("/object_info")
        return self._node_schemas

    def get_node_schema(self, class_type: str) -> Optional[dict]:
        """Get schema for a specific node type."""
        schemas = self.get_node_schemas()
        return schemas.get(class_type)

    def find_nodes_by_category(self, category_keyword: str) -> list:
        """Find nodes matching a category keyword (e.g., 'video', 'kling')."""
        schemas = self.get_node_schemas()
        results = []
        for name, info in schemas.items():
            cat = info.get("category", "").lower()
            if category_keyword.lower() in cat:
                results.append({
                    "class_type": name,
                    "category": info.get("category", ""),
                    "display_name": info.get("display_name", name),
                })
        return results

    def validate_workflow(self, workflow: dict) -> list:
        """Validate a workflow against installed nodes. Returns list of errors."""
        schemas = self.get_node_schemas()
        errors = []
        for node_id, node in workflow.items():
            ct = node.get("class_type", "UNKNOWN")
            if ct not in schemas:
                errors.append(f"Node {node_id}: unknown class_type '{ct}'")
            else:
                # Check required inputs
                schema = schemas[ct]
                required = schema.get("input", {}).get("required", {})
                for input_name in required:
                    if input_name not in node.get("inputs", {}):
                        errors.append(f"Node {node_id} ({ct}): missing required input '{input_name}'")
        return errors

    # ── Model Inventory ────────────────────────────────────

    def list_models(self, model_type: str = "checkpoints") -> list:
        """List installed models by type."""
        schemas = self.get_node_schemas()
        loader_map = {
            "checkpoints": "CheckpointLoaderSimple",
            "loras": "LoraLoader",
            "vae": "VAELoader",
            "controlnet": "ControlNetLoader",
            "upscale": "UpscaleModelLoader",
        }
        loader = loader_map.get(model_type)
        if not loader or loader not in schemas:
            return []
        required = schemas[loader].get("input", {}).get("required", {})
        # The first required input usually has the model list
        for input_name, input_spec in required.items():
            if isinstance(input_spec, list) and len(input_spec) > 0:
                if isinstance(input_spec[0], list):
                    return input_spec[0]
        return []

    # ── Workflow Execution ─────────────────────────────────

    def queue_workflow(self, workflow: dict, client_id: str = "comfyui-agent") -> dict:
        """Queue a workflow for execution. Returns prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": client_id,
        }
        return self._post("/prompt", payload)

    def get_queue(self) -> dict:
        """Get current queue status."""
        return self._get("/queue")

    def get_history(self, prompt_id: str = None) -> dict:
        """Get execution history (optionally for specific prompt)."""
        path = f"/history/{prompt_id}" if prompt_id else "/history"
        return self._get(path)

    def get_job_status(self, prompt_id: str) -> dict:
        """Check if a specific job is running, pending, or done."""
        history = self.get_history(prompt_id)
        if prompt_id in history:
            return {"status": "done", "outputs": history[prompt_id].get("outputs", {})}

        queue = self.get_queue()
        for item in queue.get("queue_running", []):
            if item[1] == prompt_id:
                return {"status": "running"}
        for item in queue.get("queue_pending", []):
            if item[1] == prompt_id:
                return {"status": "pending"}

        return {"status": "unknown"}

    # ── File Operations ────────────────────────────────────

    def upload_image(self, filepath: str, subfolder: str = "agent_inputs") -> dict:
        """Upload an image to ComfyUI's input directory."""
        import mimetypes
        boundary = "----AgentBoundary"
        filename = os.path.basename(filepath)
        mime_type = mimetypes.guess_type(filepath)[0] or "image/png"

        with open(filepath, "rb") as f:
            file_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'
            f"Content-Type: {mime_type}\r\n\r\n"
        ).encode() + file_data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
            f"{subfolder}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="type"\r\n\r\n'
            f"input\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())

    def download_output(self, filename: str, subfolder: str = "", save_to: str = None) -> bytes:
        """Download a generated output file from ComfyUI."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
        })
        req = urllib.request.Request(f"{self.base_url}/view?{params}")
        resp = urllib.request.urlopen(req)
        data = resp.read()

        if save_to:
            os.makedirs(os.path.dirname(save_to), exist_ok=True)
            with open(save_to, "wb") as f:
                f.write(data)

        return data

    def free_vram(self) -> dict:
        """Free GPU VRAM by unloading cached models."""
        return self._post("/free", {"unload_models": True, "free_memory": True})

    # ── Internal HTTP helpers ──────────────────────────────

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(f"{self.base_url}{path}")
        resp = urllib.request.urlopen(req, timeout=30)
        return json.loads(resp.read())

    def _post(self, path: str, data: dict) -> dict:
        payload = json.dumps(data).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())


# ── CLI quick test ─────────────────────────────────────────

if __name__ == "__main__":
    bridge = ComfyUIBridge()

    if bridge.is_connected():
        print("✓ ComfyUI is running")
        stats = bridge.get_system_stats()
        devices = stats.get("devices", [{}])
        if devices:
            d = devices[0]
            vram_gb = d.get("vram_total", 0) / (1024**3)
            print(f"  GPU: {d.get('name', '?')} ({vram_gb:.0f}GB VRAM)")

        # Show available model types
        for mtype in ["checkpoints", "loras", "vae"]:
            models = bridge.list_models(mtype)
            print(f"  {mtype}: {len(models)} installed")

        # Show node categories
        schemas = bridge.get_node_schemas()
        categories = set()
        for info in schemas.values():
            cat = info.get("category", "")
            if cat:
                categories.add(cat.split("/")[0])
        print(f"  Node categories: {', '.join(sorted(categories)[:10])}...")
        print(f"  Total node types: {len(schemas)}")
    else:
        print("✗ ComfyUI is not running")
        print(f"  Tried: {bridge.base_url}")
        print(f"  Start ComfyUI first, then re-run this test")
