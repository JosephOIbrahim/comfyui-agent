# ComfyUI Agent -- Setup Guide

**Your AI co-pilot for ComfyUI.** You type what you want in plain English. It talks to ComfyUI for you -- finding models, tweaking settings, running generations, and explaining what's happening along the way.

It works through Claude (Anthropic's AI), so you'll need an API key. The whole setup takes about 10 minutes.

---

## 0. What Does This Thing Do?

You know how you click through ComfyUI nodes, drag connections, and tweak values? This agent does that for you through conversation.

You say: *"Make it dreamier"* -- it lowers the CFG, adjusts the sampler, and explains why.
You say: *"Find me a good anime LoRA"* -- it searches CivitAI and tells you what's trending.
You say: *"What's in this workflow?"* -- it reads the JSON and explains it like a colleague would.

77 tools. All through natural language. Nothing to memorize.

---

## 1. Before You Start -- Checklist

> **Read this first.** If any of these aren't ready, the install will hit a wall. Better to check now.

- [ ] **Python 3.10 or newer**
- [ ] **ComfyUI installed and working**
- [ ] **An Anthropic API key**

---

### Python

Open any terminal and type:

```bash
python --version
```

You should see something like `Python 3.11.9` or `Python 3.14.0`.

If the number is 3.10 or higher, you're good. Move on.

If it says "not recognized" or shows 3.9 or lower, install Python from [python.org/downloads](https://www.python.org/downloads/). During install, **check the box that says "Add Python to PATH"** -- this is important.

---

### ComfyUI

Can you open ComfyUI and queue a prompt? Have you generated at least one image? Yes? You're good.

If not, get ComfyUI working first. The agent talks *to* ComfyUI -- it can't replace it.

---

### Anthropic API Key

This is how the agent thinks. It costs money per use -- roughly **$5-20/month** for moderate use depending on which model you pick.

Here's how to get one:

1. Go to [console.anthropic.com](https://console.anthropic.com/)
2. Sign up for an account (or log in)
3. Click **API Keys** in the left sidebar
4. Click **Create Key**
5. Give it a name like "comfyui-agent"
6. Copy the key -- it starts with `sk-ant-`

**Save this key somewhere.** You'll paste it during setup. You won't be able to see it again on Anthropic's site.

---

## 2. Download the Agent

Open a terminal. Navigate to where you want the agent to live (your home folder is fine).

```bash
git clone https://github.com/JosephOIbrahim/comfyui-agent.git
```

Then move into that folder:

```bash
cd comfyui-agent
```

> `cd` means "change directory" -- it tells the terminal to look inside that folder from now on.

**If you don't have git:** Click the green **Code** button on the [GitHub page](https://github.com/JosephOIbrahim/comfyui-agent), choose **Download ZIP**, unzip it, and open a terminal inside that folder.

You should see files like `README.md`, `CLAUDE.md`, and an `agent/` folder. If you do, you're in the right place.

---

## 3. Install It

From inside the `comfyui-agent` folder, run:

```bash
pip install -e .
```

This tells Python where the agent code lives so you can run it from anywhere.

**If it worked:** You'll see `Successfully installed comfyui-agent-0.4.0` (or similar) near the end.

**If you see red text about permissions:** Try this instead:

```bash
pip install -e . --user
```

**If `pip` isn't recognized:** Try `python -m pip install -e .` instead. If that doesn't work either, revisit the Python install step and make sure you checked "Add to PATH".

> **Alternative:** Run `python scripts/deploy.py` -- it does the install, setup, and validation all in one go.

---

## 4. Tell It Where Your Stuff Is

Run the interactive setup:

```bash
python scripts/setup.py
```

It asks 5 questions. Here's what each one means:

---

### Question 1: Anthropic API Key

Paste the `sk-ant-...` key you copied earlier.

**Heads up:** When you paste, nothing will appear on screen. That's normal -- the terminal hides passwords and keys. Paste it and press Enter.

---

### Question 2: ComfyUI Database Path

This is the folder where your **models/**, **Custom_Nodes/**, and **output/** folders live.

Common locations:
- `G:/COMFYUI_Database`
- `C:/ComfyUI`
- `~/ComfyUI` (Mac/Linux)

Not sure? Open your ComfyUI folder and look for a `models/` subfolder. That parent folder is what you want.

---

### Question 3: ComfyUI Startup Script (Windows only)

If you have a `.bat` file you double-click to start ComfyUI, paste its full path here.

This lets the launcher script start ComfyUI for you automatically. If you don't have one or aren't sure, press Enter to skip. You can always start ComfyUI yourself.

---

### Question 4: Output Directory

Press Enter unless your output folder is in a different location than your database path. Most people can skip this.

---

### Question 5: Claude Model

Pick which AI model powers the CLI chat mode.

- **Sonnet** (default) -- faster, cheaper, good for most tasks
- **Opus** -- slower, more expensive, better at complex reasoning

Start with Sonnet. You can change it later in the `.env` file.

> This only affects `agent run` (CLI mode). If you use Claude Code, it picks its own model.

---

**If it worked:** You'll see `[+] Created .env` at the end.

If something went wrong, you can create the `.env` file manually. Copy `.env.example` to `.env` and fill in your API key and ComfyUI path.

---

## 5. Launch Everything

### Option A: The launcher script (recommended for Windows)

Double-click **`scripts\comfyui_with_agent.bat`**

It does three things:
1. Starts ComfyUI in a new window
2. Waits until ComfyUI is ready (you'll see "attempt 1... attempt 2..." while it waits)
3. Prints instructions for connecting

**If it worked:** You'll see "All systems go" and a list of ways to connect.

**If it says "ComfyUI script not found":** Open the `.bat` file in a text editor and change the `COMFYUI_BAT` path at the top to point to your ComfyUI launcher.

---

### Option B: Manual start

1. Start ComfyUI however you normally do
2. Wait until it's loaded (you can see the web UI at `http://127.0.0.1:8188`)
3. Open a new terminal in the `comfyui-agent` folder
4. Run:

```bash
agent run
```

**If it worked:** You'll see a prompt waiting for your input.

**First thing to type:**

```
What models do I have?
```

You should get back a list of your installed models. Nice. That worked. You're in.

---

## 6. Now Try These

Copy-paste any of these into the agent. They go from safe to impressive.

---

**"What models do I have installed?"**

Lists everything in your models folder -- checkpoints, LoRAs, VAEs, all of it.

---

**"Load a basic txt2img workflow"**

Loads a starter workflow template. The agent will describe what's in it.

---

**"Change the steps to 30 and the CFG to 7"**

Makes a surgical edit to the loaded workflow. It shows you the change before applying it.

---

**"Find me a good anime LoRA on CivitAI"**

Searches CivitAI for trending anime LoRAs and tells you which ones are popular and compatible with your models.

---

**"Run this workflow and tell me what you think of the output"**

Executes the workflow, waits for the render, then uses vision analysis to comment on the result -- quality, artifacts, composition.

---

## 7. Using with Claude Code (Power Users)

Claude Code is a terminal app where Claude can use all 77 agent tools **and** write code at the same time. It's the best way to use the agent.

1. Install [Claude Code](https://docs.anthropic.com/en/docs/claude-code)
2. Open a terminal in the `comfyui-agent` folder
3. Run:

```bash
claude
```

That's it. The MCP tools are already configured in `.claude/settings.json`. Claude Code sees all 77 tools automatically.

> **Pro tip:** In Claude Code mode, you can do things like "write a Python script that generates 10 variations of this prompt and runs them all" -- it combines coding ability with ComfyUI control.

---

## 8. Troubleshooting

### "ANTHROPIC_API_KEY not set"

**Why:** The `.env` file is missing or doesn't have your key.

**Fix:** Make sure you're running the agent from inside the `comfyui-agent` folder. Check that `.env` exists and contains `ANTHROPIC_API_KEY=sk-ant-your-actual-key`.

---

### "Could not connect to ComfyUI"

**Why:** ComfyUI isn't running, or it's on a different port.

**Fix:** Start ComfyUI first. Verify you can open `http://127.0.0.1:8188` in a browser. If your ComfyUI uses a different port, set `COMFYUI_PORT` in your `.env` file.

---

### "pip is not recognized"

**Why:** Python's package manager isn't on your system PATH.

**Fix:** Try `python -m pip install -e .` instead. If that doesn't work, reinstall Python and check "Add Python to PATH" during install.

---

### "python is not recognized"

**Why:** Python isn't installed or isn't on your PATH.

**Fix:** Install Python from [python.org/downloads](https://www.python.org/downloads/). On Windows, make sure to check "Add Python to PATH" during the install.

---

### "The agent is slow" or "it costs too much"

**Why:** The default model (Sonnet) balances speed and cost. Opus is slower and more expensive.

**Fix:** Check your `.env` file. If `AGENT_MODEL` is set to an Opus model, switch to `claude-sonnet-4-20250514` for faster, cheaper sessions. In Claude Code mode, the model is controlled by Claude Code itself, not the `.env` file.

---

Still stuck? Ask the coworker who sent you this link. Or [open an issue on GitHub](https://github.com/JosephOIbrahim/comfyui-agent/issues).

---

## 9. Uninstall

Changed your mind? No hard feelings.

```bash
pip uninstall comfyui-agent
```

Then delete the `comfyui-agent` folder. That's it. It didn't touch your ComfyUI installation.

---

You're set. Go make something cool.
