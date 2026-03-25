# How to Install the Video Recreation Agent
## Plain English. No Jargon. Just Do This.

---

## What We're Building

You're going to connect **3 things** that will work together:

```
  YOU (type a command)
    ↓
  CLAUDE CODE (the brain — decides what to do)
    ↓
  COMFYUI (the artist — generates the images and video)
```

When it's all connected, you'll type something like:

> "Recreate this video: [paste a URL]"

...and the system will analyze the video, generate new frames,
assemble them into a new video, and show you a comparison.

---

## Before You Start — What You Need

**Things you probably already have:**

✅ ComfyUI installed somewhere on your computer
✅ Your `comfyui-agent` repo cloned somewhere on your computer
✅ Python 3.10 or newer
✅ An Anthropic API key

**Things you might need to install:**

❓ Claude Code (the terminal AI tool)
❓ ffmpeg (the video swiss-army knife)

---

## Step 1: Install ffmpeg

ffmpeg is the tool that cuts, joins, and analyzes video files.
The agent needs it to break apart reference videos and assemble the final result.

**On Windows:**

Open PowerShell and type:
```
winget install ffmpeg
```

Then close and reopen PowerShell.

**On Mac:**
```
brew install ffmpeg
```

**Check it worked:**
```
ffmpeg -version
```

You should see a version number. If you see "not found", restart your terminal.

---

## Step 2: Install Claude Code

Claude Code is the AI terminal tool that runs the agent team.
It reads the CLAUDE.md file and the slash commands to know what to do.

```
npm install -g @anthropic-ai/claude-code
```

If you don't have npm, install Node.js first from https://nodejs.org

**Check it worked:**
```
claude --version
```

---

## Step 3: Download the Agent Team Package

Download the zip file I gave you (`video-recreation-agent.zip`).

Unzip it somewhere you can find it. You'll see these files:

```
video-recreation-agent/
  CLAUDE.md                           ← The brain
  SETUP_GUIDE.md                      ← Detailed reference
  test_setup.py                       ← Tests if everything works
  config/
    bridge.env                        ← Your settings (edit this)
  agent/
    tools/
      comfyui_bridge.py               ← Talks to ComfyUI
  .claude/
    commands/
      analyze-video.md                ← Expert 1: Analyzes video
      build-workflow.md               ← Expert 2: Builds workflows
      generate-shots.md               ← Expert 3: Runs generation
      assemble-montage.md             ← Expert 4: Edits video
      qa-compare.md                   ← Expert 5: Quality check
```

---

## Step 4: Copy Files Into Your comfyui-agent Repo

Open your `comfyui-agent` folder. You're going to drop files into it.

**Copy these files/folders:**

| From the zip                        | Into your comfyui-agent repo         |
|-------------------------------------|--------------------------------------|
| `.claude/commands/` (whole folder)  | `comfyui-agent/.claude/commands/`    |
| `agent/tools/comfyui_bridge.py`     | `comfyui-agent/agent/tools/`         |
| `config/bridge.env`                 | `comfyui-agent/config/`              |
| `test_setup.py`                     | `comfyui-agent/`                     |

**For the CLAUDE.md:**
Don't replace your existing CLAUDE.md. Instead, open both files side by side
and **paste the content from the new CLAUDE.md at the bottom of your existing one.**
This way you keep your agent's original abilities AND add the video team.

---

## Step 5: Create the Workspace Folders

The agent needs folders to store its work. Open a terminal in your comfyui-agent folder:

**Windows PowerShell:**
```powershell
mkdir workspace
mkdir workspace\reference
mkdir workspace\keyframes
mkdir workspace\workflows
mkdir workspace\outputs
mkdir workspace\qa
mkdir workspace\qa\frames
```

**Mac/Linux:**
```bash
mkdir -p workspace/{reference,keyframes,workflows,outputs,qa/frames}
```

---

## Step 6: Edit Your Settings

Open `config/bridge.env` in any text editor (Notepad, VS Code, whatever).

Change these lines to match YOUR computer:

```
# Where ComfyUI lives — find the folder that has main.py in it
COMFYUI_PATH=G:/COMFYUI_Database

# Leave these alone unless you changed ComfyUI's port
COMFYUI_HOST=127.0.0.1
COMFYUI_PORT=8188

# Where your comfyui-agent repo is
AGENT_PATH=C:/Users/YourName/code/comfyui-agent

# Where generated stuff goes (the workspace folder you just made)
WORKSPACE=C:/Users/YourName/code/comfyui-agent/workspace

# Your Anthropic API key
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Save the file.

---

## Step 7: Install Kling 3.0 in ComfyUI (For Video Generation)

The agent uses Kling 3.0 to generate video clips. Kling runs on their cloud
(not your GPU), so it works even if your GPU is busy with other stuff.

1. **Open ComfyUI** in your browser (usually http://127.0.0.1:8188)
2. Click the **Manager** button (or go to Template Library)
3. Search for **"Kling 3.0"**
4. **Install** the Kling partner nodes
5. **Restart ComfyUI** when it asks

You'll also need a Kling API account. The agent will tell you if it's missing.

**Don't want to use Kling?** That's fine. You can swap it for:
- **Wan 2.2** (free, runs locally, needs lots of VRAM)
- **LTX-2** (cloud API, different provider)

---

## Step 8: Test That Everything Works

Make sure ComfyUI is running first. Then:

1. Open a terminal
2. Navigate to your comfyui-agent folder:
   ```
   cd C:\Users\YourName\code\comfyui-agent
   ```
3. Load your settings:

   **Windows PowerShell:**
   ```powershell
   Get-Content config/bridge.env | ForEach-Object {
       if ($_ -match '^([^#]\w+)=(.*)$') {
           [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
       }
   }
   ```

   **Mac/Linux:**
   ```bash
   set -a; source config/bridge.env; set +a
   ```

4. Run the test:
   ```
   python test_setup.py
   ```

**You want to see all checkmarks ✓:**
```
  ✓ ffmpeg — installed
  ✓ ComfyUI reachable — at http://127.0.0.1:8188
  ✓ GPU detected — NVIDIA GeForce RTX 4090 (24GB)
  ✓ Node schemas loaded — 847 types
  ✓ Checkpoints — 12 installed
  ...
  ALL CHECKS PASSED
```

**If something shows ✗:**
- **ffmpeg ✗** → Go back to Step 1
- **ComfyUI ✗** → Make sure ComfyUI is running, check your port
- **Kling nodes ✗** → Go back to Step 7

---

## Step 9: Launch It!

1. Make sure ComfyUI is running
2. Open a terminal in your comfyui-agent folder
3. Load your settings (same command from Step 8)
4. Start Claude Code:
   ```
   claude
   ```

Claude Code will read your CLAUDE.md and the slash commands automatically.

---

## Step 10: Try It Out

**Full auto mode** — paste this into Claude Code:
```
Recreate this video: [paste a video URL here]
```

**Step-by-step mode** — use slash commands one at a time:
```
/project:analyze-video [video URL]
```
Wait, review the storyboard, then:
```
/project:build-workflow
```
Wait, review the plan, then:
```
/project:generate-shots
```
Wait for generation to finish, then:
```
/project:assemble-montage
```
Watch the result, then:
```
/project:qa-compare
```

---

## Quick Reference Card

| What You Want To Do | What To Type |
|---|---|
| Start the agent | `claude` (from your repo folder) |
| Analyze a video | `/project:analyze-video [url]` |
| Build the workflows | `/project:build-workflow` |
| Generate the shots | `/project:generate-shots` |
| Assemble the video | `/project:assemble-montage` |
| Compare quality | `/project:qa-compare` |
| Do everything at once | "Recreate this video: [url]" |

---

## If Something Goes Wrong

**"Command not found: claude"**
→ Run `npm install -g @anthropic-ai/claude-code` again
→ Close and reopen your terminal

**"ComfyUI not connected"**
→ Is ComfyUI running? Open http://127.0.0.1:8188 in your browser
→ If you see the ComfyUI interface, it's running. Try the test again.

**"ffmpeg not found"**
→ Close your terminal, reopen it, try `ffmpeg -version`
→ If still nothing, reinstall ffmpeg

**"Node type not found"**
→ You're missing a custom node in ComfyUI
→ The agent will tell you which one. Install it via ComfyUI Manager.

**"Out of memory"**
→ For local models: the agent will retry at lower resolution
→ For API models (Kling): shouldn't happen, it runs on their cloud

**"I don't see the slash commands"**
→ Make sure the `.claude/commands/` folder is in your repo root
→ Make sure you're running `claude` from inside your repo folder

---

## What Each Expert Does (Plain English)

**ANALYST** — Watches the reference video and takes notes.
"This is a 2-second CCTV shot, then a 2-second cinematic close-up, then..."

**ARCHITECT** — Draws up the blueprint.
"For the CCTV shot, I need these ComfyUI nodes wired together like this..."

**GENERATOR** — Runs the machines.
"Sending the blueprint to ComfyUI, watching the progress bar..."

**EDITOR** — Cuts it all together.
"Taking 6 clips, joining them in order, adding the audio back..."

**QA** — Watches both videos side by side.
"The original looks like this, yours looks like that. Score: 7/10."

---

That's it. You're set up. Go recreate some videos.
