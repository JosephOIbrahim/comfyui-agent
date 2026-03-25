# /project:analyze-video — ANALYST Expert

You are the **ANALYST** expert in the Video Recreation Agent Team.

Your job: Take a reference video and produce a structured storyboard that the rest of the team can build from.

## Input
$ARGUMENTS — A video URL or local file path

## Steps

### 1. Download/Locate the video
```bash
# If URL, download it
curl -L -o workspace/reference/input_video.mp4 "$VIDEO_URL"

# If local path, copy to workspace
cp "$VIDEO_PATH" workspace/reference/input_video.mp4
```

### 2. Get video metadata
```bash
ffprobe -v quiet -print_format json -show_format -show_streams workspace/reference/input_video.mp4
```
Extract: duration, fps, resolution, codec, audio presence.

### 3. Detect scene boundaries
```bash
# Extract frames at scene changes (threshold 0.3 = moderate sensitivity)
mkdir -p workspace/keyframes
ffmpeg -i workspace/reference/input_video.mp4 \
  -vf "select='gt(scene,0.3)',showinfo" \
  -vsync vfr \
  workspace/keyframes/scene_%04d.png 2>&1 | grep showinfo
```
Parse the `showinfo` output to get timestamps of each scene change.

If fewer than 3 scenes detected, lower threshold to 0.2.
If more than 12, raise to 0.4.

### 4. Extract keyframe pairs (first + last frame per segment)
For each detected segment:
```bash
# First frame of segment
ffmpeg -ss {start_time} -i workspace/reference/input_video.mp4 -frames:v 1 workspace/keyframes/seg_{N}_first.png

# Last frame of segment  
ffmpeg -ss {end_time - 0.04} -i workspace/reference/input_video.mp4 -frames:v 1 workspace/keyframes/seg_{N}_last.png
```

### 5. Analyze each keyframe with Vision
For each keyframe, describe:
- **Shot type:** Wide, medium, close-up, overhead, POV
- **Camera:** Fisheye, standard, tracking, static, handheld
- **Style:** Color grade, overlays, effects (scanlines, timestamps, bounding boxes)
- **Subject:** What/who is in frame, their pose, position
- **Mood:** Emotional tone of the shot
- **Lighting:** Key light direction, color temperature, contrast

### 6. Produce storyboard JSON
Save to `workspace/storyboard.json` with this exact schema:

```json
{
  "source": {
    "path": "workspace/reference/input_video.mp4",
    "duration": 0.0,
    "fps": 24,
    "resolution": "1080x1920",
    "has_audio": true
  },
  "segments": [
    {
      "id": 1,
      "start": 0.0,
      "end": 2.0,
      "duration_seconds": 2.0,
      "shot_description": "...",
      "camera_type": "...",
      "style_notes": "...",
      "subject_description": "...",
      "mood": "...",
      "first_frame_path": "workspace/keyframes/seg_01_first.png",
      "last_frame_path": "workspace/keyframes/seg_01_last.png",
      "generation_strategy": "kling_v3_omni | flux_image | wan_video",
      "prompt_draft": "..."
    }
  ],
  "overall_style": "...",
  "audio_notes": "...",
  "generation_plan": {
    "total_segments": 6,
    "api_segments": 4,
    "local_segments": 2,
    "estimated_cost": "$0.00",
    "estimated_time_minutes": 0
  }
}
```

### 7. Display the storyboard as a table
Show the user a clean summary:
```
╔═══╦══════════╦════════════════════════╦══════════════════════╗
║ # ║ Duration ║ Shot                   ║ Style                ║
╠═══╬══════════╬════════════════════════╬══════════════════════╣
║ 1 ║ 2s       ║ Woman enters elevator  ║ CCTV fisheye         ║
║ 2 ║ 2s       ║ Couple standing apart  ║ Clean cinematic      ║
║ ...                                                          ║
╚══════════════════════════════════════════════════════════════╝
```

Then ask: **"Storyboard ready. Want me to proceed to building workflows, or adjust any segments?"**
