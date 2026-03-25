# /project:assemble-montage — EDITOR Expert

You are the **EDITOR** expert in the Video Recreation Agent Team.

Your job: Take the generated video segments and assemble them into a final montage using ffmpeg.

## Prerequisites
- Generated clips in `workspace/outputs/`
- Storyboard in `workspace/storyboard.json` (for ordering and timing)

## Steps

### 1. Verify all clips exist and probe their properties
```bash
for clip in workspace/outputs/seg_*.mp4; do
  echo "=== $(basename $clip) ==="
  ffprobe -v quiet -print_format json -show_format -show_streams "$clip" | \
    python3 -c "import json,sys; d=json.load(sys.stdin); s=d['streams'][0]; \
    print(f'  Duration: {d[\"format\"][\"duration\"]}s'); \
    print(f'  Resolution: {s[\"width\"]}x{s[\"height\"]}'); \
    print(f'  FPS: {s.get(\"r_frame_rate\", \"?\")}'); \
    print(f'  Codec: {s[\"codec_name\"]}')"
done
```

### 2. Normalize clips (match resolution, fps, pixel format)
```bash
# Normalize all clips to consistent format before concat
for clip in workspace/outputs/seg_*.mp4; do
  out="workspace/outputs/norm_$(basename $clip)"
  ffmpeg -i "$clip" \
    -vf "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:-1:-1,fps=24" \
    -c:v libx264 -preset fast -crf 18 \
    -an \
    "$out"
done
```

### 3. Build concat file list
```bash
# Create ordered file list from storyboard
python3 -c "
import json
sb = json.load(open('workspace/storyboard.json'))
with open('workspace/outputs/concat_list.txt', 'w') as f:
    for seg in sorted(sb['segments'], key=lambda s: s['id']):
        f.write(f\"file 'norm_seg_{seg['id']:02d}.mp4'\n\")
print('Concat list written')
"
cat workspace/outputs/concat_list.txt
```

### 4. Assemble with ffmpeg
```bash
# Simple concatenation (hard cuts matching reference)
cd workspace/outputs
ffmpeg -f concat -safe 0 -i concat_list.txt \
  -c:v libx264 -preset medium -crf 18 \
  -pix_fmt yuv420p \
  ../montage_v1.mp4

# Verify output
ffprobe -v quiet -print_format json -show_format ../montage_v1.mp4 | \
  python3 -c "import json,sys; d=json.load(sys.stdin); \
  print(f'Montage: {d[\"format\"][\"duration\"]}s')"
cd ../..
```

### 5. Mix reference audio (if available)
```bash
# Extract audio from reference
ffmpeg -i workspace/reference/input_video.mp4 \
  -vn -acodec aac -b:a 192k \
  workspace/reference/ref_audio.aac

# Mux audio onto montage
ffmpeg -i workspace/montage_v1.mp4 \
  -i workspace/reference/ref_audio.aac \
  -c:v copy -c:a aac -b:a 192k \
  -map 0:v -map 1:a \
  -shortest \
  workspace/montage_v1_audio.mp4
```

### 6. Apply style overlays (for CCTV segments etc.)
```bash
# Example: Add scanlines + timestamp overlay to CCTV segments
# This would be done per-segment BEFORE concat if needed
ffmpeg -i seg_01.mp4 \
  -vf "drawtext=text='2026-01-14 22\:03\:07 CAM 03':fontsize=24:fontcolor=white:x=10:y=h-40,\
       noise=alls=20:allf=t+u" \
  -c:v libx264 -crf 18 \
  seg_01_styled.mp4
```

### 7. Show result summary
```
╔══════════════════════════════════════════════╗
║  MONTAGE ASSEMBLED                           ║
╠══════════════════════════════════════════════╣
║  Output: workspace/montage_v1.mp4            ║
║  Duration: 8.3s                              ║
║  Resolution: 1080x1920                       ║
║  Segments: 6 (hard cuts)                     ║
║  Audio: Reference audio mixed ✓              ║
║  Size: ~12MB                                 ║
╚══════════════════════════════════════════════╝

Open it and check if the beats land right!
```

Ask: **"Montage assembled. Want me to run QA comparison against the reference?"**
