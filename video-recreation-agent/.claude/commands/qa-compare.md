# /project:qa-compare — QA Expert

You are the **QA** expert in the Video Recreation Agent Team.

Your job: Compare the recreated montage against the reference video frame-by-frame.

## Prerequisites
- Reference video: `workspace/reference/input_video.mp4`
- Recreation: `workspace/montage_v1.mp4`
- Storyboard: `workspace/storyboard.json`

## Steps

### 1. Extract comparison frames at matching timestamps
```bash
mkdir -p workspace/qa/frames

# For each segment boundary, extract frames from both videos
python3 -c "
import json, subprocess
sb = json.load(open('workspace/storyboard.json'))
for seg in sb['segments']:
    mid = (seg['start'] + seg['end']) / 2  # midpoint of segment
    sid = seg['id']
    
    # Reference frame
    subprocess.run(['ffmpeg', '-ss', str(mid), '-i', 'workspace/reference/input_video.mp4',
                    '-frames:v', '1', f'workspace/qa/frames/ref_seg{sid:02d}.png', '-y'],
                   capture_output=True)
    
    # Recreation frame
    subprocess.run(['ffmpeg', '-ss', str(mid), '-i', 'workspace/montage_v1.mp4',
                    '-frames:v', '1', f'workspace/qa/frames/rec_seg{sid:02d}.png', '-y'],
                   capture_output=True)
    
    print(f'Segment {sid}: extracted comparison frames at t={mid:.1f}s')
"
```

### 2. Create side-by-side comparison images
```bash
# For each segment, create a 3-panel comparison (Original | Recreation | Diff)
for i in $(seq -w 1 6); do
  ffmpeg -y \
    -i workspace/qa/frames/ref_seg${i}.png \
    -i workspace/qa/frames/rec_seg${i}.png \
    -filter_complex "
      [0:v]scale=360:-1,drawtext=text='Original':fontsize=20:fontcolor=white:x=10:y=10[left];
      [1:v]scale=360:-1,drawtext=text='Recreation':fontsize=20:fontcolor=white:x=10:y=10[right];
      [left][right]hstack[out]
    " \
    -map "[out]" \
    workspace/qa/compare_seg${i}.png
done
```

### 3. Analyze each comparison with Vision
For each pair, send both frames to Claude Vision and evaluate:

**Scoring criteria (1-10 each):**
- **Composition:** Subject placement, framing, camera angle match
- **Style:** Color grade, lighting, visual effects match
- **Subject:** Character appearance, pose, clothing accuracy
- **Motion:** Body language, movement direction (inferred from pose)
- **Temporal:** Does this frame appear at the right moment in the sequence

### 4. Check temporal alignment
```bash
# Compare durations
python3 -c "
import subprocess, json

def get_duration(path):
    r = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', 
                        '-show_format', path], capture_output=True, text=True)
    return float(json.loads(r.stdout)['format']['duration'])

ref_dur = get_duration('workspace/reference/input_video.mp4')
rec_dur = get_duration('workspace/montage_v1.mp4')
diff = abs(ref_dur - rec_dur)

print(f'Reference: {ref_dur:.1f}s')
print(f'Recreation: {rec_dur:.1f}s')
print(f'Difference: {diff:.1f}s')
print(f'Match: {\"✓\" if diff < 0.5 else \"✗ — needs adjustment\"} ')
"
```

### 5. Generate QA report
Save to `workspace/qa_report.json` and display:

```
╔══════════════════════════════════════════════════════════╗
║  QA REPORT: montage_v1 vs reference                     ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  OVERALL SCORE: 7.2 / 10                                ║
║  Duration Match: 8.3s vs 8.3s ✓                         ║
║  Segment Count: 6/6 ✓                                   ║
║                                                          ║
║  PER-SEGMENT BREAKDOWN:                                  ║
║  ┌─────┬────────┬───────┬───────┬────────┬──────┬──────┐║
║  │ Seg │ Comp.  │ Style │ Subj. │ Motion │ Time │ AVG  │║
║  ├─────┼────────┼───────┼───────┼────────┼──────┼──────┤║
║  │  1  │  8     │  9    │  7    │  7     │  9   │ 8.0  │║
║  │  2  │  6     │  7    │  5    │  6     │  8   │ 6.4  │║
║  │  3  │  7     │  8    │  7    │  7     │  8   │ 7.4  │║
║  │  4  │  8     │  8    │  8    │  8     │  9   │ 8.2  │║
║  │  5  │  5     │  7    │  6    │  5     │  7   │ 6.0  │║
║  │  6  │  8     │  9    │  8    │  7     │  9   │ 8.2  │║
║  └─────┴────────┴───────┴───────┴────────┴──────┴──────┘║
║                                                          ║
║  RECOMMENDATIONS:                                        ║
║  • Seg 2: Re-generate with stronger pose reference       ║
║  • Seg 5: Adjust door animation timing                   ║
║  • Consider: Re-run lowest 2 segments, keep rest         ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
```

Ask: **"QA complete. Want me to re-generate the weakest segments, or is this version good enough to ship?"**
