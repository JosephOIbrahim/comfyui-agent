# Audio Generation Workflows in ComfyUI

## Text-to-Speech (TTS)

ComfyUI supports TTS through several custom node packs.

### CosyVoice Pipeline
1. **CosyVoiceLoader** -- loads CosyVoice model
2. **CosyVoiceGenerate** -- generates speech from text
3. **SaveAudio** -- exports audio file

### Key Parameters
- **text**: the text to speak
- **voice/speaker**: voice selection or reference audio
- **speed**: playback speed multiplier (0.5-2.0)
- **sample_rate**: typically 22050 or 44100 Hz

## Audio Processing Nodes

### Common Node Packs
- **ComfyUI-CosyVoice**: CosyVoice TTS integration
- **ComfyUI-AudioTools**: audio processing utilities
- **ComfyUI-Qwen-Audio**: Qwen audio model integration

### Output Formats
- `.wav` -- uncompressed, highest quality
- `.mp3` -- compressed, smaller files
- `.flac` -- lossless compression

## Multi-Modal Pipelines

### Image + Audio Composition
1. Generate image with standard txt2img pipeline
2. Generate narration with TTS pipeline
3. Combine with video compositor nodes

### Video + Audio
1. Generate video with Wan2.1 or AnimateDiff
2. Generate matching audio/narration
3. Mux together with AudioVideoMux nodes

## Model Directory Conventions
- TTS models: `models/TTS/` or `models/audio/`
- Audio models often use directory-based HuggingFace repos
- Multiple files per model (config.json, model.safetensors, tokenizer files)

## Tips
- Audio generation is typically fast (seconds, not minutes)
- TTS quality depends heavily on the reference audio for voice cloning
- For pipeline chaining, audio files are saved to ComfyUI's output directory
- Check sample rate compatibility when combining audio from different sources
