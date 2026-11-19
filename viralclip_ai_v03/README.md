# ViralClip AI v0.3 — multimodal-ready MVP

## What's improved
- Speech transcription via Faster-Whisper.
- Audio-energy peak detection (RMS windows).
- Periodic and audio-peak-triggered video frame sampling.
- Optional visual reasoning through a configurable OpenAI-compatible vision endpoint.
- Combines speech, audio and visual candidate signals, merges nearby events, ranks candidates.
- Renders selected moments in 9:16 vertical and/or 16:9 landscape.

## Important limitations
- This is an engineering MVP, not a proven viral prediction model.
- Vision analysis is optional and requires a compatible multimodal endpoint plus API key/model settings. Without these settings, the app still uses transcript + audio peaks.
- Frame sampling is sparse; it can miss brief events. Vision results are only as good as the selected provider/model.
- Vertical video is center-cropped, not yet speaker-aware or gameplay-aware.
- The MVP accepts authorized uploaded video files only. It does not download YouTube videos. A video URL does not grant rights to extract its audiovisual stream.
- No login, user isolation, rate limits, billing, automatic data expiry or production job queue yet. Do not deploy as a public multi-user service until those protections are added.

## Run locally
Requires Python 3.10–3.12 and FFmpeg/FFprobe installed.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000

The first run downloads the Whisper model. Use `WHISPER_MODEL=tiny` for lighter CPU usage or `small` for better transcription at increased cost.

## Optional visual AI provider
Set all three environment variables:
- `VISION_API_BASE_URL` — base URL of an OpenAI-compatible chat-completions API (without `/chat/completions`)
- `VISION_API_KEY`
- `VISION_MODEL`

The server sends small batches of sampled JPEG frames to this provider. Review provider privacy, retention and pricing before use. Keep API keys on the server; never put them in browser JavaScript.

## Docker
```bash
docker build -t viralclip-ai .
docker run --rm -p 8000:8000 -v viralclip-data:/var/data viralclip-ai
```

## Online deployment preparation
`render.yaml` and `Dockerfile` provide a deployment starting point. The app is not yet safe for public multi-user launch: authentication, per-user job access, upload quotas, private storage, cleanup/expiry, queue workers and monitoring must be added. Hosting also requires your own hosting account and payment/plan selection. Do not share API keys in chat.

## Planned next
1. Secure accounts and private jobs.
2. Durable worker queue and progress updates.
3. Better event classification and confidence calibration.
4. Smart vertical reframing and subtitles.
5. YouTube creator-authorized workflows / approved integrations.
