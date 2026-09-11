"""Run separately: uvicorn runtime.whisper_worker:app --host 127.0.0.1 --port 8011.

Accepts only mono 16kHz signed little-endian PCM. Never stores audio or text.
One inference at a time; admission is fail-fast rather than an unbounded queue.
"""

import asyncio
import hmac
import os
import re
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, Request

MAX_AUDIO_BYTES = 120 * 16000 * 2
LANGUAGE_RE = re.compile(r"^[a-z]{2,3}$")


def create_app(model_factory=None, token=None):
    token = token if token is not None else os.environ.get("VOICE_RUNTIME_WHISPER_TOKEN", "")

    @asynccontextmanager
    async def lifespan(app):
        if not token:
            raise RuntimeError("VOICE_RUNTIME_WHISPER_TOKEN is required")
        if model_factory is None:
            from faster_whisper import WhisperModel

            app.state.model = await asyncio.to_thread(
                WhisperModel, os.environ.get("WHISPER_MODEL", "small"),
                device=os.environ.get("WHISPER_DEVICE", "cpu"),
                compute_type=os.environ.get("WHISPER_COMPUTE_TYPE", "int8"),
                cpu_threads=int(os.environ.get("WHISPER_CPU_THREADS", "4")),
                num_workers=1,
            )
        else:
            app.state.model = model_factory()
        app.state.busy = False
        yield

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ready": hasattr(app.state, "model"), "busy": app.state.busy}

    @app.post("/transcribe")
    async def transcribe(request: Request):
        if not token or not hmac.compare_digest(
            request.headers.get("authorization", ""), "Bearer " + token
        ):
            raise HTTPException(401, "Unauthorized")
        language = request.query_params.get("language")
        if language is not None and not LANGUAGE_RE.fullmatch(language):
            raise HTTPException(400, "Invalid language")
        if app.state.busy:
            raise HTTPException(503, "Whisper worker busy")
        app.state.busy = True
        inference = None
        try:
            audio = bytearray()
            async with asyncio.timeout(10):
                async for chunk in request.stream():
                    audio.extend(chunk)
                    if len(audio) > MAX_AUDIO_BYTES:
                        raise HTTPException(413, "Audio exceeds 120 seconds")
            if not audio or len(audio) % 2:
                raise HTTPException(400, "Expected mono PCM16 at 16000 Hz")

            def run():
                samples = np.frombuffer(audio, dtype="<i2").astype(np.float32) / 32768
                # Auto-detection on a short clip can pick the wrong language and
                # invent text in it; the session already knows its language.
                segments, _ = app.state.model.transcribe(
                    samples, beam_size=1, vad_filter=True,
                    condition_on_previous_text=False, language=language,
                )
                return " ".join(segment.text.strip() for segment in segments).strip()

            inference = asyncio.create_task(asyncio.to_thread(run))
            # Cancellation cannot kill a native inference thread. Keep the
            # admission slot held until that thread really finishes.
            text = await asyncio.shield(inference)
            return {"text": text}
        except TimeoutError:
            raise HTTPException(408, "Audio upload timed out") from None
        finally:
            if inference is not None and not inference.done():
                def release(task):
                    app.state.busy = False
                    if not task.cancelled():
                        task.exception()
                inference.add_done_callback(release)
            else:
                app.state.busy = False

    return app


app = create_app()
