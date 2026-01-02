#!/usr/bin/env python3
"""GLM-ASR FastAPI server for audio transcription."""

import os
import tempfile
import typing
from pathlib import Path
from contextlib import asynccontextmanager

from loguru import logger
import torch
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import PlainTextResponse

from transcribers import get_transcriber, func_time, context_time

transcriber = get_transcriber()

@asynccontextmanager
async def lifespan(app: FastAPI) -> None:
    transcriber.load()
    yield
    torch.cuda.empty_cache()
    logger.info("Model unloaded")


app = FastAPI(lifespan=lifespan)


@func_time
def save_uploadfile(uploadfile: UploadFile, out_dir: Path) -> Path:
    file_path = out_dir / uploadfile.filename
    file_path.write_bytes(uploadfile.file.read())
    logger.info(f"uploadfile saved to {file_path}")
    return file_path


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: typing.Annotated[str, Form()] = None,  # whisper-1
    language: typing.Annotated[str, Form()] = None,  # None for auto detect
    response_format: typing.Annotated[str, Form()] = None,  # json, text, srt, vtt
):
    """Transcribe audio file to text."""
    logger.info(f"{transcriber.__class__.__name__} transcribing {type(file)} {file.filename} model {model} -> language {language} format {response_format}")

    response_format = response_format or "json"

    if response_format == "json":
        format = "text"  # json == {"text": text}
    else:
        format = response_format  # text, srt

    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = save_uploadfile(file, Path(temp_dir))
        # format: text, srt
        text = transcriber.transcribe(file_path, language=language, format=format)

    if response_format == "json":
        return {"text": text}
    else:
        return PlainTextResponse(text)


if __name__ == "__main__":
    import uvicorn
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
