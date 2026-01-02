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
def save_uploadfile(uploadfile: UploadFile) -> str:
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / uploadfile.filename
    temp_file.write_bytes(uploadfile.file.read())
    logger.info(f"upload file saved to {temp_file}")
    return temp_file


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: typing.Annotated[str, Form()] = "whisper-1",  # placeholder
    language: typing.Annotated[str, Form()] = None,
    response_format: typing.Annotated[str, Form()] = "text",  # text, srt, vtt
):
    """Transcribe audio file to text."""
    logger.info(f"{transcriber.__class__.__name__} transcribing {type(file)} {file.filename} -> language {language} format {response_format}")

    # load can accept a file-like object
    # file.file: SpooledTemporaryFile
    # supported formats: wav|mp3|flac|ogg
    # sr=None to keep orig sample rate
    # audio_ndarray, sr = librosa.load(file.file, sr=16000)
    file_path = save_uploadfile(file)
    # path/ndarray/torch.Tensor
    return transcriber.transcribe(file_path, language=language, format=response_format)


if __name__ == "__main__":
    import uvicorn
    port = os.getenv("PORT", 8000)
    uvicorn.run(app, host="0.0.0.0", port=int(port))
