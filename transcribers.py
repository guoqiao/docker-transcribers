#!/usr/bin/env python3
"""ASR: Automatic Speech Recognition."""

from contextlib import contextmanager
from functools import wraps
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from loguru import logger


def sec2ts(t: float|str) -> str:
    # 65.234 -> 00:01:05.234
    f = float(t)
    n = int(t)
    ms = int((f - n) * 1000)  # 0.234 -> 234
    h, s = divmod(n, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class StopWatch:
    def __init__(self, task):
        self.task = task
        self.t0 = time.time()
        logger.debug(f"[0.0s]task start: {task}")

    def stop(self):
        self.span = time.time() - self.t0
        logger.debug(f"[{self.span:.1f}s]task stop: {self.task}")


@contextmanager
def context_time(task):
    """Track time for a context using yield.

    Usage:
        with context_time("generating mesh") as sw:
            # do your task
        print(sw.span)
    """
    sw = StopWatch(task)
    try:
        # This is where the code block will execute
        yield sw
    finally:
        sw.stop()


def func_time(func):
    """Decorator to get func exec time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with context_time(f"{func.__module__}.{func.__qualname__}"):
            return func(*args, **kwargs)

    return wrapper


def run_cmd(cmd, check=True, text=True, cwd=None, **kwargs):
    if isinstance(cmd, str):
        cmdline = cmd
        shell = True
    else:
        # ensure each arg is a string
        cmd = [str(arg) for arg in cmd if arg]
        cmdline = shlex.join(cmd)
        shell = False

    if cwd:
        cmdline = f"cd {cwd} && {cmdline}"

    with context_time(cmdline):
        return subprocess.run(cmd, shell=shell, check=check, text=text, cwd=cwd, **kwargs)


def get_cmd_output(cmd):
    return run_cmd(cmd, capture_output=True).stdout.strip()


def run_script(script_path, *args):
    assert os.path.exists(script_path), f"script not found: {script_path}"
    return run_cmd([script_path, *args])


def ffmpeg_extract_audio(file: Path, ext: str = "opus") -> Path:
    suffix = f".{ext}"
    if file.suffix.lower() == suffix:
        return file
    script = f"scripts/ffmpeg_extract_{ext}.sh"
    run_script(script, str(file), file.stem)
    out_file = file.with_suffix(suffix)
    assert out_file.exists()
    return out_file


class Transcriber:
    supported_formats = ["text", "txt"]

    @func_time
    def load(self):
        logger.info(f"{self.__class__.__name__} loading ...")
        self._load()

    def _load(self):
        # for time consuming operations, e.g. loading model, downloading, etc.
        pass

    def clean_file(self, file: str) -> str:
        # check file data type, ext, etc.
        return ffmpeg_extract_audio(file)

    def clean_format(self, format: str) -> bool:
        if self.supported_formats:
            if format not in self.supported_formats:
                raise ValueError(f"requested format '{format}' not in {self.supported_formats}")
        return format

    def clean_language(self, language: str) -> str:
        # check supported langs, map lang code, etc.
        return language

    @func_time
    def transcribe(self, file, language: str = None, format: str = "text") -> str:
        file = self.clean_file(file)
        language = self.clean_language(language)
        format = self.clean_format(format)
        with context_time(f"{self.__class__.__name__}._transcribe"):
            return self._transcribe(file, language=language, format=format)


class AssemblyAITranscriber(Transcriber):

    supported_formats = ["text", "srt", "vtt"]

    def __init__(self):
        import assemblyai as aai
        aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

    def _transcribe(self, file: str, language: str = None, format: str = "text") -> str:
        import assemblyai as aai

        if language:
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.universal,
                language_code=language,
                punctuate=True,
            )
        else:
            options = aai.LanguageDetectionOptions(expected_languages=["zh", "en"])
            config = aai.TranscriptionConfig(
                speech_model=aai.SpeechModel.universal,
                language_detection=True,
                language_detection_options=options,
                punctuate=True,
            )
        transcript_obj: aai.transcriber.Transcript = aai.Transcriber(
            config=config
        ).transcribe(str(file))
        if transcript_obj.status == "error":
            raise RuntimeError(f"AssemblyAI transcribe failed for {file}: {transcript_obj.error}")

        if format in ["text", "txt"]:
            return transcript_obj.text
        elif format in ["srt"]:
            return transcript_obj.export_subtitles_srt()
        elif format in ["vtt"]:
            return transcript_obj.export_subtitles_vtt()


class OpenAITranscriber(Transcriber):

    supported_formats = ["txt", "text", "srt"]

    def __init__(self, api_key: str = None, base_url: str = None):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("OPENAI_MODEL", "whisper-1")
        # user can request these formats for this transcriber

    def call_api(self, file: str, language: str = None, format: str = "text"):
        if format == "txt":
            format = "text"
        logger.info(f"calling {self.__class__.__name__} with model {self.model}, format {format}, language {language}")
        ret = self.client.audio.transcriptions.create(
            file=open(file, "rb"),
            model=self.model,
            language=language,
            response_format=format,
        )
        logger.info(f"openai api transcribe ret type: {type(ret)}")
        return ret

    def _transcribe(self, file: str, language: str = None, format: str = "text") -> str:

        ret = self.call_api(file, language=language, format=format)
        if format == "json":
            # for 'json', ret is a Transcript obj
            return ret.text
        elif format in ["text", "srt", "vtt"]:
            # ret is a json encoded str
            return json.loads(ret)
        else:
            raise ValueError(f"Unsupported format: {format}")


class LemonfoxAITranscriber(OpenAITranscriber):

    def __init__(self):
        super().__init__(
            api_key=os.environ["LEMONFOX_AI_API_KEY"],
            base_url="https://api.lemonfox.ai/v1",
        )


class GLMASRTranscriber(Transcriber):

    supported_formats = ["text"]
    model_id = "zai-org/GLM-ASR-Nano-2512"
    max_new_tokens = 500

    def _load(self):
        from transformers import AutoModelForSeq2SeqLM, AutoProcessor
        logger.info(f"Loading model {self.model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, dtype="auto", device_map="auto")

    def _transcribe(self, file: str|Path, language: str = None, format: str = "text"):
        # TODO: Audio duration (2983.4s) exceeds 655s; truncating to first 655s.
        # audio: path, ndarray, torch.Tensor
        inputs = self.processor.apply_transcription_request(str(file))
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        outputs = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )
        decoded_outputs = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return decoded_outputs[0]


class WhisperTranscriber(Transcriber):
    supported_formats = ["json", "text", "srt"]
    model_size = os.getenv("WHISPER_MODEL", "turbo")

    def _load(self):
        import whisper
        self.model = whisper.load_model(self.model_size)

    def _transcribe(self, audio, language: str = None, format: str = "text") -> str:
        # audio: Union[str, np.ndarray, torch.Tensor]
        result = self.model.transcribe(audio)
        if format in ["text", "txt"]:
            return result.get("text", "")

        segments = result.get("segments", [])
        lines = []
        if format in ["srt"]:
            for i, seg in enumerate(segments, start=1):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                parts = [
                    str(i),
                    f"{sec2ts(start)} --> {sec2ts(end)}",
                    seg.get("text", "").strip(),
                ]
                caption = "\n".join(parts).strip()
                lines.append(caption)
            return "\n\n".join(lines).strip()


class WhisperCPUTranscriber(WhisperTranscriber):

    def _load(self):
        from whisper import Whisper
        self.model = Whisper(
            model=os.getenv("WHISPER_MODEL", "small"),
            device="cpu",
            compute_type="int8",
        )


class FasterWhisperTranscriber(Transcriber):

    supported_formats = ["text", "srt"]

    device = "cuda"
    compute_type = "float16"
    batch_size = 16
    beam_size = 5
    best_of = 5
    model_size = os.getenv("FASTER_WHISPER_MODEL_SIZE", "large-v3")

    def _load(self):
        from faster_whisper import WhisperModel, BatchedInferencePipeline
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        self.pipeline = BatchedInferencePipeline(model=self.model)

    def _transcribe(self, file: str, language: str = None, format: str = "text") -> str:

        logger.info(f"transcibe with faster-whisper {self.model}: {file}")
        segments, info = self.pipeline.transcribe(
            file,
            language=language or None,
            batch_size=self.batch_size,
            log_progress=False,  # a progress bar, not text
            beam_size=self.beam_size,
            best_of=self.best_of,
            chunk_length=10,
        )

        lines = []
        if format in ["text", "txt"]:
            for seg in segments:
                lines.append(seg.text.strip())
        elif format in ["srt"]:
            for i, seg in enumerate(segments, start=1):
                parts = [
                    str(i),
                    f"{sec2ts(seg.start)} --> {sec2ts(seg.end)}",
                    seg.text.strip(),
                ]
                caption = "\n".join(parts) + "\n"
                lines.append(caption)

        return "\n".join(lines).strip()


class FasterWhisperCPUTranscriber(FasterWhisperTranscriber):
    device = "cpu"
    compute_type = "int8"
    batch_size = 8
    beam_size = 1
    best_of = 1
    model_size = os.getenv("FASTER_WHISPER_MODEL_SIZE_CPU", "small")


def get_transcriber(backend: str = "") -> Transcriber:
    backend = backend or os.getenv("TRANSCRIBER_BACKEND", "glm-asr")
    if backend in ["lemonfoxai", "lemonfox", "lemonfox-ai"]:
        return LemonfoxAITranscriber()
    if backend in ["assembly", "assemblyai", "aai"]:
        return AssemblyAITranscriber()
    if backend in ["glm", "glmasr", "glmasr-server", "glm-asr", "glm-asr-server"]:
        return GLMASRTranscriber()
    elif backend == "openai":
        return OpenAITranscriber()
    elif backend in ["whisper", "whisper-gpu"]:
        return WhisperTranscriber()
    elif backend in ["whisper-cpu"]:
        return WhisperCPUTranscriber()
    elif backend in ["faster-whisper", "faster-whisper-gpu"]:
        return FasterWhisperTranscriber()
    elif backend in ["faster-whisper-cpu"]:
        return FasterWhisperCPUTranscriber()


def transcribe(file: str, language: str = None, format: str = "text", backend: str = "") -> str:
    transcriber = get_transcriber(backend)
    transcriber.load()
    return transcriber.transcribe(file, language=language, format=format)


def cli():
    import argparse

    parser = argparse.ArgumentParser(
        description="Transcribe audio file to text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("file", help="audio file path")
    parser.add_argument("-l", "--language", help="language code, None|zh|en|etc.")
    parser.add_argument("-f", "--format", default="text", help="output format, text|srt|vtt")
    parser.add_argument("-b", "--backend", help="transcribe backend name")
    parser.add_argument("-o", "--output", help="output file path")
    args = parser.parse_args()
    text = transcribe(args.file, language=args.language, format=args.format, backend=args.backend)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        logger.info(f"transcript saved to {output_path}")
    else:
        print(text)


if __name__ == "__main__":
    cli()
