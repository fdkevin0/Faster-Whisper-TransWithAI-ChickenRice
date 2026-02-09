#!/usr/bin/env python3
"""
Local inference pipeline backed by Transformers with optional Intel XPU runtime.
"""

import argparse
import logging
import math
import os
import sys
from collections import ChainMap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import librosa
import numpy as np
import pyjson5
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


logger = logging.getLogger(__name__)
log_handler = logging.StreamHandler()
log_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(log_handler)

DEFAULT_HF_MODEL_ID = "chickenrice0721/whisper-large-v2-translate-zh-v0.2-st"


@dataclass
class Segment:
    start: int  # ms
    end: int  # ms
    text: str


@dataclass(frozen=True)
class SegmentMergeOptions:
    enabled: bool = True
    max_gap_ms: int = 2_000
    max_duration_ms: int = 20_000


@dataclass
class InferenceTask:
    audio_path: str
    sub_prefix: str
    sub_formats: list[str]


def _normalize_merge_text(text: str) -> str:
    return " ".join(text.strip().split())


def merge_segments(segments: list[Segment], options: SegmentMergeOptions | None = None) -> list[Segment]:
    if options is None:
        options = SegmentMergeOptions()

    segments = [s for s in segments if s.text.strip()]
    segments.sort(key=lambda s: (s.start, s.end))
    if not options.enabled:
        return segments

    merged: list[Segment] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue

        last = merged[-1]
        gap_ms = seg.start - last.end
        if gap_ms > options.max_gap_ms:
            merged.append(seg)
            continue

        merged_duration_ms = seg.end - last.start
        if merged_duration_ms > options.max_duration_ms:
            merged.append(seg)
            continue

        last_norm = _normalize_merge_text(last.text)
        seg_norm = _normalize_merge_text(seg.text)

        if seg_norm.startswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue
        if last_norm.startswith(seg_norm) or last_norm.endswith(seg_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=last.text)
            continue
        if seg_norm.endswith(last_norm):
            merged[-1] = Segment(start=last.start, end=max(last.end, seg.end), text=seg.text)
            continue

        merged.append(seg)

    return merged


class SubWriter:
    @classmethod
    def txt(cls, segments: list[Segment], path: str) -> None:
        lines = [f"{segment.text}\n" for segment in segments]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def lrc(cls, segments: list[Segment], path: str) -> None:
        lines: list[str] = []
        for idx, segment in enumerate(segments):
            start_ts = cls.lrc_timestamp(segment.start)
            end_ts = cls.lrc_timestamp(segment.end)
            lines.append(f"[{start_ts}]{segment.text}\n")
            if idx != len(segments) - 1:
                next_start = segments[idx + 1].start
                if next_start is not None and end_ts == cls.lrc_timestamp(next_start):
                    continue
            lines.append(f"[{end_ts}]\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @staticmethod
    def lrc_timestamp(ms: int) -> str:
        m = ms // 60_000
        ms = ms - m * 60_000
        s = ms // 1_000
        ms = ms - s * 1_000
        ms = ms // 10
        return f"{m:02d}:{s:02d}.{ms:02d}"

    @classmethod
    def vtt(cls, segments: list[Segment], path: str) -> None:
        lines = ["WebVTT\n\n"]
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.vtt_timestamp(segment.start)} --> {cls.vtt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def vtt_timestamp(cls, ms: int) -> str:
        return cls._timestamp(ms, ".")

    @classmethod
    def srt(cls, segments: list[Segment], path: str) -> None:
        lines: list[str] = []
        for idx, segment in enumerate(segments):
            lines.append(f"{idx + 1}\n")
            lines.append(f"{cls.srt_timestamp(segment.start)} --> {cls.srt_timestamp(segment.end)}\n")
            lines.append(f"{segment.text}\n\n")
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    @classmethod
    def srt_timestamp(cls, ms: int) -> str:
        return cls._timestamp(ms, ",")

    @classmethod
    def _timestamp(cls, ms: int, delim: str) -> str:
        h = ms // 3_600_000
        ms -= h * 3_600_000
        m = ms // 60_000
        ms -= m * 60_000
        s = ms // 1_000
        ms -= s * 1_000
        return f"{h:02d}:{m:02d}:{s:02d}{delim}{ms:03d}"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local ASR/translation pipeline with Intel XPU support.")
    parser.add_argument("--model_name_or_path", type=str, default=DEFAULT_HF_MODEL_ID)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--generation_config", type=str, default="generation_config.json5")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--chunk_duration_sec", type=float, default=30.0)
    parser.add_argument("--min_chunk_duration_sec", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--audio_suffixes", type=str, default="wav,flac,mp3,m4a,aac,ogg,wma,mp4,mkv,avi,mov,webm,flv,wmv")
    parser.add_argument("--sub_formats", type=str, default="srt,vtt,lrc,txt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--merge_segments", dest="merge_segments", action="store_true", default=None)
    parser.add_argument("--no_merge_segments", dest="merge_segments", action="store_false", default=None)
    parser.add_argument("--merge_max_gap_ms", type=int, default=None)
    parser.add_argument("--merge_max_duration_ms", type=int, default=None)
    parser.add_argument("base_dirs", nargs=argparse.REMAINDER)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested != "auto":
        if requested == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            raise RuntimeError("Requested device xpu but torch.xpu is not available.")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested device cuda but CUDA is not available.")
        return requested

    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_torch_dtype(dtype_name: str, runtime_device: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[dtype_name]
    if runtime_device in {"cpu", "xpu"} and dtype == torch.float16:
        logger.warning("float16 on %s is often unstable, fallback to float32.", runtime_device)
        return torch.float32
    return dtype


def _load_generation_config(path: str) -> Dict[str, Any]:
    defaults = {
        "language": "ja",
        "task": "translate",
    }
    if not os.path.exists(path):
        return defaults
    with open(path, "r", encoding="utf-8") as f:
        loaded = pyjson5.decode_io(f)
    return dict(**ChainMap(loaded, defaults))


def _load_segment_merge_options(args: argparse.Namespace, generation_config: Dict[str, Any]) -> SegmentMergeOptions:
    from_file = generation_config.pop("segment_merge", None)
    options = SegmentMergeOptions()

    if isinstance(from_file, dict):
        options = SegmentMergeOptions(
            enabled=bool(from_file.get("enabled", options.enabled)),
            max_gap_ms=int(from_file.get("max_gap_ms", options.max_gap_ms)),
            max_duration_ms=int(from_file.get("max_duration_ms", options.max_duration_ms)),
        )

    return SegmentMergeOptions(
        enabled=args.merge_segments if args.merge_segments is not None else options.enabled,
        max_gap_ms=args.merge_max_gap_ms if args.merge_max_gap_ms is not None else options.max_gap_ms,
        max_duration_ms=(
            args.merge_max_duration_ms if args.merge_max_duration_ms is not None else options.max_duration_ms
        ),
    )


def _build_generate_kwargs(generation_config: Dict[str, Any], max_new_tokens: int) -> Dict[str, Any]:
    allowed = {
        "language",
        "task",
        "num_beams",
        "temperature",
        "repetition_penalty",
        "length_penalty",
        "do_sample",
    }
    kwargs = {k: v for k, v in generation_config.items() if k in allowed}
    kwargs["max_new_tokens"] = max_new_tokens
    return kwargs


class Inference:
    sub_writers = {"lrc": SubWriter.lrc, "srt": SubWriter.srt, "vtt": SubWriter.vtt, "txt": SubWriter.txt}

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model_name_or_path = args.model_name_or_path.strip() if args.model_name_or_path else DEFAULT_HF_MODEL_ID
        if not self.model_name_or_path:
            self.model_name_or_path = DEFAULT_HF_MODEL_ID
        self.device = resolve_device(args.device)
        self.torch_dtype = resolve_torch_dtype(args.torch_dtype, self.device)
        self.generate_config = _load_generation_config(args.generation_config)
        self.segment_merge_options = _load_segment_merge_options(args, self.generate_config)
        self.generate_kwargs = _build_generate_kwargs(self.generate_config, args.max_new_tokens)
        self.chunk_duration_sec = max(1.0, args.chunk_duration_sec)
        self.min_chunk_duration_sec = max(0.1, args.min_chunk_duration_sec)
        self.overwrite = args.overwrite
        self.output_dir = args.output_dir

        if self.output_dir and not os.path.isabs(self.output_dir):
            self.output_dir = os.path.join(os.getcwd(), self.output_dir)

        self.audio_suffixes = {k.strip().lower(): True for k in args.audio_suffixes.split(",") if k.strip()}
        self.sub_formats: list[str] = []
        for sub_format in args.sub_formats.split(","):
            key = sub_format.strip().lower()
            if key not in self.sub_writers:
                raise ValueError(f"Unknown subtitle format: {key}")
            self.sub_formats.append(key)

        logger.info("Runtime device: %s", self.device)
        logger.info("Torch dtype: %s", str(self.torch_dtype).replace("torch.", ""))
        logger.info("Generate kwargs: %s", self.generate_kwargs)
        logger.info(
            "Segment merge: enabled=%s, max_gap_ms=%s, max_duration_ms=%s",
            self.segment_merge_options.enabled,
            self.segment_merge_options.max_gap_ms,
            self.segment_merge_options.max_duration_ms,
        )

    def _build_asr_pipeline(self):
        logger.info("Loading model: %s", self.model_name_or_path)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_name_or_path)

        try:
            return pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=self.device,
            )
        except Exception:
            # Some pipeline versions are strict about non-CUDA string devices.
            return pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                return_timestamps=True,
                torch_dtype=self.torch_dtype,
                device=-1,
            )

    def generates(self, base_dirs: Iterable[str]) -> None:
        base_dirs = list(base_dirs)
        if not base_dirs:
            logger.warning("Please provide at least one input file or directory.")
            return

        tasks = self._scan(base_dirs)
        if not tasks:
            logger.info("No files found.")
            return

        asr_pipe = self._build_asr_pipeline()
        logger.info("Total tasks: %d", len(tasks))

        for idx, task in enumerate(tasks, 1):
            logger.info("[%d/%d] Processing: %s", idx, len(tasks), task.audio_path)
            segments = self._transcribe_single_file(asr_pipe, task.audio_path)
            segments = merge_segments(segments, self.segment_merge_options)

            os.makedirs(os.path.dirname(task.sub_prefix), exist_ok=True)
            for sub_suffix in task.sub_formats:
                sub_path = f"{task.sub_prefix}.{sub_suffix}"
                self.sub_writers[sub_suffix](segments, sub_path)
                logger.info("Written: %s", sub_path)

    def _transcribe_single_file(self, asr_pipe, audio_path: str) -> list[Segment]:
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio = audio.astype(np.float32)
        total_duration = len(audio) / sr if len(audio) else 0.0
        logger.info("Audio duration: %.2fs", total_duration)

        chunk_samples = int(self.chunk_duration_sec * sr)
        min_samples = int(self.min_chunk_duration_sec * sr)
        total_chunks = max(1, math.ceil(len(audio) / chunk_samples)) if chunk_samples > 0 else 1
        all_segments: list[Segment] = []

        for i in range(total_chunks):
            start_sample = i * chunk_samples
            end_sample = min((i + 1) * chunk_samples, len(audio))
            current_audio = audio[start_sample:end_sample]
            if len(current_audio) < min_samples:
                continue

            time_offset_sec = start_sample / sr
            try:
                result = asr_pipe(
                    {"raw": current_audio, "sampling_rate": sr},
                    generate_kwargs=self.generate_kwargs,
                )
            except Exception as exc:
                logger.warning("Chunk %d/%d failed: %s", i + 1, total_chunks, exc)
                continue

            all_segments.extend(self._extract_segments(result, time_offset_sec))
            logger.info("Chunk %d/%d done.", i + 1, total_chunks)

        return all_segments

    @staticmethod
    def _extract_segments(result: Dict[str, Any], time_offset_sec: float) -> list[Segment]:
        extracted: list[Segment] = []
        chunks = result.get("chunks", [])

        if not chunks:
            text = str(result.get("text", "")).strip()
            if text:
                start_ms = int(time_offset_sec * 1_000)
                extracted.append(Segment(start=start_ms, end=start_ms + 2_000, text=text))
            return extracted

        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            timestamp = chunk.get("timestamp")
            if not text or not timestamp:
                continue
            local_start, local_end = timestamp
            if local_start is None:
                local_start = 0.0
            if local_end is None:
                local_end = local_start + 2.0
            global_start_ms = int((time_offset_sec + float(local_start)) * 1_000)
            global_end_ms = int((time_offset_sec + float(local_end)) * 1_000)
            if global_end_ms <= global_start_ms:
                global_end_ms = global_start_ms + 200
            extracted.append(Segment(start=global_start_ms, end=global_end_ms, text=text))

        return extracted

    def _scan(self, base_dirs: Iterable[str]) -> list[InferenceTask]:
        tasks: list[InferenceTask] = []

        def process(base_path: str, audio_path: str) -> None:
            p = Path(audio_path)
            suffix = p.suffix.lower().lstrip(".")
            if suffix not in self.audio_suffixes:
                return

            rel_path = p.relative_to(base_path)
            abs_path = Path(os.path.join(self.output_dir or base_path, rel_path))
            sub_formats: list[str] = []
            for sub_suffix in self.sub_formats:
                sub_path = abs_path.parent / f"{abs_path.stem}.{sub_suffix}"
                if sub_path.exists() and not self.overwrite:
                    continue
                sub_formats.append(sub_suffix)
            if not sub_formats:
                return

            tasks.append(
                InferenceTask(
                    audio_path=audio_path,
                    sub_prefix=str(abs_path.parent / abs_path.stem),
                    sub_formats=sub_formats,
                )
            )

        for base_dir in base_dirs:
            base_dir = os.path.expanduser(base_dir)
            parent_dir = os.path.dirname(base_dir)
            if os.path.isdir(base_dir):
                for root, _, files in os.walk(base_dir, topdown=True):
                    for file in files:
                        process(parent_dir, os.path.join(root, file))
            else:
                process(parent_dir, base_dir)

        return tasks


def main() -> None:
    args = parse_arguments()
    logger.setLevel(args.log_level)

    if not args.base_dirs:
        logger.warning("Please drag files/folders or pass input paths.")
        sys.exit(1)

    inference = Inference(args)
    inference.generates(args.base_dirs)


if __name__ == "__main__":
    if getattr(sys, "frozen", False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.dirname(__file__))
    main()
