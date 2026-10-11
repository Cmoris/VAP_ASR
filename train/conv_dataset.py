"""Conversation JSON/JSONL annotations with the original VapDataset output."""

import json
import math
from pathlib import Path
from typing import Dict, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

try:
    from .audio import load_waveform
    from .utils import vad_list_to_onehot
except ImportError:
    from audio import load_waveform
    from utils import vad_list_to_onehot


DEFAULT_DATASET_PATH = Path(
    "C:/Program_moris/StreamingSpeechLLM/finetune_labels_whisper/"
    "finetune_labels_whisperx/l3_conv_train_with_backchannel"
)


def asr_segments_from_utterances(utterances):
    segments = []
    for utterance in sorted(utterances, key=lambda u: (u["start"], u["end"])):
        if utterance.get("is_silence", False):
            continue
        text = utterance.get("text", "")
        if not isinstance(text, str):
            raise ValueError("utterances[].text must be a string")
        if text.strip():
            segments.append((utterance["speaker"], text.strip()))
    return segments


def collate_conversations(samples):
    """Right-pad conversation audio and frame labels for training."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {}
    for key in ("waveform", "vad", "bc", "bc_detect", "turn_ending", "pause", "silence"):
        axis = -1 if key == "waveform" else 0
        length = max(sample[key].shape[axis] for sample in samples)
        batch[key] = torch.stack([
            torch.nn.functional.pad(
                sample[key], (0, length - sample[key].shape[-1])
                if key == "waveform" else (0, 0, 0, length - sample[key].shape[0])
            ) for sample in samples
        ])
    batch["valid_frames"] = torch.tensor([s["valid_frames"] for s in samples])
    if all("valid_samples" in sample for sample in samples):
        batch["valid_samples"] = torch.tensor([s["valid_samples"] for s in samples])
    return batch


def _read_last_line(path, buf=4096):
    """Read the seek-index footer without loading the entire JSONL file."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos, last = f.tell(), b""
        while pos > 0:
            read_size = min(buf, pos)
            pos -= read_size
            f.seek(pos)
            lines = (f.read(read_size) + last).split(b"\n")
            last = lines[0]
            non_empty = [line for line in lines[1:] if line.strip()]
            if non_empty:
                return non_empty[-1].decode("utf-8")
    return last.decode("utf-8")


class VapDataset(Dataset):
    """One annotation record per item; waveform channels are ordered A, B.

    The record spans its first through last utterance, preserving gaps and
    overlap. Missing speakers produce a silent channel. VAD includes the
    original extra horizon; times outside the record have no annotations.
    Override audio_root_a/b to replace the audio paths stored in annotations.
    Records have variable lengths, so default batching requires equal lengths
    (or batch_size=1), just as for variable-length CSV clips.
    """

    def __init__(
        self,
        path=DEFAULT_DATASET_PATH,
        horizon: float = 2,
        sample_rate: int = 16_000,
        frame_hz: int = 50,
        mono: bool = False,
        audio_root_a=None,
        audio_root_b=None,
        include_event_labels: bool = False,
        include_asr_text: bool = False,
    ):
        if horizon < 0 or sample_rate <= 0 or frame_hz <= 0:
            raise ValueError("horizon must be nonnegative; sample_rate and frame_hz positive")
        self.path = path
        self.sample_rate = sample_rate
        self.frame_hz = frame_hz
        self.horizon = horizon
        self.mono = mono
        self.include_event_labels = include_event_labels
        self.include_asr_text = include_asr_text
        self.audio_root_a = Path(audio_root_a) if audio_root_a is not None else None
        self.audio_root_b = Path(audio_root_b) if audio_root_b is not None else None
        self.audio_path_conv_cash = {}
        self.handles = []

        root = Path(path)
        files = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in (".json", ".jsonl")
        ) if root.is_dir() else [root]
        if not files:
            raise FileNotFoundError(f"No JSON/JSONL annotations found in {root}")
        for annotation_path in files:
            if annotation_path.suffix.lower() == ".jsonl":
                seeks = json.loads(_read_last_line(annotation_path))
                if not isinstance(seeks, list) or any(
                    type(seek) is not int or seek < 0 for seek in seeks
                ):
                    raise ValueError(f"Invalid seek-index footer in {annotation_path}")
                self.handles.extend((annotation_path, seek) for seek in seeks)
            elif annotation_path.suffix.lower() == ".json":
                self.handles.append((annotation_path, -1))
            else:
                raise ValueError(f"Unsupported annotation format: {annotation_path}")

    def __len__(self):
        return len(self.handles)

    def load_record(self, index):
        path, seek = self.handles[index]
        if seek == -1:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        with open(path, "rb") as f:
            f.seek(seek)
            return json.loads(f.readline().decode("utf-8"))

    def get_asr_segments(self, index):
        """Read transcript annotations without decoding audio."""
        record = self.load_record(index)
        ele = record[0] if isinstance(record, list) else record
        utterances = next(part["utterances"] for part in ele["content"]
                          if "utterances" in part)
        return asr_segments_from_utterances(utterances)

    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, str, int, float]]:
        record = self.load_record(idx)
        ele = record[0] if isinstance(record, list) else record
        content = ele["content"]
        utterances = sorted(
            next(part["utterances"] for part in content if "utterances" in part),
            key=lambda u: (u["start"], u["end"]),
        )
        session = str(next(part["id"] for part in content if "id" in part))
        if not utterances:
            raise ValueError(f"No utterances in session {session}")
        for u in utterances:
            if u["speaker"] not in ("A", "B") or not 0 <= u["start"] < u["end"]:
                raise ValueError(f"Invalid utterance in session {session}: {u}")

        start = utterances[0]["start"]
        end = max(u["end"] for u in utterances)
        # As in the reference, pad clips shorter than one second.
        duration = max(end - start, 1.0)
        num_samples = int(duration * self.sample_rate)
        channels, vad_list = [], [[], []]
        for channel, speaker, audio_root in (
            (0, "A", self.audio_root_a), (1, "B", self.audio_root_b)
        ):
            speaker_utterances = [u for u in utterances if u["speaker"] == speaker]
            if speaker_utterances:
                audio_path = (
                    audio_root / f"{session}_{speaker.lower()}.wav"
                    if audio_root is not None else Path(speaker_utterances[0]["audio"])
                )
                waveform, _ = load_waveform(
                    str(audio_path), start_time=start, end_time=end,
                    sample_rate=self.sample_rate, mono=True,
                )
                waveform = torch.nn.functional.pad(
                    waveform[:, :num_samples],
                    (0, max(0, num_samples - waveform.shape[-1])),
                )
                # Backchannels are speech too; silence annotations are not VAD.
                vad_list[channel] = [
                    [u["start"] - start, u["end"] - start]
                    for u in speaker_utterances if not u.get("is_silence", False)
                ]
            else:
                waveform = torch.zeros((1, num_samples))
            channels.append(waveform)

        waveform = torch.cat(channels, dim=0)
        if self.mono:
            waveform = waveform.mean(dim=0, keepdim=True)
        vad_duration = duration if self.include_event_labels else round(duration)
        vad = vad_list_to_onehot(
            vad_list, duration=vad_duration + self.horizon, frame_hz=self.frame_hz
        )
        result = {
            "session": session,
            "waveform": waveform,
            "vad": vad,
            "dataset": DEFAULT_DATASET_PATH.name,
        }
        if self.include_event_labels:
            # All event heads are scalar, matching the original BC heads:
            # either speaker's marked utterance makes the frame positive.
            n_frames = math.ceil(duration * self.frame_hz)
            events = {name: torch.zeros(n_frames, 1) for name in
                      ("bc", "bc_detect", "turn_ending", "pause", "silence")}
            for u in utterances:
                first = max(0, int((u["start"] - start) * self.frame_hz))
                last = min(n_frames, max(first + 1,
                           math.ceil((u["end"] - start) * self.frame_hz)))
                for name, flag in (("bc_detect", "is_back_channel"),
                                   ("turn_ending", "is_turn_ending"),
                                   ("pause", "is_pause"), ("silence", "is_silence")):
                    marked = u.get(flag, u.get("is_turn_end", False)
                                   if name == "turn_ending" else False)
                    if marked:
                        events[name][first:last] = 1
            # Match ObjectiveVAP.get_labels_bc: predict BC 0.5 s ahead.
            shift = int(0.5 * self.frame_hz)
            if shift:
                events["bc"][:-shift] = events["bc_detect"][shift:]
            else:
                events["bc"][:] = events["bc_detect"]
            result.update(events)
            result["valid_frames"] = int((end - start) * self.frame_hz)
            result["valid_samples"] = int((end - start) * self.sample_rate)
        if self.include_asr_text:
            result["asr_segments"] = asr_segments_from_utterances(utterances)
        return result
