"""Conversation dataset producing ready-to-train ASR and event tensors."""

import torch
import torch.nn.functional as F

try:
    from .conv_dataset import DEFAULT_DATASET_PATH, VapDataset, collate_conversations
except ImportError:
    from conv_dataset import DEFAULT_DATASET_PATH, VapDataset, collate_conversations


def make_asr_tensors(segments, tokenizer, max_tokens, session="unknown"):
    sequence = tokenizer.encode(segments)
    if len(sequence) - 1 > max_tokens:
        raise ValueError(
            f"Session {session}: ASR target has {len(sequence) - 1} tokens, "
            f"exceeding max_tokens={max_tokens}; increase the limit or split the record"
        )
    tokens = torch.tensor(sequence, dtype=torch.long)
    return {"decoder_input_ids": tokens[:-1], "asr_labels": tokens[1:]}


class ConvASRBCDataset(VapDataset):
    """A/B waveforms, VAD, five event labels, valid lengths, and shifted ASR IDs.

    Bind the same Qwen ASRTokenizer to train/validation datasets. With no
    tokenizer, item loading fails explicitly. Empty transcripts have BOS
    input and EOS target.
    """

    def __init__(self, path=DEFAULT_DATASET_PATH, *, tokenizer=None, max_tokens=2048,
                 horizon=2, sample_rate=16000, frame_hz=50,
                 audio_root_a=None, audio_root_b=None):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        super().__init__(path, horizon=horizon, sample_rate=sample_rate,
                         frame_hz=frame_hz, mono=False, audio_root_a=audio_root_a,
                         audio_root_b=audio_root_b, include_event_labels=True,
                         include_asr_text=True)
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self.tokenizer is None:
            raise ValueError("Bind an ASRTokenizer with set_tokenizer() before loading items")
        sample = super().__getitem__(index)
        sample.update(make_asr_tensors(sample["asr_segments"], self.tokenizer,
                                      self.max_tokens, sample["session"]))
        return sample

    def collate_fn(self, samples):
        if self.tokenizer is None:
            raise ValueError("No ASR tokenizer bound to the dataset")
        return collate_asr_conversations(samples, self.tokenizer, self.max_tokens)


def collate_asr_conversations(samples, tokenizer, max_tokens):
    """Pad audio/frame targets and already-tokenized ASR samples.

    Raw asr_segments are also accepted for compatibility with existing callers.
    The returned dictionary contains only tensors and can go straight to the
    joint trainer; ASR target padding is -100 and decoder padding is PAD.
    """
    batch = collate_conversations(samples)
    encoded = []
    for sample in samples:
        if "decoder_input_ids" in sample and "asr_labels" in sample:
            item = {key: sample[key] for key in ("decoder_input_ids", "asr_labels")}
        else:
            item = make_asr_tensors(sample["asr_segments"], tokenizer, max_tokens,
                                    sample.get("session", "unknown"))
        inputs, labels = item["decoder_input_ids"], item["asr_labels"]
        if inputs.ndim != 1 or labels.shape != inputs.shape or not 1 <= len(inputs) <= max_tokens:
            raise ValueError("ASR inputs/labels must be aligned nonempty 1D tensors within max_tokens")
        encoded.append(item)
    length = max(len(item["decoder_input_ids"]) for item in encoded)
    for key, pad_value in (("decoder_input_ids", tokenizer.pad_id), ("asr_labels", -100)):
        batch[key] = torch.stack([
            F.pad(item[key], (0, length - len(item[key])), value=pad_value) for item in encoded
        ])
    return batch
