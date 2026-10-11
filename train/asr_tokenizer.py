"""Qwen Hugging Face tokenizer adapter for speaker-labelled ASR targets."""

import hashlib
import json
from pathlib import Path

DEFAULT_QWEN_TOKENIZER = "Qwen/Qwen2-Audio-7B-Instruct"


class ASRTokenizer:
    SPEAKER_TOKENS = ("<speaker_A>", "<speaker_B>")

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        vocabulary = tokenizer.get_vocab()
        if tokenizer.eos_token_id is None:
            if "<|im_end|>" not in vocabulary:
                raise ValueError("Qwen tokenizer must provide an EOS token")
            tokenizer.add_special_tokens({"eos_token": "<|im_end|>"})
        if tokenizer.bos_token_id is None or tokenizer.bos_token_id == tokenizer.eos_token_id:
            bos_token = "<|im_start|>" if "<|im_start|>" in vocabulary else "<|asr_bos|>"
            tokenizer.add_special_tokens({"bos_token": bos_token})
        if tokenizer.pad_token_id is None or tokenizer.pad_token_id in (
            tokenizer.bos_token_id, tokenizer.eos_token_id
        ):
            pad_token = ("<|endoftext|>" if "<|endoftext|>" in vocabulary
                         and tokenizer.convert_tokens_to_ids("<|endoftext|>")
                         not in (tokenizer.bos_token_id, tokenizer.eos_token_id)
                         else "<|asr_pad|>")
            tokenizer.add_special_tokens({"pad_token": pad_token})
        # Preserve Qwen's existing audio/chat special tokens.
        tokenizer.add_tokens(list(self.SPEAKER_TOKENS), special_tokens=True)
        tokenizer.padding_side = "right"
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.ids = {token: tokenizer.convert_tokens_to_ids(token) for token in self.SPEAKER_TOKENS}
        self._metadata = self._make_metadata()

    def __len__(self):
        # len includes speaker tokens added after the original vocabulary.
        return len(self.tokenizer)

    @classmethod
    def from_pretrained(cls, name_or_path=DEFAULT_QWEN_TOKENIZER, **kwargs):
        from transformers import AutoTokenizer
        return cls(AutoTokenizer.from_pretrained(name_or_path, **kwargs))

    def encode(self, segments):
        ids = [self.bos_id]
        for speaker, text in segments:
            if speaker not in ("A", "B"):
                raise ValueError(f"Invalid ASR speaker: {speaker}")
            if text:
                ids.append(self.ids[f"<speaker_{speaker}>"])
                ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
        ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        result = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id == self.eos_id:
                break
            if token_id not in (self.pad_id, self.bos_id):
                result.append(token_id)
        return self.tokenizer.decode(result, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)

    def _make_metadata(self):
        vocabulary = json.dumps(self.tokenizer.get_vocab(), ensure_ascii=False,
                                sort_keys=True, separators=(",", ":"))
        backend = getattr(self.tokenizer, "backend_tokenizer", None)
        # Fingerprint token IDs and BPE rules, independent of save location.
        rules = ""
        if backend is not None:
            rules_data = json.loads(backend.to_str())
            rules_data.pop("padding", None)
            rules_data.pop("truncation", None)
            rules = json.dumps(rules_data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return {"type": "qwen_hf_v1", "tokenizer_class": type(self.tokenizer).__name__,
                "vocab_size": len(self), "pad_id": self.pad_id, "bos_id": self.bos_id,
                "eos_id": self.eos_id, "speaker_ids": self.ids.copy(),
                "vocab_sha256": hashlib.sha256(vocabulary.encode("utf-8")).hexdigest(),
                "rules_sha256": hashlib.sha256(rules.encode("utf-8")).hexdigest()}

    def to_dict(self):
        return dict(self._metadata)

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(str(directory))
        (directory / "asr_tokenizer_config.json").write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory, **kwargs):
        directory = Path(directory)
        if directory.is_file():
            raise ValueError("Use a Qwen tokenizer directory, not the old character tokenizer.json")
        tokenizer = cls.from_pretrained(str(directory), **kwargs)
        config = directory / "asr_tokenizer_config.json"
        if config.exists():
            expected = json.loads(config.read_text(encoding="utf-8"))
            actual = tokenizer.to_dict()
            differences = {key: (value, actual.get(key)) for key, value in expected.items()
                           if value != actual.get(key)}
            if differences:
                raise ValueError(f"Saved ASR tokenizer configuration mismatch: {differences}")
        return tokenizer
