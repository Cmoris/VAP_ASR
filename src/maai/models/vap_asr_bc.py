"""VAP-BC with a teacher-forced autoregressive ASR transformer decoder."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .vap_bc import VapGPT_bc


@dataclass
class ASRConfig:
    vocab_size: int
    decoder_layers: int = 4
    num_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2
    max_tokens: int = 2048


def add_positions(x):
    """Sinusoidal positions without a fixed audio-length limit."""
    dim = x.shape[-1]
    positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)[:, None]
    frequencies = torch.exp(torch.arange(0, dim, 2, device=x.device).float()
                            * (-math.log(10000.0) / dim))
    angles = positions * frequencies
    encoding = torch.zeros(x.shape[1], dim, device=x.device)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, :dim // 2].cos()
    return x + encoding.to(x.dtype).unsqueeze(0)


class VapGPT_asr_bc(VapGPT_bc):
    def __init__(self, conf, asr_conf: ASRConfig):
        super().__init__(conf)
        if (asr_conf.vocab_size <= 6 or asr_conf.decoder_layers < 1
                or asr_conf.num_heads < 1 or conf.dim % asr_conf.num_heads
                or asr_conf.ff_multiplier < 1 or asr_conf.max_tokens < 2):
            raise ValueError("Invalid ASR vocabulary/decoder dimensions")
        special_ids = (asr_conf.pad_id, asr_conf.bos_id, asr_conf.eos_id)
        if len(set(special_ids)) != 3 or any(i < 0 or i >= asr_conf.vocab_size for i in special_ids):
            raise ValueError("ASR PAD/BOS/EOS IDs must be distinct and within the tokenizer vocabulary")
        self.asr_conf = asr_conf
        self.asr_embedding = nn.Embedding(asr_conf.vocab_size, conf.dim,
                                         padding_idx=asr_conf.pad_id)
        layer = nn.TransformerDecoderLayer(
            d_model=conf.dim, nhead=asr_conf.num_heads,
            dim_feedforward=conf.dim * asr_conf.ff_multiplier,
            dropout=asr_conf.dropout, batch_first=True,
        )
        self.asr_decoder = nn.TransformerDecoder(layer, asr_conf.decoder_layers,
                                                 norm=nn.LayerNorm(conf.dim))
        self.asr_head = nn.Linear(conf.dim, asr_conf.vocab_size)

    def decode_asr(self, memory, decoder_input_ids, memory_lengths=None):
        if decoder_input_ids.ndim != 2 or decoder_input_ids.shape[1] < 1:
            raise ValueError("decoder_input_ids must have shape (batch, positive token count)")
        if decoder_input_ids.shape[1] > self.asr_conf.max_tokens:
            raise ValueError("ASR sequence exceeds max_tokens")
        if memory.shape[1] == 0:
            raise ValueError("No encoded audio frames")
        length = decoder_input_ids.shape[1]
        causal_mask = torch.ones(length, length, device=memory.device,
                                 dtype=torch.bool).triu(1)
        memory_mask = None
        if memory_lengths is not None:
            if (memory_lengths < 1).any():
                raise ValueError("Each ASR sample needs at least one valid audio frame")
            memory_mask = (torch.arange(memory.shape[1], device=memory.device)[None, :]
                           >= memory_lengths[:, None])
        target = add_positions(self.asr_embedding(decoder_input_ids)
                               * math.sqrt(memory.shape[-1]))
        decoded = self.asr_decoder(
            target, add_positions(memory), tgt_mask=causal_mask,
            tgt_key_padding_mask=decoder_input_ids.eq(self.asr_conf.pad_id),
            memory_key_padding_mask=memory_mask,
        )
        return self.asr_head(decoded)

    def forward(self, x1, x2, cache=None, return_all_frames=False,
                return_logits=False, return_features=False,
                decoder_input_ids=None, memory_lengths=None):
        if decoder_input_ids is None:
            return super().forward(x1, x2, cache, return_all_frames,
                                   return_logits, return_features)
        if cache:
            raise ValueError("Teacher-forced ASR expects a complete record without streaming cache")
        out, new_cache = super().forward(x1, x2, return_logits=True, return_features=True)
        out["asr_logits"] = self.decode_asr(out["features"], decoder_input_ids, memory_lengths)
        if not return_features:
            out.pop("features")
        return out, new_cache

    @torch.no_grad()
    def generate_asr(self, memory, memory_lengths=None, max_new_tokens=256):
        """Greedy decoding from encoded memory; call eval() for deterministic inference."""
        if max_new_tokens < 1 or max_new_tokens > self.asr_conf.max_tokens:
            raise ValueError("max_new_tokens must be in [1, max_tokens]")
        tokens = torch.full((memory.shape[0], 1), self.asr_conf.bos_id,
                            device=memory.device, dtype=torch.long)
        finished = torch.zeros(memory.shape[0], device=memory.device, dtype=torch.bool)
        for _ in range(max_new_tokens):
            scores = self.decode_asr(memory, tokens, memory_lengths)[:, -1].clone()
            scores[:, self.asr_conf.pad_id] = float("-inf")
            scores[:, self.asr_conf.bos_id] = float("-inf")
            next_token = scores.argmax(-1)
            next_token = torch.where(finished, self.asr_conf.eos_id, next_token)
            tokens = torch.cat((tokens, next_token[:, None]), dim=1)
            finished |= next_token.eq(self.asr_conf.eos_id)
            if finished.all():
                break
        return tokens
