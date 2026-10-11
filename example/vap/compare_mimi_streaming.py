"""
Compare offline Mimi embeddings against MaAI-style incremental embeddings.

This script encodes the same waveform in two ways:
1. Offline: one full forward pass over the complete waveform.
2. Streaming: chunked processing that matches MaAI's buffer update rule.

The output reports frame count, cosine similarity, and L2 distance.
An optional plot can be saved for frame-wise cosine similarity.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src/")))

from maai.encoder import EncoderMimi


@dataclass
class StreamingChunkResult:
    embedding: torch.Tensor
    chunk_end_sample: int
    emitted_frames: int


def load_wav_mono_16k(wav_path: str, device: torch.device) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
    return waveform.to(device=device, dtype=torch.float32)


def run_offline(
    encoder: EncoderMimi,
    waveform: torch.Tensor,
) -> torch.Tensor:
    encoder.reset_streaming_state()
    with torch.no_grad():
        embedding = encoder(
            waveform.unsqueeze(0),
            streaming=False,
            finalize_stream=True,
        )
    return embedding.squeeze(0).cpu()


def run_streaming(
    encoder: EncoderMimi,
    waveform: torch.Tensor,
    frame_rate: int,
    chunk_samples: int,
    context_samples: int,
) -> List[StreamingChunkResult]:
    audio_frame_size = 16000 // frame_rate + context_samples
    buffer = torch.zeros(context_samples, dtype=waveform.dtype, device=waveform.device)
    emitted: List[StreamingChunkResult] = []

    encoder.reset_streaming_state()

    with torch.no_grad():
        for start in range(0, waveform.shape[-1], chunk_samples):
            end = min(start + chunk_samples, waveform.shape[-1])
            chunk = waveform[0, start:end]
            buffer = torch.cat([buffer, chunk], dim=0)

            if buffer.shape[0] < audio_frame_size:
                continue

            encoder_input = buffer.unsqueeze(0).unsqueeze(0)
            embedding = encoder(
                encoder_input,
                streaming=True,
                finalize_stream=False,
                has_overlap_context=True,
            )

            embedding = embedding.squeeze(0).cpu()
            emitted.append(
                StreamingChunkResult(
                    embedding=embedding,
                    chunk_end_sample=end,
                    emitted_frames=embedding.shape[0],
                )
            )

            buffer = buffer[-context_samples:].clone()

    return emitted


def run_causal_stream_reference(
    encoder: EncoderMimi,
    waveform: torch.Tensor,
    frame_rate: int,
) -> List[StreamingChunkResult]:
    emitted: List[StreamingChunkResult] = []
    previous_num_frames = 0
    emit_samples = 16000 // frame_rate

    with torch.no_grad():
        for end in range(emit_samples, waveform.shape[-1] + 1, emit_samples):
            prefix = waveform[:, :end].unsqueeze(0)
            encoder.reset_streaming_state()
            full_embedding = encoder(
                prefix,
                streaming=False,
                finalize_stream=True,
                has_overlap_context=False,
            ).squeeze(0).cpu()

            if full_embedding.shape[0] <= previous_num_frames:
                embedding = full_embedding[:0]
            else:
                embedding = full_embedding[previous_num_frames:]
                previous_num_frames = full_embedding.shape[0]

            emitted.append(
                StreamingChunkResult(
                    embedding=embedding,
                    chunk_end_sample=end,
                    emitted_frames=embedding.shape[0],
                )
            )

    return emitted


def concatenate_streaming(results: List[StreamingChunkResult]) -> torch.Tensor:
    valid = [result.embedding for result in results if result.embedding.shape[0] > 0]
    if not valid:
        return torch.zeros((0, 0), dtype=torch.float32)
    return torch.cat(valid, dim=0)


def compare_embeddings(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict:
    n_frames = min(reference.shape[0], candidate.shape[0])
    if n_frames == 0:
        raise ValueError("No comparable frames were produced. Check waveform length and frame rate.")

    reference_aligned = reference[:n_frames]
    candidate_aligned = candidate[:n_frames]

    cosine = F.cosine_similarity(reference_aligned, candidate_aligned, dim=-1)
    l2 = torch.norm(reference_aligned - candidate_aligned, dim=-1)

    return {
        "num_frames": n_frames,
        "cosine": cosine,
        "l2": l2,
        "cosine_mean": cosine.mean().item(),
        "cosine_min": cosine.min().item(),
        "cosine_max": cosine.max().item(),
        "l2_mean": l2.mean().item(),
        "l2_max": l2.max().item(),
    }


def summarize_range(name: str, cosine: torch.Tensor, l2: torch.Tensor) -> dict:
    if cosine.numel() == 0:
        return {
            "name": name,
            "frames": 0,
            "cosine_mean": float("nan"),
            "cosine_min": float("nan"),
            "l2_mean": float("nan"),
            "l2_max": float("nan"),
        }

    return {
        "name": name,
        "frames": int(cosine.numel()),
        "cosine_mean": cosine.mean().item(),
        "cosine_min": cosine.min().item(),
        "l2_mean": l2.mean().item(),
        "l2_max": l2.max().item(),
    }


def summarize_segments(cosine: torch.Tensor, l2: torch.Tensor, warmup_frames: int) -> List[dict]:
    n = int(cosine.numel())
    if n == 0:
        return []

    third = max(1, n // 3)
    summaries = [
        summarize_range("head", cosine[:third], l2[:third]),
        summarize_range("middle", cosine[third : min(2 * third, n)], l2[third : min(2 * third, n)]),
        summarize_range("tail", cosine[min(2 * third, n) :], l2[min(2 * third, n) :]),
    ]

    if warmup_frames > 0:
        warmup_frames = min(warmup_frames, n)
        summaries.append(
            summarize_range(
                f"after_warmup_{warmup_frames}",
                cosine[warmup_frames:],
                l2[warmup_frames:],
            )
        )

    return summaries


def save_cosine_plot(cosine: torch.Tensor, plot_path: str) -> None:
    import matplotlib.pyplot as plt

    x = torch.arange(cosine.shape[0]).cpu().numpy()
    y = cosine.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, color="blue", linewidth=1.5)
    plt.ylim(-1.0, 1.0)
    plt.xlabel("Frame")
    plt.ylabel("Cosine similarity")
    plt.title("Offline vs streaming Mimi embedding similarity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


def print_metric_block(name: str, metrics: dict, warmup_frames: int) -> None:
    print(f"[{name}]")
    print(f"compared_frames: {metrics['num_frames']}")
    print(f"cosine_mean: {metrics['cosine_mean']:.6f}")
    print(f"cosine_min: {metrics['cosine_min']:.6f}")
    print(f"cosine_max: {metrics['cosine_max']:.6f}")
    print(f"l2_mean: {metrics['l2_mean']:.6f}")
    print(f"l2_max: {metrics['l2_max']:.6f}")

    segment_summaries = summarize_segments(metrics["cosine"], metrics["l2"], warmup_frames)
    for summary in segment_summaries:
        print(
            "segment={name} frames={frames} cosine_mean={cosine_mean:.6f} cosine_min={cosine_min:.6f} l2_mean={l2_mean:.6f} l2_max={l2_max:.6f}".format(
                **summary
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare offline and MaAI-style streaming Mimi embeddings")
    parser.add_argument("wav", type=str, help="Path to input wav file")
    parser.add_argument("--frame-rate", type=int, default=10, help="Target frame rate used by MaAI")
    parser.add_argument("--chunk-samples", type=int, default=160, help="Input chunk size passed to MaAI process loop")
    parser.add_argument("--context-samples", type=int, default=320, help="Left context samples kept by MaAI")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--mimi-model-name", type=str, default="kyutai/mimi", help="Hugging Face Mimi model name")
    parser.add_argument("--plot", type=str, default="", help="Optional output path for cosine similarity plot")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Frames to ignore in an extra summary for warm-up analysis")
    args = parser.parse_args()

    device = torch.device(args.device)
    waveform = load_wav_mono_16k(args.wav, device)

    encoder = EncoderMimi(
        frame_hz=args.frame_rate,
        freeze=True,
        mimi_model_name=args.mimi_model_name,
        context_samples=args.context_samples,
    ).to(device)
    encoder.eval()

    offline = run_offline(encoder, waveform)
    causal_results = run_causal_stream_reference(
        encoder,
        waveform,
        frame_rate=args.frame_rate,
    )
    causal_stream = concatenate_streaming(causal_results)
    streaming_results = run_streaming(
        encoder,
        waveform,
        frame_rate=args.frame_rate,
        chunk_samples=args.chunk_samples,
        context_samples=args.context_samples,
    )
    streaming = concatenate_streaming(streaming_results)
    metrics_offline_vs_maai = compare_embeddings(offline, streaming)
    metrics_offline_vs_causal = compare_embeddings(offline, causal_stream)
    metrics_causal_vs_maai = compare_embeddings(causal_stream, streaming)

    total_emitted = sum(result.emitted_frames for result in streaming_results)
    total_causal_emitted = sum(result.emitted_frames for result in causal_results)

    print(f"wav: {args.wav}")
    print(f"input_samples: {waveform.shape[-1]}")
    print(f"offline_frames: {offline.shape[0]}")
    print(f"causal_stream_frames: {causal_stream.shape[0]}")
    print(f"streaming_frames: {streaming.shape[0]}")
    print(f"causal_stream_steps: {len(causal_results)}")
    print(f"streaming_steps: {len(streaming_results)}")
    print(f"causal_stream_emitted_frames: {total_causal_emitted}")
    print(f"streaming_emitted_frames: {total_emitted}")
    print_metric_block("offline_vs_maai_streaming", metrics_offline_vs_maai, args.warmup_frames)
    print_metric_block("offline_vs_causal_streaming", metrics_offline_vs_causal, args.warmup_frames)
    print_metric_block("causal_streaming_vs_maai_streaming", metrics_causal_vs_maai, args.warmup_frames)

    if args.plot:
        save_cosine_plot(metrics_offline_vs_maai["cosine"], args.plot)
        print(f"saved_plot: {args.plot}")


if __name__ == "__main__":
    main()