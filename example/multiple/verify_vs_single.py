"""
Verify that ``MaaiMultiple`` produces the same per-frame outputs as running
the equivalent ``Maai`` instances independently, and compare their
processing times.

Strategy:
- Load two channels of audio from wav files.
- Build three independent ``Maai`` instances (vap / bc / nod) and one
  ``MaaiMultiple`` configured with the same three sub-models.
- Bypass each instance's worker thread and feed audio chunks directly via
  ``process(x1, x2)``. Drain the result queues after every chunk.
- Compare the per-frame numerical outputs across the two approaches and
  report processing time totals.

Use ``model_type="normal"`` (CPC encoder) for an apples-to-apples test:
CPC is stateless, so the outputs should match bit-for-bit. ``"normal-ver2"``
(streaming Mimi encoder) should also match because the shared encoders see
exactly the same history under the same input ordering, but tiny differences
of ~1e-6 may appear from floating-point reordering.
"""

import sys
import os
import time
import queue
import argparse
import numpy as np
import soundfile as sf

# Allow running directly from the source tree without ``pip install -e .``.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/')))

from maai import Maai, MaaiInput, MaaiMultiple


def load_wav_frames(path: str, frame_size: int = 160) -> list[np.ndarray]:
    """Load a 16 kHz mono wav file and split it into ``frame_size`` chunks."""
    data, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz wav, got {sr} Hz: {path}")
    if data.ndim > 1:
        data = data[:, 0]
    n = len(data) // frame_size
    return [
        np.asarray(data[i * frame_size : (i + 1) * frame_size], dtype=np.float32)
        for i in range(n)
    ]


def drain(q: queue.Queue) -> list:
    """Pull every available item off ``q`` without blocking."""
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except queue.Empty:
            return out


def max_abs_diff(a, b) -> float:
    """Element-wise max absolute difference between two scalars or arrays."""
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        return float("nan")
    if aa.size == 0:
        return 0.0
    return float(np.max(np.abs(aa - bb)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    here = os.path.dirname(os.path.abspath(__file__))
    default_wav1 = os.path.normpath(
        os.path.join(here, "../wav_sample/jpn_inoue_16k.wav")
    )
    default_wav2 = os.path.normpath(
        os.path.join(here, "../wav_sample/jpn_sumida_16k.wav")
    )
    parser.add_argument("--wav1", default=default_wav1, help="audio for ch1 (user)")
    parser.add_argument("--wav2", default=default_wav2, help="audio for ch2 (system)")
    parser.add_argument("--model-type", default="normal", choices=["normal", "normal-ver2"])
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Defaults to 10 for normal, 12.5 for normal-ver2.",
    )
    parser.add_argument("--lang", default="jp")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Optional truncation of the wav to this many seconds.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help=(
            "Maximum acceptable absolute difference per field before the "
            "comparison is flagged. Default 1e-5 covers Mimi/ONNX FP rounding."
        ),
    )
    args = parser.parse_args()

    if args.frame_rate is None:
        args.frame_rate = 10.0 if args.model_type == "normal" else 12.5

    print(f"Loading wav: {args.wav1}")
    print(f"Loading wav: {args.wav2}")
    frames_x1 = load_wav_frames(args.wav1)
    frames_x2 = load_wav_frames(args.wav2)

    if args.wav1 == default_wav1:
        frames_x1 = frames_x1[: 16000 * 15]  # first 15 seconds of the default wav
    if args.wav2 == default_wav2:
        frames_x2 = frames_x2[: 16000 * 15]

    n = min(len(frames_x1), len(frames_x2))
    if args.max_seconds is not None:
        n = min(n, int(args.max_seconds * 16000 // 160))
    frames_x1 = frames_x1[:n]
    frames_x2 = frames_x2[:n]
    audio_sec = n * 160 / 16000.0
    print(
        f"#frames (160-sample chunks): {n}  ({audio_sec:.2f} sec)"
        f"   |  model_type={args.model_type}, frame_rate={args.frame_rate}"
    )

    shared_kwargs = dict(
        frame_rate=args.frame_rate,
        device=args.device,
        model_type=args.model_type,
        context_len_sec=20,
    )
    modes = ["vap", "bc", "nod"]

    # The audio sources are only used by the constructor (never started),
    # because we feed audio manually via process(x1, x2).
    zero1 = MaaiInput.Zero()
    zero2 = MaaiInput.Zero()

    print("\nBuilding 3 single Maais ...")
    t0 = time.perf_counter()
    singles: dict[str, Maai] = {}
    for m in modes:
        singles[m] = Maai(
            mode=m, lang=args.lang, audio_ch1=zero1, audio_ch2=zero2, **shared_kwargs
        )
    t_build_single = time.perf_counter() - t0
    print(f"  done in {t_build_single:.2f} sec")

    print("Building MaaiMultiple ...")
    t0 = time.perf_counter()
    multi = MaaiMultiple(
        configs=[{"mode": m, "lang": args.lang} for m in modes],
        audio_ch1=zero1,
        audio_ch2=zero2,
        **shared_kwargs,
    )
    t_build_multi = time.perf_counter() - t0
    print(f"  done in {t_build_multi:.2f} sec")

    # ----------------------------------------------------------------
    # Run the three single Maais, summing their per-call wall time.
    # ----------------------------------------------------------------
    print("\nRunning single Maais (sum of 3 sequential process() calls) ...")
    single_outputs: dict[str, list] = {m: [] for m in modes}
    single_time: dict[str, float] = {m: 0.0 for m in modes}
    for x1, x2 in zip(frames_x1, frames_x2):
        for m in modes:
            ts = time.perf_counter()
            singles[m].process(x1, x2)
            single_time[m] += time.perf_counter() - ts
            single_outputs[m].extend(drain(singles[m].result_dict_queue))
    print(
        "  inferences per mode: "
        + ", ".join(f"{m}={len(single_outputs[m])}" for m in modes)
    )

    # ----------------------------------------------------------------
    # Run MaaiMultiple over the same audio.
    # ----------------------------------------------------------------
    print("\nRunning MaaiMultiple ...")
    multi_outputs: list = []
    multi_time = 0.0
    for x1, x2 in zip(frames_x1, frames_x2):
        ts = time.perf_counter()
        multi.process(x1, x2)
        multi_time += time.perf_counter() - ts
        multi_outputs.extend(drain(multi.result_dict_queue))
    print(f"  inferences: {len(multi_outputs)}")

    # ----------------------------------------------------------------
    # Compare numerical outputs
    # ----------------------------------------------------------------
    print("\n=== Numerical comparison (single vs MaaiMultiple) ===")
    n_compare = min(len(multi_outputs), *[len(single_outputs[m]) for m in modes])
    if n_compare == 0:
        print("No inferences produced; the audio was probably too short.")
    else:
        compare_keys = {
            "vap": ["p_now", "p_future", "vad"],
            "bc": ["p_bc"],
            "nod": ["p_bc", "p_nod_short", "p_nod_long", "p_nod_long_p"],
        }
        any_failed = False
        for m in modes:
            print(f"\n[{m}] comparing {n_compare} frames:")
            for k in compare_keys[m]:
                worst = 0.0
                for i in range(n_compare):
                    a = single_outputs[m][i][k]
                    b = multi_outputs[i][m][k]
                    worst = max(worst, max_abs_diff(a, b))
                if worst <= args.tolerance:
                    flag = "OK"
                else:
                    flag = "FAIL"
                    any_failed = True
                print(f"  {k:14}: max |Δ| = {worst:.3e}  [{flag}]")
        if any_failed:
            print(
                f"\n!! At least one field exceeded tolerance ({args.tolerance:.0e}). "
                "Single and multi outputs disagree."
            )
        else:
            print(
                f"\nAll fields within tolerance ({args.tolerance:.0e}). "
                "Single and multi outputs agree."
            )

    # ----------------------------------------------------------------
    # Timing summary
    # ----------------------------------------------------------------
    print("\n=== Processing time ===")
    total_single = sum(single_time.values())
    print(f"audio length            : {audio_sec:.2f} sec  ({n} chunks of 160 samples)")
    print(f"build (sum of 3 single) : {t_build_single:.3f} sec")
    print(f"build (MaaiMultiple)    : {t_build_multi:.3f} sec")
    print(f"single Maais (3 total)  : {total_single:.3f} sec  "
          f"(RTF = {total_single/audio_sec:.3f})")
    for m in modes:
        print(f"  - {m:<6}              : {single_time[m]:.3f} sec  "
              f"(RTF = {single_time[m]/audio_sec:.3f})")
    print(f"MaaiMultiple            : {multi_time:.3f} sec  "
          f"(RTF = {multi_time/audio_sec:.3f})")
    if multi_time > 0:
        speedup = total_single / multi_time
        savings_pct = (1.0 - multi_time / total_single) * 100.0 if total_single > 0 else 0.0
        print(f"speedup vs single x 3   : x{speedup:.2f}  "
              f"(savings: {savings_pct:.1f}% of wall time)")


if __name__ == "__main__":
    main()
