"""Check a VAP checkpoint's encoder with CPU audio; only PyTorch is required."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import wave

import torch


def load_encoder_module():
    path = Path(__file__).resolve().parents[1] / "src/maai/checkpoint_encoder.py"
    spec = importlib.util.spec_from_file_location("vap_checkpoint_encoder", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--frame-hz", type=float)
    parser.add_argument("--wav", help="Optional 16-bit PCM 16 kHz WAV for a real-audio check")
    args = parser.parse_args()
    module = load_encoder_module()
    encoder = module.load_vap_encoder(args.checkpoint, frame_hz=args.frame_hz)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)
    for key, tensor in encoder.state_dict().items():
        torch.testing.assert_close(tensor, state[f"encoder.{key}"], rtol=0, atol=0)
    records = []
    torch.manual_seed(1)
    with torch.no_grad():
        for duration in (1, 2, 20):
            # Nonzero input plus silence: validate numerical stability and lengths.
            waveform = torch.randn(2, 1, duration * 16000) * 0.05
            waveform[1] = 0
            output = encoder(waveform)
            expected = encoder.output_lengths(torch.tensor([waveform.shape[-1]])).item()
            assert output.shape == (2, expected, 256)
            assert torch.isfinite(output).all()
            torch.testing.assert_close(output, encoder(waveform))
            records.append({"input_seconds": duration, "output_shape": list(output.shape),
                            "finite": True, "repeatable": True})
        if args.wav:
            with wave.open(args.wav, "rb") as audio:
                if audio.getsampwidth() != 2 or audio.getframerate() != 16000:
                    raise ValueError("Use a 16-bit PCM WAV at 16 kHz")
                channels = audio.getnchannels()
                samples = torch.frombuffer(bytearray(audio.readframes(audio.getnframes())),
                                           dtype=torch.int16).float() / 32768
                waveform = samples.reshape(-1, channels).T.unsqueeze(1)
            output = encoder(waveform)
            assert torch.isfinite(output).all()
            records.append({"wav": args.wav, "output_shape": list(output.shape), "finite": True})
    print(json.dumps({"encoder_tensor_count": len(encoder.state_dict()),
                      "encoder_parameter_count": sum(p.numel() for p in encoder.parameters()),
                      "frame_hz": encoder.frame_hz, "output_dim": encoder.output_dim,
                      "all_weights_exact": True, "checks": records}))


if __name__ == "__main__":
    main()
