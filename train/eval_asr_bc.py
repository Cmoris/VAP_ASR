"""Evaluate joint ASR/event checkpoints or infer from two speaker WAV files."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from train_asr_bc import ASRConfig, ASRTokenizer, EVENTS, VapConfig, VapGPT_asr_bc
from train_bc import compute_losses
from conv_asr_bc_dataset import ConvASRBCDataset
from audio import load_waveform
from maai.checkpoint_encoder import encoder_from_state_dict


def load_eval_model(payload, tokenizer, device, crop_frames=None):
    for key in ("state_dict", "conf", "asr_conf", "tokenizer"):
        if key not in payload:
            raise ValueError(f"Expected a joint training checkpoint; missing {key}")
    if payload["tokenizer"] != tokenizer.to_dict():
        raise ValueError("Checkpoint/tokenizer mismatch; use the tokenizer saved with this run")
    conf = VapConfig(**payload["conf"])
    asr_conf = ASRConfig(**payload["asr_conf"])
    if (asr_conf.vocab_size, asr_conf.pad_id, asr_conf.bos_id, asr_conf.eos_id) != (
        len(tokenizer), tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id
    ):
        raise ValueError("ASR configuration does not match tokenizer IDs")
    if crop_frames is None:
        crop_frames = payload.get("encoder_spec", {}).get("crop_frames")
    if crop_frames is None:
        if conf.frame_hz == 50:
            raise ValueError("Old 50 Hz checkpoint needs --encoder-crop-frames: "
                             "0 for VAP encoder loading, 2 for original CPC loading")
        crop_frames = 0
    model = VapGPT_asr_bc(conf, asr_conf)
    state = payload["state_dict"]
    for channel in (1, 2):
        encoder = encoder_from_state_dict(state, conf.frame_hz, prefix=f"encoder{channel}.",
                                          crop_frames=crop_frames)
        setattr(model, f"encoder{channel}", encoder)
    if model.encoder1.output_dim != conf.dim:
        model.decrease_dimension = torch.nn.Linear(model.encoder1.output_dim, conf.dim)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for index, character in enumerate(reference, 1):
        current = [index]
        for other_index, other_character in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[other_index] + 1,
                               previous[other_index - 1] + (character != other_character)))
        previous = current
    return previous[-1]


def transcript_text(tokenizer, ids):
    # Speaker tags are reported but excluded from ASR character error rate.
    speaker_ids = set(tokenizer.ids.values())
    text = tokenizer.decode([int(i) for i in ids if int(i) not in speaker_ids and int(i) != -100])
    return "".join(text.split())


def event_counts(probabilities, target, valid_frames, frame_hz, name, threshold):
    count = min(probabilities.shape[0], int(valid_frames))
    if name == "bc":
        count = min(count, max(0, int(valid_frames) - int(0.5 * frame_hz)))
    pred = probabilities[:count] >= threshold
    ref = target[:count].bool()
    return {"tp": int((pred & ref).sum()), "fp": int((pred & ~ref).sum()),
            "fn": int((~pred & ref).sum()), "tn": int((~pred & ~ref).sum())}


def event_metrics(counts):
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    return {**counts, "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "accuracy": (tp + tn) / max(1, tp + fp + fn + tn)}


@torch.no_grad()
def predict(model, batch, max_new_tokens, teacher_forcing=False):
    for encoder in (model.encoder1, model.encoder2):
        encoder.reset_streaming_state()
    x1, x2 = model.encode_audio(batch["waveform"][:, 1:2], batch["waveform"][:, :1])
    lengths = model.encoder1.output_lengths(batch["valid_samples"]).clamp(max=x1.shape[1])
    kwargs = ({"decoder_input_ids": batch["decoder_input_ids"], "memory_lengths": lengths}
              if teacher_forcing else {})
    out, _ = model(x1, x2, return_logits=True, return_features=True,
                   **kwargs)
    generated = model.generate_asr(out["features"], lengths,
                                    max_new_tokens=min(max_new_tokens, model.asr_conf.max_tokens))
    return out, generated


def evaluate_dataset(model, dataset, tokenizer, device, output, threshold,
                     max_new_tokens, num_workers=0, max_samples=None, loss_weights=None):
    loader_dataset = dataset if max_samples is None else Subset(dataset, range(min(len(dataset), max_samples)))
    loader = DataLoader(loader_dataset, batch_size=1, shuffle=False,
                         num_workers=num_workers, collate_fn=dataset.collate_fn)
    totals, counts = {}, {name: {key: 0 for key in ("tp", "fp", "fn", "tn")} for name in EVENTS}
    errors = reference_characters = samples = truncated = 0
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as file:
        for index, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            out, generated = predict(model, batch, max_new_tokens, teacher_forcing=True)
            losses = compute_losses(model, batch, out=out)
            losses["asr"] = F.cross_entropy(out["asr_logits"].transpose(1, 2),
                                            batch["asr_labels"], ignore_index=-100)
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item()
            hypothesis_ids = generated[0].tolist()
            reference_ids = batch["asr_labels"][0].tolist()
            hypothesis = transcript_text(tokenizer, hypothesis_ids)
            reference = transcript_text(tokenizer, reference_ids)
            errors += edit_distance(reference, hypothesis)
            reference_characters += len(reference)
            reached_limit = tokenizer.eos_id not in hypothesis_ids
            truncated += int(reached_limit)
            n = min(out["vad"].shape[1], int(batch["valid_frames"][0]))
            probabilities = {name: out[name][0, :n, 0].sigmoid().cpu().tolist() for name in EVENTS}
            for name in EVENTS:
                values = event_counts(out[name][0, :, 0].sigmoid(), batch[name][0, :, 0],
                                      batch["valid_frames"][0], model.frame_hz, name, threshold)
                for key, value in values.items():
                    counts[name][key] += value
            file.write(json.dumps({"index": index, "asr_reference": tokenizer.decode(
                [i for i in reference_ids if i != -100]), "asr_prediction": tokenizer.decode(hypothesis_ids),
                "generation_limit_reached": reached_limit, "frame_hz": model.frame_hz,
                "event_probabilities": probabilities}, ensure_ascii=False) + "\n")
            samples += 1
    if not samples:
        raise ValueError("Empty test dataset")
    losses = {key: value / samples for key, value in totals.items()}
    losses["total"] = sum(value * (loss_weights or {}).get(key, 1.0) for key, value in losses.items())
    return {"samples": samples, "losses": losses,
            "cer": errors / reference_characters if reference_characters else None,
            "character_errors": errors, "reference_characters": reference_characters,
            "generation_limit_reached_samples": truncated, "threshold": threshold,
            "events": {name: event_metrics(values) for name, values in counts.items()}}


def infer_wavs(model, tokenizer, device, wav_a, wav_b, max_new_tokens):
    channels = [load_waveform(path, sample_rate=model.sample_rate, mono=True)[0]
                for path in (wav_a, wav_b)]
    samples = max(channel.shape[-1] for channel in channels)
    waveform = torch.cat([F.pad(channel, (0, samples - channel.shape[-1])) for channel in channels])
    batch = {"waveform": waveform.unsqueeze(0).to(device),
             "valid_samples": torch.tensor([samples], device=device)}
    out, generated = predict(model, batch, max_new_tokens)
    return {"wav_a": wav_a, "wav_b": wav_b, "frame_hz": model.frame_hz,
            "asr_prediction": tokenizer.decode(generated[0].tolist()),
            "generation_limit_reached": tokenizer.eos_id not in generated[0].tolist(),
            "event_probabilities": {name: out[name][0, :, 0].sigmoid().cpu().tolist() for name in EVENTS}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--test-path")
    source.add_argument("--wav-a")
    parser.add_argument("--wav-b")
    parser.add_argument("--audio-root-a")
    parser.add_argument("--audio-root-b")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="eval_asr_bc")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--encoder-crop-frames", type=int, choices=(0, 2))
    args = parser.parse_args()
    if bool(args.wav_a) != bool(args.wav_b):
        parser.error("Provide both --wav-a and --wav-b")
    if (args.num_workers < 0 or not 0 <= args.threshold <= 1 or args.max_new_tokens < 1
            or args.max_samples is not None and args.max_samples < 1):
        parser.error("Invalid worker count, threshold, or sample/token limit")
    checkpoint = Path(args.checkpoint)
    tokenizer = ASRTokenizer.load(args.tokenizer_path or checkpoint.parent / "tokenizer",
                                  local_files_only=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = load_eval_model(payload, tokenizer, device, args.encoder_crop_frames)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.test_path:
        dataset = ConvASRBCDataset(args.test_path, tokenizer=tokenizer,
                                   horizon=max(2.0, model.horizon_time), sample_rate=model.sample_rate,
                                   frame_hz=model.frame_hz, max_tokens=model.asr_conf.max_tokens,
                                   audio_root_a=args.audio_root_a, audio_root_b=args.audio_root_b)
        results = evaluate_dataset(model, dataset, tokenizer, device, output, args.threshold,
                                   args.max_new_tokens, args.num_workers, args.max_samples,
                                   payload.get("loss_weights"))
        filename = "metrics.json"
    else:
        results = infer_wavs(model, tokenizer, device, args.wav_a, args.wav_b, args.max_new_tokens)
        filename = "wav_predictions.json"
    (output / filename).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output / filename),
                      **({"samples": results["samples"], "cer": results["cer"]} if args.test_path
                         else {"asr_prediction": results["asr_prediction"]})}, ensure_ascii=True))


if __name__ == "__main__":
    main()
