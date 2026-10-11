"""Train the inference VapGPT_bc model on conversation annotations."""

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from maai.models.config import VapConfig
from maai.models.vap_bc import VapGPT_bc
from conv_dataset import DEFAULT_DATASET_PATH, VapDataset, collate_conversations

EVENTS = ("bc", "bc_detect", "turn_ending", "pause", "silence")


def masked_mean(loss, mask):
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(loss)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


def compute_losses(model, batch, out=None):
    # encode_audio swaps its two inputs; pass B then A to keep labels in A/B order.
    if out is None:
        for encoder in (model.encoder1, model.encoder2):
            encoder.reset_streaming_state()
        x1, x2 = model.encode_audio(batch["waveform"][:, 1:2], batch["waveform"][:, :1])
        out, _ = model(x1, x2, return_logits=True)
    n = out["vad"].shape[1]
    times = torch.arange(n, device=out["vad"].device)[None, :]
    lengths = batch["valid_frames"][:, None]
    valid = times < lengths
    losses = {"vad": masked_mean(F.binary_cross_entropy_with_logits(
        out["vad"], batch["vad"][:, :n], reduction="none"), valid)}
    horizon = model.objective.projection_window_extractor.horizon
    labels = model.objective.get_labels(batch["vad"])
    projection_n = min(n, labels.shape[1])
    losses["vap"] = masked_mean(F.cross_entropy(
        out["logits"][:, :projection_n].transpose(1, 2),
        labels[:, :projection_n], reduction="none"),
        times[:, :projection_n] + horizon < lengths)
    for name in EVENTS:
        mask = (times + int(0.5 * model.frame_hz) < lengths) if name == "bc" else valid
        losses[name] = masked_mean(F.binary_cross_entropy_with_logits(
            out[name], batch[name][:, :n], reduction="none"), mask)
    return losses


def run_epoch(model, loader, device, optimizer=None, max_batches=None,
              loss_function=None, loss_weights=None):
    training = optimizer is not None
    model.train(training)
    if model.conf.freeze_encoder:
        model.encoder1.eval()
        model.encoder2.eval()
    totals, count = {}, 0
    with torch.set_grad_enabled(training):
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            losses = (loss_function or compute_losses)(model, batch)
            total = sum(value * (loss_weights or {}).get(key, 1.0)
                        for key, value in losses.items())
            if not torch.isfinite(total):
                raise RuntimeError("Non-finite training loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            size = batch["waveform"].shape[0]
            for key, value in {**losses, "total": total}.items():
                totals[key] = totals.get(key, 0.0) + value.detach().item() * size
            count += size
    if not count:
        raise ValueError("Empty data loader")
    return {key: value / count for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--audio-root-a")
    parser.add_argument("--audio-root-b")
    encoder_source = parser.add_mutually_exclusive_group(required=True)
    encoder_source.add_argument("--cpc-model", help="Original CPC checkpoint")
    encoder_source.add_argument("--encoder-checkpoint", help="VAP checkpoint; load encoder weights only")
    parser.add_argument("--checkpoint", help="Optional original BC weights to fine-tune")
    parser.add_argument("--output-dir", default="trained_bc")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3.63e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="One train and validation batch")
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.set_defaults(vap_frame_hz=None)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("batch-size and epochs must be positive")
    if Path(args.train_path).resolve() == Path(args.val_path).resolve():
        parser.error("Use a separate validation dataset")
    torch.manual_seed(args.seed)
    if args.vap_frame_hz is None:
        if args.encoder_checkpoint:
            from maai.checkpoint_encoder import checkpoint_frame_hz
            args.vap_frame_hz = checkpoint_frame_hz(args.encoder_checkpoint)
        else:
            args.vap_frame_hz = 50
    conf = VapConfig.args_to_conf(args)
    if conf.encoder_type != "cpc":
        parser.error("This training entry currently supports the differentiable CPC encoder")
    if conf.sample_rate != 16000:
        parser.error("CPC requires sample_rate=16000")
    if not args.encoder_checkpoint and conf.frame_hz != 50:
        parser.error("The current maai CPC encoder outputs 50 Hz; use --vap_frame_hz 50")
    model = VapGPT_bc(conf)
    model.load_encoder(args.cpc_model, encoder_checkpoint=args.encoder_checkpoint)
    for encoder in (model.encoder1, model.encoder2):
        encoder.encoder.gAR.keepHidden = False
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed = {f"{name}_head.{parameter}" for name in EVENTS[2:]
                   for parameter in ("weight", "bias")}
        if set(missing) - allowed or unexpected:
            raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
        print(f"New heads initialized: {missing}")
    device = torch.device(args.device)
    model.to(device)
    loaders = []
    for path, shuffle in ((args.train_path, True), (args.val_path, False)):
        dataset = VapDataset(path, horizon=max(2.0, model.horizon_time),
                             sample_rate=conf.sample_rate, frame_hz=conf.frame_hz,
                             audio_root_a=args.audio_root_a, audio_root_b=args.audio_root_b,
                             include_event_labels=True)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                                  num_workers=args.num_workers, collate_fn=collate_conversations))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    for epoch in range(1 if args.smoke_test else args.epochs):
        limit = 1 if args.smoke_test else None
        train_metrics = run_epoch(model, loaders[0], device, optimizer, limit)
        val_metrics = run_epoch(model, loaders[1], device, max_batches=limit)
        scheduler.step(val_metrics["total"])
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}))
        payload = {"state_dict": model.state_dict(), "conf": vars(conf), "epoch": epoch,
                   "encoder_spec": {"crop_frames": 0 if args.encoder_checkpoint else 2},
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "val_metrics": val_metrics}
        torch.save(payload, output / "last.ckpt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(payload, output / "best.ckpt")


if __name__ == "__main__":
    main()
