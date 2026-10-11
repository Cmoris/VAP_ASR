"""Joint training of VAP, conversation events, and autoregressive ASR."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_bc import EVENTS, compute_losses, run_epoch, VapConfig
from conv_asr_bc_dataset import DEFAULT_DATASET_PATH, ConvASRBCDataset, collate_asr_conversations
from asr_tokenizer import ASRTokenizer, DEFAULT_QWEN_TOKENIZER
from maai.models.vap_asr_bc import ASRConfig, VapGPT_asr_bc


def compute_joint_losses(model, batch):
    for encoder in (model.encoder1, model.encoder2):
        encoder.reset_streaming_state()
    x1, x2 = model.encode_audio(batch["waveform"][:, 1:2], batch["waveform"][:, :1])
    # Account for CPC convolution boundaries and cropping, not just nominal Hz.
    memory_lengths = model.encoder1.output_lengths(batch["valid_samples"]).clamp(max=x1.shape[1])
    out, _ = model(x1, x2, decoder_input_ids=batch["decoder_input_ids"],
                   memory_lengths=memory_lengths, return_logits=True)
    losses = compute_losses(model, batch, out=out)
    losses["asr"] = F.cross_entropy(out["asr_logits"].transpose(1, 2),
                                    batch["asr_labels"], ignore_index=-100)
    return losses


def load_initial_weights(model, checkpoint, tokenizer):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "tokenizer" in payload and payload["tokenizer"] != tokenizer.to_dict():
        raise ValueError("Checkpoint vocabulary differs from tokenizer; use its saved tokenizer directory")
    state = payload.get("state_dict", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {key for key in model.state_dict() if key.startswith("asr_")}
    allowed.update(f"{name}_head.{parameter}" for name in EVENTS[2:]
                   for parameter in ("weight", "bias"))
    if set(missing) - allowed or unexpected:
        raise ValueError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
    print(f"Initialized {len(missing)} new parameter tensors")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--audio-root-a")
    parser.add_argument("--audio-root-b")
    encoder_source = parser.add_mutually_exclusive_group(required=True)
    encoder_source.add_argument("--cpc-model", help="Original CPC checkpoint")
    encoder_source.add_argument("--encoder-checkpoint", help="VAP checkpoint; load encoder weights only")
    parser.add_argument("--checkpoint", help="BC or ASR-BC initialization weights; not optimizer resume")
    parser.add_argument("--tokenizer-name-or-path", default=DEFAULT_QWEN_TOKENIZER,
                        help="Qwen Hugging Face model ID or local tokenizer directory")
    parser.add_argument("--tokenizer-path", help="Saved ASR tokenizer directory; overrides tokenizer-name-or-path")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--output-dir", default="trained_asr_bc")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=3.63e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--asr-decoder-layers", type=int, default=4)
    parser.add_argument("--asr-num-heads", type=int, default=4)
    parser.add_argument("--asr-ff-multiplier", type=int, default=4)
    parser.add_argument("--asr-dropout", type=float, default=0.1)
    parser.add_argument("--asr-max-tokens", type=int, default=2048)
    parser.add_argument("--asr-loss-weight", type=float, default=1.0)
    parser.add_argument("--event-loss-weight", type=float, default=1.0)
    parser.add_argument("--vap-loss-weight", type=float, default=1.0)
    parser.add_argument("--vad-loss-weight", type=float, default=1.0)
    parser, _ = VapConfig.add_argparse_args(parser)
    parser.set_defaults(vap_frame_hz=None)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.num_workers < 0:
        parser.error("Positive batch-size/epochs and nonnegative num-workers are required")
    if Path(args.train_path).resolve() == Path(args.val_path).resolve():
        parser.error("Use a separate validation dataset")
    weights = {name: args.event_loss_weight for name in EVENTS}
    weights.update(asr=args.asr_loss_weight, vap=args.vap_loss_weight, vad=args.vad_loss_weight)
    if min(weights.values()) < 0 or args.asr_loss_weight <= 0 or args.event_loss_weight <= 0:
        parser.error("Joint training requires positive ASR/event weights and nonnegative VAP/VAD weights")
    torch.manual_seed(args.seed)
    if args.vap_frame_hz is None:
        if args.encoder_checkpoint:
            from maai.checkpoint_encoder import checkpoint_frame_hz
            args.vap_frame_hz = checkpoint_frame_hz(args.encoder_checkpoint)
        else:
            args.vap_frame_hz = 50
    conf = VapConfig.args_to_conf(args)
    if conf.encoder_type != "cpc" or conf.sample_rate != 16000:
        parser.error("Current CPC training requires encoder_type=cpc and sample_rate=16000")
    if not args.encoder_checkpoint and conf.frame_hz != 50:
        parser.error("Original CPC loading requires frame_hz=50; use --encoder-checkpoint for VAP weights")
    datasets = [ConvASRBCDataset(path, horizon=max(2.0, sum(conf.bin_times)),
                           sample_rate=conf.sample_rate, frame_hz=conf.frame_hz,
                           audio_root_a=args.audio_root_a, audio_root_b=args.audio_root_b,
                           max_tokens=args.asr_max_tokens)
                for path in (args.train_path, args.val_path)]
    tokenizer = (ASRTokenizer.load(args.tokenizer_path,
                                   local_files_only=args.tokenizer_local_files_only)
                 if args.tokenizer_path else ASRTokenizer.from_pretrained(
                     args.tokenizer_name_or_path,
                     local_files_only=args.tokenizer_local_files_only))
    for dataset in datasets:
        dataset.set_tokenizer(tokenizer)
    asr_conf = ASRConfig(vocab_size=len(tokenizer), decoder_layers=args.asr_decoder_layers,
                         num_heads=args.asr_num_heads, ff_multiplier=args.asr_ff_multiplier,
                         dropout=args.asr_dropout, max_tokens=args.asr_max_tokens,
                         pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id)
    model = VapGPT_asr_bc(conf, asr_conf)
    model.load_encoder(args.cpc_model, encoder_checkpoint=args.encoder_checkpoint)
    for encoder in (model.encoder1, model.encoder2):
        encoder.encoder.gAR.keepHidden = False
    if args.checkpoint:
        load_initial_weights(model, args.checkpoint, tokenizer)
    device = torch.device(args.device)
    model.to(device)
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=(index == 0),
                          num_workers=args.num_workers, collate_fn=dataset.collate_fn)
               for index, dataset in enumerate(datasets)]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(output / "tokenizer")
    best = float("inf")
    for epoch in range(1 if args.smoke_test else args.epochs):
        limit = 1 if args.smoke_test else None
        train_metrics = run_epoch(model, loaders[0], device, optimizer, limit,
                                  loss_function=compute_joint_losses, loss_weights=weights)
        val_metrics = run_epoch(model, loaders[1], device, max_batches=limit,
                                loss_function=compute_joint_losses, loss_weights=weights)
        scheduler.step(val_metrics["total"])
        print(json.dumps({"epoch": epoch, "train": train_metrics, "val": val_metrics}))
        payload = {"state_dict": model.state_dict(), "conf": vars(conf),
                   "asr_conf": asdict(asr_conf), "tokenizer": tokenizer.to_dict(),
                   "loss_weights": weights, "epoch": epoch,
                   "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                   "val_metrics": val_metrics}
        torch.save(payload, output / "last.ckpt")
        if val_metrics["total"] < best:
            best = val_metrics["total"]
            torch.save(payload, output / "best.ckpt")


if __name__ == "__main__":
    main()
