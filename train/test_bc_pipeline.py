"""Isolated CPU pipeline checks; audio/encoder dependencies are test doubles.

Run with a Python environment containing torch. This is not a real CPC/audio
smoke test; use train_bc.py --smoke-test for that after installing dependencies.
"""
import importlib.util
import ast
import json
from pathlib import Path
import sys
import tempfile
import types

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    value = types.ModuleType(name)
    sys.modules[name] = value
    return value


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def main():
    torch.manual_seed(1)
    # Use the real length method without importing unavailable encoder dependencies.
    encoder_tree = ast.parse((ROOT / "src/maai/encoder.py").read_text(encoding="utf-8"))
    encoder_class = next(node for node in encoder_tree.body
                         if isinstance(node, ast.ClassDef) and node.name == "EncoderCPC")
    length_method = next(node for node in encoder_class.body
                         if isinstance(node, ast.FunctionDef) and node.name == "output_lengths")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[length_method], type_ignores=[]), "encoder_lengths", "exec"), namespace)
    convolutions = [nn.Conv1d(1, 1, k, stride=s, padding=p)
                    for k, s, p in ((10, 5, 3), (8, 4, 2), (4, 2, 1),
                                    (4, 2, 1), (4, 2, 1))]
    downsample = nn.Sequential(nn.Conv1d(1, 1, 5, stride=2))
    length_encoder = types.SimpleNamespace(
        encoder=types.SimpleNamespace(gEncoder=types.SimpleNamespace(
            **{f"conv{i}": layer for i, layer in enumerate(convolutions)})),
        downsample=downsample)
    for sample_count in (8000, 16000, 64013):
        features = torch.zeros(1, 1, sample_count)
        for layer in convolutions:
            features = layer(features)
        actual_frames = downsample(features[:, :, 1:-1]).shape[-1]
        predicted = namespace["output_lengths"](length_encoder, torch.tensor([sample_count]))
        assert predicted.item() == actual_frames
    module("maai").__path__ = [str(ROOT / "src/maai")]
    module("maai.models").__path__ = [str(ROOT / "src/maai/models")]
    module("maai.encoder").build_audio_encoder = lambda *a, **k: None

    class GPT(nn.Module):
        def __init__(self, dim, **kwargs):
            super().__init__()
            self.linear = nn.Linear(dim, dim)

        def forward(self, x, **kwargs):
            return {"x": self.linear(x), "past_k": [], "past_v": []}

    class Stereo(GPT):
        def forward(self, x1, x2, **kwargs):
            result = {"x": self.linear(x1 + x2), "x1": x1, "x2": x2}
            result.update({f"past_{suffix}": [] for suffix in
                           ("k1", "v1", "k2", "v2", "k1_c", "v1_c", "k2_c", "v2_c")})
            return result

    module("maai.modules").GPT = GPT
    sys.modules["maai.modules"].GPTStereo = Stereo

    class Objective:
        def __init__(self, bin_times, frame_hz):
            self.n_classes = 2
            self.horizon_time = sum(bin_times)
            self.projection_window_extractor = types.SimpleNamespace(
                horizon=int(self.horizon_time * frame_hz))

        def get_labels(self, vad):
            horizon = self.projection_window_extractor.horizon
            return (vad[:, horizon:, :].sum(-1) > 0).long()

    module("maai.objective").ObjectiveVAP = Objective
    audio = module("audio")
    audio.time_to_frames = lambda t, hop: int(t / hop)
    audio.load_waveform = lambda path, start_time, end_time, sample_rate, **kw: (
        torch.ones(1, int((end_time - start_time) * sample_rate)), sample_rate)
    sys.path.insert(0, str(ROOT / "train"))
    load("utils", ROOT / "train/utils.py")
    dataset_module = load("conv_dataset", ROOT / "train/conv_dataset.py")
    model_module = load("maai.models.vap_bc", ROOT / "src/maai/models/vap_bc.py")
    trainer = load("train_bc", ROOT / "train/train_bc.py")

    with tempfile.TemporaryDirectory(dir=ROOT / "train") as directory:
        path = Path(directory) / "sample.jsonl"
        utterances = [
            {"speaker": "A", "audio": "a.wav", "start": 0, "end": 1, "text": "你好",
             "is_turn_ending": True, "is_back_channel": True},
            {"speaker": "B", "audio": "b.wav", "start": 1.2, "end": 2, "text": "はい",
             "is_pause": True},
            {"speaker": "A", "audio": "a.wav", "start": 2, "end": 4,
             "is_silence": True},
        ]
        record = [{"content": [{"utterances": utterances}, {"id": "test"}]}]
        encoded = json.dumps(record, ensure_ascii=False).encode("utf-8") + b"\n"
        path.write_bytes(encoded + json.dumps([0]).encode("utf-8") + b"\n")
        original = dataset_module.VapDataset(path, frame_hz=10)[0]
        assert set(original) == {"session", "waveform", "vad", "dataset"}
        dataset = dataset_module.VapDataset(path, frame_hz=10, include_event_labels=True,
                                             include_asr_text=True)
        sample = dataset[0]
        assert sample["turn_ending"][:10].sum() == 10
        assert sample["pause"][12:20].sum() == 8
        assert sample["silence"][20:40].sum() == 20
        assert sample["vad"][20:40].sum() == 0
        assert sample["bc"][:5].sum() == 5
        short = {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}
        short["waveform"] = short["waveform"][:, :16000]
        short["valid_frames"] = 10
        short["valid_samples"] = 16000
        for key in trainer.EVENTS:
            short[key] = short[key][:10]
        short["vad"] = short["vad"][:30]
        batch = trainer.collate_conversations([sample, short])
        conf = trainer.VapConfig(dim=8, frame_hz=10)
        model = model_module.VapGPT_bc(conf)

        class Encoder(nn.Module):
            def reset_streaming_state(self):
                pass

            def output_lengths(self, lengths):
                return lengths // 1600

        model.encoder1 = Encoder()
        model.encoder2 = Encoder()
        model.encode_audio = lambda b, a: (
            torch.ones(a.shape[0], 40, 8), torch.ones(b.shape[0], 40, 8) * 2)
        losses = trainer.compute_losses(model, batch)
        sum(losses.values()).backward()
        for name in trainer.EVENTS:
            gradient = getattr(model, f"{name}_head").weight.grad
            assert gradient is not None and gradient.abs().sum() > 0, name
        changed = {k: v.clone() for k, v in batch.items()}
        for key in trainer.EVENTS:
            changed[key][1, 10:] = 1 - changed[key][1, 10:]
        changed_losses = trainer.compute_losses(model, changed)
        for name in trainer.EVENTS:
            torch.testing.assert_close(losses[name], changed_losses[name])
        prediction, _ = model(torch.ones(1, 4, 8), torch.ones(1, 4, 8))
        assert set(prediction) == {f"p_{name}" for name in trainer.EVENTS}
        optimizer = torch.optim.AdamW(model.parameters())
        metrics = trainer.run_epoch(model, [batch], torch.device("cpu"), optimizer)
        assert metrics["total"] > 0
        # Exercise the actual PyTorch ASR TransformerDecoder with the isolated backbone.
        asr_module = load("maai.models.vap_asr_bc", ROOT / "src/maai/models/vap_asr_bc.py")
        tokenizer_module = load("asr_tokenizer", ROOT / "train/asr_tokenizer.py")
        asr_trainer = load("train_asr_bc", ROOT / "train/train_asr_bc.py")
        # A tiny real HF Qwen tokenizer keeps isolated tests offline and fast.
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from transformers import Qwen2TokenizerFast
        backend = Tokenizer(models.BPE(unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(["你好 はい unknown other words"], trainers.BpeTrainer(
            vocab_size=300, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"]))
        vocab_file, merges_file = backend.model.save(directory)
        raw_tokenizer = Qwen2TokenizerFast(vocab=vocab_file, merges=merges_file,
                                          unk_token="<unk>",
                                          eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                          bos_token=None)
        tokenizer = tokenizer_module.ASRTokenizer(raw_tokenizer)
        assert tokenizer.bos_id == raw_tokenizer.convert_tokens_to_ids("<|im_start|>")
        assert tokenizer.encode([("A", "你好")])[2:-1] == raw_tokenizer.encode(
            "你好", add_special_tokens=False)
        token_path = Path(directory) / "tokenizer"
        tokenizer.save(token_path)
        assert tokenizer_module.ASRTokenizer.load(token_path).to_dict() == tokenizer.to_dict()
        assert tokenizer.decode(tokenizer.encode(sample["asr_segments"])) == "<speaker_A>你好<speaker_B>はい"
        assert tokenizer.decode(tokenizer.encode([("A", "unseen こんにちは")])) == "<speaker_A>unseen こんにちは"
        joint_dataset_module = load("conv_asr_bc_dataset", ROOT / "train/conv_asr_bc_dataset.py")
        joint_dataset = joint_dataset_module.ConvASRBCDataset(path, frame_hz=10, max_tokens=32)
        assert joint_dataset.get_asr_segments(0) == sample["asr_segments"]
        try:
            joint_dataset[0]
        except ValueError:
            pass
        else:
            raise AssertionError("Dataset must require a shared tokenizer before item loading")
        joint_dataset.set_tokenizer(tokenizer)
        joint_item = joint_dataset[0]
        torch.testing.assert_close(joint_item["decoder_input_ids"],
                                   torch.tensor(tokenizer.encode(sample["asr_segments"])[:-1]))
        torch.testing.assert_close(joint_item["asr_labels"],
                                   torch.tensor(tokenizer.encode(sample["asr_segments"])[1:]))
        assert joint_item["waveform"].shape[0] == 2
        actual_loader = torch.utils.data.DataLoader(joint_dataset, batch_size=1,
                                                    collate_fn=joint_dataset.collate_fn)
        actual_batch = next(iter(actual_loader))
        required = {"waveform", "vad", "valid_samples", "valid_frames",
                    "decoder_input_ids", "asr_labels", *trainer.EVENTS}
        assert set(actual_batch) == required and all(torch.is_tensor(v) for v in actual_batch.values())
        assert actual_batch["decoder_input_ids"].dtype == torch.long
        empty_record = [{"content": [{"utterances": [utterances[-1]]}, {"id": "empty"}]}]
        empty_path = Path(directory) / "empty.json"
        empty_path.write_text(json.dumps(empty_record), encoding="utf-8")
        empty_dataset = joint_dataset_module.ConvASRBCDataset(empty_path, frame_hz=10,
                                                              tokenizer=tokenizer)
        empty_item = empty_dataset[0]
        assert empty_item["decoder_input_ids"].tolist() == [tokenizer.bos_id]
        assert empty_item["asr_labels"].tolist() == [tokenizer.eos_id]
        short["asr_segments"] = [("A", "你")]
        asr_batch = joint_dataset.collate_fn([joint_item, short])
        assert asr_batch["decoder_input_ids"][0, 0] == tokenizer.bos_id
        assert asr_batch["asr_labels"][0, 0] == tokenizer.ids["<speaker_A>"]
        assert (asr_batch["asr_labels"][1] == -100).any()
        try:
            asr_trainer.collate_asr_conversations([sample], tokenizer, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Token limit must reject incomplete transcripts")
        asr_conf = asr_module.ASRConfig(len(tokenizer), decoder_layers=1,
                                        num_heads=2, dropout=0.0, max_tokens=32,
                                        pad_id=tokenizer.pad_id, bos_id=tokenizer.bos_id,
                                        eos_id=tokenizer.eos_id)
        joint_model = asr_module.VapGPT_asr_bc(conf, asr_conf)
        joint_model.encoder1, joint_model.encoder2 = Encoder(), Encoder()
        joint_model.encode_audio = model.encode_audio
        joint_losses = asr_trainer.compute_joint_losses(joint_model, asr_batch)
        assert "asr" in joint_losses and torch.isfinite(joint_losses["asr"])
        joint_losses["asr"].backward()
        assert joint_model.asr_head.weight.grad.abs().sum() > 0
        assert joint_model.asr_decoder.layers[0].multihead_attn.in_proj_weight.grad.abs().sum() > 0
        assert joint_model.ar.linear.weight.grad.abs().sum() > 0
        joint_model.zero_grad(set_to_none=True)
        sum(asr_trainer.compute_joint_losses(joint_model, asr_batch).values()).backward()
        for name in trainer.EVENTS:
            assert getattr(joint_model, f"{name}_head").weight.grad.abs().sum() > 0
        joint_model.eval()
        memory = torch.randn(2, 12, 8)
        lengths = torch.tensor([12, 4])
        ids = asr_batch["decoder_input_ids"]
        before = joint_model.decode_asr(memory, ids, lengths)
        future_ids = ids.clone()
        future_ids[0, -1] = raw_tokenizer.encode("x", add_special_tokens=False)[0]
        after = joint_model.decode_asr(memory, future_ids, lengths)
        torch.testing.assert_close(before[0, :-1], after[0, :-1])
        padded_memory = memory.clone()
        padded_memory[1, 4:] = 1000 * torch.randn_like(padded_memory[1, 4:])
        torch.testing.assert_close(before, joint_model.decode_asr(padded_memory, ids, lengths))
        checkpoint_path = Path(directory) / "joint.ckpt"
        torch.save({"state_dict": joint_model.state_dict(), "tokenizer": tokenizer.to_dict()}, checkpoint_path)
        asr_trainer.load_initial_weights(joint_model, checkpoint_path, tokenizer)
        bc_checkpoint_path = Path(directory) / "bc.ckpt"
        torch.save({"state_dict": model.state_dict()}, bc_checkpoint_path)
        asr_trainer.load_initial_weights(joint_model, bc_checkpoint_path, tokenizer)
        reloaded = tokenizer_module.ASRTokenizer.load(token_path)
        reloaded.tokenizer.add_tokens(["<extra_test_token>"], special_tokens=True)
        bad_tokenizer = tokenizer_module.ASRTokenizer(reloaded.tokenizer)
        try:
            asr_trainer.load_initial_weights(joint_model, checkpoint_path, bad_tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched checkpoint vocabulary must be rejected")
        with torch.no_grad():
            joint_model.asr_head.weight.zero_()
            joint_model.asr_head.bias.fill_(-100)
            joint_model.asr_head.bias[tokenizer.eos_id] = 100
        generated = joint_model.generate_asr(memory, lengths, max_new_tokens=3)
        assert generated.shape == (2, 2) and (generated[:, 1] == tokenizer.eos_id).all()
        optimizer = torch.optim.AdamW(joint_model.parameters())
        metrics = trainer.run_epoch(joint_model, [asr_batch], torch.device("cpu"), optimizer,
                                     loss_function=asr_trainer.compute_joint_losses,
                                     loss_weights={"asr": 0.5})
        assert "asr" in metrics and metrics["total"] > 0
        actual_checkpoint = ROOT / "vap_state_dict_jp_20hz_20000msec.pt"
        # A small real CPC encoder fixture makes evaluation checks unconditional.
        if not actual_checkpoint.exists():
            encoder_module = load("maai.checkpoint_encoder", ROOT / "src/maai/checkpoint_encoder.py")
            fixture = encoder_module.VAPCheckpointEncoder(256, 256, 1, 5, 20)
            actual_checkpoint = Path(directory) / "vap_fixture_20hz.pt"
            torch.save({f"encoder.{key}": value for key, value in fixture.state_dict().items()},
                       actual_checkpoint)
        if actual_checkpoint.exists():
            checkpoint_model = asr_module.VapGPT_asr_bc(trainer.VapConfig(dim=8, frame_hz=20), asr_conf)
            original_head = checkpoint_model.bc_head.weight.detach().clone()
            checkpoint_model.load_encoder(encoder_checkpoint=actual_checkpoint)
            torch.testing.assert_close(original_head, checkpoint_model.bc_head.weight)
            assert checkpoint_model.encoder1 is not checkpoint_model.encoder2
            assert checkpoint_model.encoder1.output_dim == 256
            assert not any(p.requires_grad for p in checkpoint_model.encoder1.parameters())
            audio = torch.randn(1, 2, 16000) * 0.01
            encoded_a, encoded_b = checkpoint_model.encode_audio(audio[:, 1:2], audio[:, :1])
            assert encoded_a.shape == encoded_b.shape == (1, 20, 8)
            result, _ = checkpoint_model(encoded_a, encoded_b,
                                          decoder_input_ids=ids[:1],
                                          memory_lengths=torch.tensor([20]))
            assert result["vad"].shape == (1, 20, 2)
            assert torch.isfinite(result["asr_logits"]).all()
            evaluator = load("eval_asr_bc", ROOT / "train/eval_asr_bc.py")
            payload = {"state_dict": checkpoint_model.state_dict(), "conf": vars(checkpoint_model.conf),
                       "asr_conf": vars(asr_conf), "tokenizer": tokenizer.to_dict(),
                       "encoder_spec": {"crop_frames": 0}}
            restored = evaluator.load_eval_model(payload, tokenizer, torch.device("cpu"))
            checkpoint_model.eval()
            before, _ = checkpoint_model(encoded_a, encoded_b, return_logits=True)
            after, _ = restored(encoded_a, encoded_b, return_logits=True)
            torch.testing.assert_close(before["bc"], after["bc"])
            evaluation_dataset = joint_dataset_module.ConvASRBCDataset(path, frame_hz=20,
                                                                        tokenizer=tokenizer, max_tokens=32)
            metrics = evaluator.evaluate_dataset(restored, evaluation_dataset, tokenizer,
                                                  torch.device("cpu"), Path(directory),
                                                  threshold=0.5, max_new_tokens=2)
            assert metrics["samples"] == 1 and metrics["reference_characters"] == 4
            assert set(metrics["events"]) == set(trainer.EVENTS)
            assert evaluator.edit_distance("abc", "adc") == 1
            assert evaluator.event_metrics({"tp": 1, "fp": 0, "fn": 0, "tn": 1})["f1"] == 1
            evaluator.load_waveform = lambda path, sample_rate, mono: (torch.ones(1, 16000), sample_rate)
            wav_result = evaluator.infer_wavs(restored, tokenizer, torch.device("cpu"),
                                              "a.wav", "b.wav", max_new_tokens=2)
            assert len(wav_result["event_probabilities"]["bc"]) == 20
            print("PASS: self-contained checkpoint reload, test-set metrics/CER, dual-WAV inference")
            print("PASS: CPC checkpoint encoder-only initialization, independent towers, 20 Hz features, joint-model forward")
    print("PASS: indexed reading, default output, event labels, padding masks, five head gradients, inference, optimizer step")
    print("PASS: joint dataset/DataLoader format, empty transcripts, CPC encoded lengths, Qwen HF tokenizer, teacher forcing, causal/audio masks, decoder and shared gradients, BC/joint checkpoints, EOS generation, joint optimizer step")


if __name__ == "__main__":
    main()
