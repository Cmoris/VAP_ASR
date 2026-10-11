# VAP-BC conversation training

`train.py` trains the separate legacy `train/model.py` VAP model. To train
`src/maai/models/vap_bc.py`, use `train_bc.py` instead.

Run from the repository root, using an environment with the package dependencies
and `torchaudio` installed:

```powershell
python train/train_bc.py --val-path "<validation JSONL directory>" --audio-root-a "<A WAV directory>" --audio-root-b "<B WAV directory>" --cpc-model "<CPC checkpoint>" --device cpu --smoke-test
```

The training path defaults to the requested `l3_conv_train_with_backchannel`
directory. Replace the placeholders with real paths. For full training remove
`--smoke-test` and choose `--epochs`, `--batch-size`, and `--device`. The default
batch size is 1; larger batches are right-padded with masked losses. Use a
separate validation split, preferably with no sessions shared with training.
`--checkpoint` accepts original BC weights, allowing only the six missing
weight/bias entries for the three new heads. It initializes fine-tuning rather
than resuming the optimizer. `best.ckpt` and `last.ckpt` save weights,
configuration, epoch, optimizer, scheduler, and validation losses.

The current training entry supports the differentiable CPC encoder at 16 kHz /
50 Hz. Encoder hidden states reset between records. A/B labels retain their
channel order despite the inference model's input swap. The default frozen CPC
backbone stays in evaluation mode; its existing trainable downsampling layers
remain included in optimization.

Alternatively replace `--cpc-model` with
`--encoder-checkpoint vap_state_dict_jp_20hz_20000msec.pt` to strictly load only
the VAP checkpoint's encoder weights. The default frame rate is then taken from
the filename (20 Hz here), and labels/objective use that rate. The two channel
encoders start from the same weights but have separate parameters. With this
loader, `--vap_freeze_encoder 1` freezes the complete encoder including its
downsampling layers; `0` allows fine-tuning. Non-encoder checkpoint parameters
are ignored. Verify the encoder with `train/check_vap_encoder.py`.

| Output head | Frame label |
| --- | --- |
| `bc_detect` | Inside either speaker's `is_back_channel` utterance |
| `bc` | `bc_detect` shifted 0.5 seconds into the future |
| `turn_ending` | Inside an `is_turn_ending` utterance (`is_turn_end` is also accepted) |
| `pause` | Inside an `is_pause` utterance |
| `silence` | Inside an `is_silence` utterance |

These five scalar heads predict a conversation-level flag, consistent with the
original BC heads. Positive flags cover the marked utterance interval, not only
its last frame. Unmarked gaps are negative; silence flags are not inferred from
VAD gaps. Backchannels remain speech in VAD, while explicit silence annotations
are excluded. The unweighted total loss sums VAP cross entropy, VAD BCE, and
the five event BCE losses. Every component and the total are reported for
training and validation; the scheduler and best checkpoint use total validation
loss.

Annotations describe only the current record. The appended VAD horizon has no
known future annotations. The training loss masks VAP frames whose entire future
window is not known, as well as BC frames whose 0.5-second future is unknown.
Very short records may contribute no VAP loss but still train VAD and event
detection. Padding beyond the true record duration contributes no loss.

`VapDataset(..., include_event_labels=True)` enables the additional training
labels and valid-frame count. Its default output remains the original four keys.
`VapGPT_bc.forward(..., return_logits=True)` returns differentiable tensors plus
cache. Default inference still returns the last-frame probability dictionary
(or all frames), now including `p_turn_ending`, `p_pause`, and `p_silence`.
When old weights are loaded, the new heads are random until fine-tuned.

Run the isolated CPU checks with:

```powershell
python train/test_bc_pipeline.py
```

These checks use test doubles for the audio loader, CPC encoder, transformer,
and VAP objective. They exercise the actual dataset, model heads, loss wiring,
masking, and optimizer step. They do not replace the real-data `--smoke-test`.
