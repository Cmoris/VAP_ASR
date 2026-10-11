# Train and evaluate VAP-ASR-BC

All paths and parameters are edited directly inside `train.sh` and `eval.sh`.
The scripts do not read environment-variable overrides or forward command-line
arguments. Run them from Git Bash, WSL, or Linux after activating your Python
environment. Alternatively edit `PYTHON` to the executable path.

Before training, fill in `VAL_PATH`, `AUDIO_ROOT_A`, and `AUDIO_ROOT_B` at the
top of `train.sh`. Check `ENCODER_CHECKPOINT` points to the Japanese 20 Hz VAP
weights. The training path defaults to `l3_conv_train_with_backchannel`.
Validation should use a separate conversation split. Change epochs, device,
batch size, and other options directly in the Python command inside the script.

```bash
bash scripts/train.sh
bash scripts/eval.sh
```

Evaluation reads `trained_asr_bc/best.ckpt` and its saved Qwen tokenizer by
default. Fill in the audio directories at the top of `eval.sh`. Dataset mode
writes `eval_asr_bc/metrics.json` and `predictions.jsonl` containing event metrics,
ASR CER, references, generated text, and event probabilities.

For two-WAV inference, comment out the `SOURCE_ARGS=(--test-path ...)` line and
uncomment the `SOURCE_ARGS=(--wav-a ... --wav-b ...)` line, then fill in the two
WAV paths. Run the same `bash scripts/eval.sh` command. Results are saved in
`eval_asr_bc/wav_predictions.json`. The recordings should share a time origin.

For a one-batch training check, add `--smoke-test` directly to the training
command. To evaluate fewer records, add `--max-samples 10` directly to the
evaluation command. For an old 50 Hz checkpoint without encoder metadata, add
`--encoder-crop-frames 0` for VAP encoder loading or `2` for original CPC loading.
