#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
# Edit paths and parameters here before running.
PYTHON="python"
TRAIN_OUTPUT_DIR="$ROOT_DIR/trained_asr_bc"
TEST_PATH="$ROOT_DIR/../../finetune_labels_whisper/finetune_labels_whisperx/l3_conv_test_with_backchannel"
AUDIO_ROOT_A="/path/to/A_wavs"
AUDIO_ROOT_B="/path/to/B_wavs"
OUTPUT_DIR="$ROOT_DIR/eval_asr_bc"

# Test dataset by default. For two-WAV inference, replace this line with the next one.
SOURCE_ARGS=(--test-path "$TEST_PATH")
# SOURCE_ARGS=(--wav-a "/path/to/speaker_a.wav" --wav-b "/path/to/speaker_b.wav")

"$PYTHON" -u train/eval_asr_bc.py \
    --checkpoint "$TRAIN_OUTPUT_DIR/best.ckpt" \
    --tokenizer-path "$TRAIN_OUTPUT_DIR/tokenizer" \
    "${SOURCE_ARGS[@]}" \
    --audio-root-a "$AUDIO_ROOT_A" \
    --audio-root-b "$AUDIO_ROOT_B" \
    --output-dir "$OUTPUT_DIR" \
    --device cpu \
    --num-workers 0 \
    --max-new-tokens 256 \
    --threshold 0.5
