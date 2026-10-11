#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Edit paths here before running. Use a separate validation split.
PYTHON="python"
DATASET_ROOT="$ROOT_DIR/../../finetune_labels_whisper/finetune_labels_whisperx"
TRAIN_PATH="$DATASET_ROOT/l3_conv_train_with_backchannel"
VAL_PATH="/path/to/validation_annotations"
AUDIO_ROOT_A="/path/to/A_wavs"
AUDIO_ROOT_B="/path/to/B_wavs"
ENCODER_CHECKPOINT="$ROOT_DIR/vap_state_dict_jp_20hz_20000msec.pt"
OUTPUT_DIR="$ROOT_DIR/trained_asr_bc"

"$PYTHON" -u train/train_asr_bc.py \
    --train-path "$TRAIN_PATH" \
    --val-path "$VAL_PATH" \
    --audio-root-a "$AUDIO_ROOT_A" \
    --audio-root-b "$AUDIO_ROOT_B" \
    --encoder-checkpoint "$ENCODER_CHECKPOINT" \
    --tokenizer-name-or-path "Qwen/Qwen2-Audio-7B-Instruct" \
    --output-dir "$OUTPUT_DIR" \
    --device cpu \
    --batch-size 1 \
    --num-workers 0 \
    --epochs 25 \
    --learning-rate 0.000363 \
    --vap_freeze_encoder 1 \
    --seed 1
