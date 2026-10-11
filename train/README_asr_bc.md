# Joint ASR and conversation-event training

`src/maai/models/vap_asr_bc.py` defines `VapGPT_asr_bc`, extending `VapGPT_bc`.
The original CPC encoder, channel/cross-channel transformers, VAP/VAD heads,
and five event heads are shared with the ASR task. A new PyTorch
`TransformerDecoder` attends to the fused VAP audio features and predicts text
tokens. Its default is four layers, four attention heads, and feedforward width
four times the VAP feature dimension. Text and audio memory receive sinusoidal
positions. Decoder self-attention is causal; cross-attention can use the whole
valid audio record. This ASR path transcribes complete records, not streaming
audio. Original BC-style inference remains available without decoder inputs.

Run from the repository root, after installing the MaAI package dependencies and
`torchaudio` in the Python environment:

```powershell
python train/train_asr_bc.py --val-path "<validation annotation directory>" --audio-root-a "<A WAV directory>" --audio-root-b "<B WAV directory>" --cpc-model "<CPC checkpoint>" --device cpu --smoke-test
```

The default training annotations are the requested
`l3_conv_train_with_backchannel` directory. Replace the placeholders with real
paths. Training and validation must use separate datasets, ideally with disjoint
conversation sessions. For a full run remove `--smoke-test`; set `--epochs`,
`--batch-size`, and `--device`. Start with batch size 1 because both audio and
transcript lengths vary. Original CPC loading uses 16 kHz / 50 Hz. A VAP
encoder-only checkpoint can use the frame rate indicated in its filename.

To initialize only the encoder from the local Japanese 20 Hz VAP weights,
replace `--cpc-model` with:

```powershell
--encoder-checkpoint vap_state_dict_jp_20hz_20000msec.pt
```

This strictly loads only its 28 `encoder.*` tensors (CNN, LSTM, downsampling),
into two independent A/B encoders. VAP transformers, event heads, and the ASR
decoder remain newly initialized unless `--checkpoint` is separately supplied.
The filename supplies the default 20 Hz rate; dataset labels and model objective
use that same rate. `--vap_frame_hz` overrides it; checkpoint tensors do not
encode convolution stride or context duration. The legacy encoder uses no
two-frame crop and resets LSTM state between training records. The `20000msec`
filename component does not impose a 20-second input limit in this loader.
`--vap_freeze_encoder 1` freezes the entire loaded encoder, including downsampling;
use `0` to fine-tune it.

The encoder can be checked without MaAI GUI/audio-library dependencies:

```powershell
python train/check_vap_encoder.py vap_state_dict_jp_20hz_20000msec.pt --wav train/sample/jpn_mix_16k.wav
```

`train/conv_asr_bc_dataset.py` provides `ConvASRBCDataset`, used directly by
the joint trainer. The existing JSON/JSONL annotations require no conversion.
Each item includes:

| Field | Item shape/type | Purpose |
| --- | --- | --- |
| `waveform` | float tensor `(2, samples)` | A/B audio |
| `vad` | float tensor `(frames + horizon, 2)` | Voice activity targets |
| `bc`, `bc_detect`, `turn_ending`, `pause`, `silence` | float tensors `(frames, 1)` | Event targets |
| `valid_samples`, `valid_frames` | integers | True lengths before padding |
| `decoder_input_ids` | long tensor `(tokens,)` | BOS plus right-shifted transcript |
| `asr_labels` | long tensor `(tokens,)` | Next-token targets ending in EOS |
| `session`, `dataset`, `asr_segments` | metadata | Record identity and sorted text |

`dataset.collate_fn` produces a dictionary of tensors with a batch dimension,
audio/event padding, and `-100` for padded ASR targets. It omits metadata from
the training batch. The trainer loads one Qwen tokenizer, binds it to both
datasets, and uses this collator. Empty transcripts train BOS-to-EOS. The
original `VapDataset` still returns its four original fields by default.

For direct dataset use:

```python
from conv_asr_bc_dataset import ConvASRBCDataset
from asr_tokenizer import ASRTokenizer
from torch.utils.data import DataLoader

train_ds = ConvASRBCDataset(train_path, audio_root_a=root_a, audio_root_b=root_b)
tokenizer = ASRTokenizer.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
train_ds.set_tokenizer(tokenizer)
val_ds = ConvASRBCDataset(val_path, tokenizer=tokenizer,
                          audio_root_a=root_a, audio_root_b=root_b)
loader = DataLoader(train_ds, batch_size=2, collate_fn=train_ds.collate_fn)
```

This snippet assumes the `train` directory is on the Python import path.

Transcripts come from the user-side `utterances[].text`, sorted by start/end
time. Empty text and explicitly marked silence are excluded. Speech in pause or
backchannel utterances remains in the transcript. Each utterance starts with a
structural `<speaker_A>` or `<speaker_B>` token. These are registered as special
tokens in Qwen's tokenizer. There are no event tags in the ASR target: the event
heads receive their own frame-level supervision.

`asr_tokenizer.py` directly delegates text encoding/decoding to Hugging Face
`AutoTokenizer`, defaulting to `Qwen/Qwen2-Audio-7B-Instruct`. The pretrained
Qwen vocabulary is shared across training and validation; no vocabulary is
built from the dataset. Use `--tokenizer-name-or-path` to select another Qwen
model ID or a local tokenizer directory. `--tokenizer-local-files-only` disables
downloads. Only tokenizer files are loaded, not Qwen model weights; the ASR
decoder remains initialized from scratch.

For the default Qwen tokenizer, PAD is its existing `<|endoftext|>`, BOS is
registered as `<|im_start|>`, and EOS is its existing `<|im_end|>`. Other Qwen
tokenizers retain their existing distinct IDs or receive a separate PAD/BOS
when necessary. A/B speaker tokens are added without replacing Qwen's existing
audio/chat special tokens. Decoder embedding/output dimensions use
`len(tokenizer)` including added tokens, and special IDs are supplied explicitly
to `ASRConfig`. Do not assume PAD/BOS/EOS are 0/1/2.

Teacher forcing uses `[BOS, speaker, text, ...]` as decoder inputs and
`[speaker, text, ..., EOS]` as targets. Target padding is `-100` and excluded from
cross entropy. Sequences exceeding `--asr-max-tokens` (default 2048 shifted
tokens) fail explicitly; they are not silently truncated. Increase the limit or
split those annotation records.

Audio padding is excluded from ASR attention using lengths computed from the
CPC convolution stack, feature crop, and downsampling layers. This accounts for
the difference between nominal 50 Hz duration and actual encoded frame count.
Records too short to produce any real encoded frame fail explicitly. Event,
VAD, and VAP losses retain the BC training masks for padding and unknown future.

The total loss is:

```text
vap_weight * VAP_CE + vad_weight * VAD_BCE
+ event_weight * (BC_prediction_BCE + BC_detection_BCE
                  + turn_ending_BCE + pause_BCE + silence_BCE)
+ asr_weight * ASR_token_CE
```

Control these with `--vap-loss-weight`, `--vad-loss-weight`,
`--event-loss-weight`, and `--asr-loss-weight` (all default to 1). ASR and event
weights must be positive for joint training. Each loss component and weighted
total are reported on training and validation. The LR scheduler and best-model
selection monitor total validation loss. Validation currently reports losses;
it does not calculate CER/WER. Events and ASR update the shared VAP transformers
together; the CPC backbone follows `--vap_freeze_encoder`.

The output directory (default `trained_asr_bc`) contains a `tokenizer/` directory,
`last.ckpt`, and `best.ckpt`. Checkpoints include model weights, VAP/ASR configs,
tokenizer vocabulary/rule fingerprints, loss weights, optimizer, scheduler,
epoch, and validation losses. The tokenizer directory uses HF `save_pretrained`
files plus `asr_tokenizer_config.json`. Reload it with
`--tokenizer-path trained_asr_bc/tokenizer`. Old character-tokenizer ASR
checkpoints are incompatible; BC-only checkpoints can still initialize the
shared backbone and event heads.
`--checkpoint` initializes from BC or ASR-BC weights; it does not resume optimizer
training. When using an ASR checkpoint, pass its saved tokenizer and decoder
configuration; vocabulary mismatches are rejected. BC checkpoints may omit ASR
parameters and the three new event heads; other missing/unexpected keys fail.

For direct model use, instantiate `VapGPT_asr_bc(conf, ASRConfig(...))`, load its
encoder and checkpoint, then call:

```python
out, cache = model(x1, x2, decoder_input_ids=input_ids,
                   memory_lengths=encoded_lengths, return_logits=True)
# out contains the existing event/VAP/VAD logits plus asr_logits.
```

For greedy ASR inference, call `model.eval()`, request audio features with
`model(x1, x2, return_logits=True, return_features=True)`, and pass
`out["features"]` plus the valid encoded lengths to `model.generate_asr`.
Decode the returned token IDs with the saved `ASRTokenizer`. Generation stops
at EOS or the configured maximum number of new tokens.

`python train/test_bc_pipeline.py` runs isolated CPU checks, including the actual
ASR decoder, a small offline HF Qwen tokenizer, shifted targets, causal/audio masks, gradients
into decoder and shared layers, vocabulary compatibility, EOS generation, and a
joint optimizer step. Audio loading, CPC, VAP transformers, and the VAP objective
remain test doubles there. Use the real-data `--smoke-test` to validate the full
installed audio pipeline.
