"""Load only a legacy VAP CPC encoder; requires PyTorch and no model downloads."""

from pathlib import Path
import re

import torch
from torch import nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dim, 1))
        self.bias = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):
        return ((x - x.mean(1, keepdim=True))
                * torch.rsqrt(x.var(1, keepdim=True) + 1e-5)) * self.weight + self.bias


class CPCConvolutions(nn.Module):
    def __init__(self, dim):
        super().__init__()
        for index, (kernel, stride, padding) in enumerate(
            ((10, 5, 3), (8, 4, 2), (4, 2, 1), (4, 2, 1), (4, 2, 1))
        ):
            setattr(self, f"conv{index}", nn.Conv1d(1 if index == 0 else dim, dim,
                                                  kernel, stride=stride, padding=padding))
            setattr(self, f"batchNorm{index}", ChannelNorm(dim))

    def forward(self, x):
        for index in range(5):
            x = F.relu(getattr(self, f"batchNorm{index}")(getattr(self, f"conv{index}")(x)))
        return x


class CPCRecurrent(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.baseNet = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.keepHidden = False
        self.hidden = None

    def forward(self, x):
        self.baseNet.flatten_parameters()
        output, hidden = self.baseNet(x, self.hidden if self.keepHidden else None)
        self.hidden = tuple(value.detach() for value in hidden) if self.keepHidden else None
        return output


class Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)


class FeatureLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class VAPCheckpointEncoder(nn.Module):
    """Legacy training architecture: CPC CNN -> LSTM -> Conv/LayerNorm/GELU.

    Checkpoints store tensors, not stride or context duration. The requested
    frame rate determines downsample stride. The legacy training encoder did
    not apply the newer inference encoder's two-frame crop. Hidden state resets
    per record by default; context/window management is the caller's job.
    """

    def __init__(self, cnn_dim, recurrent_dim, layers, kernel_size, frame_hz, freeze=True,
                 crop_frames=0):
        super().__init__()
        if frame_hz <= 0:
            raise ValueError("frame_hz must be positive")
        stride = 100 / frame_hz
        if stride < 1 or not stride.is_integer():
            raise ValueError("frame_hz must divide the 100 Hz CPC feature rate")
        self.sample_rate = 16000
        self.frame_hz = frame_hz
        self.crop_frames = crop_frames
        self.downsample_ratio = 160 * int(stride)
        self.output_dim = recurrent_dim
        self.encoder = nn.Module()
        self.encoder.gEncoder = CPCConvolutions(cnn_dim)
        self.encoder.gAR = CPCRecurrent(cnn_dim, recurrent_dim, layers)
        self.downsample = nn.Sequential(
            Transpose(), nn.Conv1d(recurrent_dim, recurrent_dim, kernel_size, stride=int(stride)),
            FeatureLayerNorm(recurrent_dim), nn.GELU(), Transpose(),
        )
        if freeze:
            self.freeze()

    def freeze(self):
        self.requires_grad_(False)

    def unfreeze(self):
        self.requires_grad_(True)

    def reset_streaming_state(self):
        self.encoder.gAR.hidden = None

    def forward_shared(self, waveform):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError("CPC input must have shape (batch, 1, samples)")
        features = self.encoder.gEncoder(waveform).transpose(1, 2)
        if self.crop_frames:
            if self.crop_frames != 2:
                raise ValueError("Supported feature crops are 0 or 2 frames")
            features = features[:, 1:-1]
        return self.encoder.gAR(features)

    def forward_specific(self, features):
        return self.downsample(features)

    def forward(self, waveform):
        return self.forward_specific(self.forward_shared(waveform))

    def output_lengths(self, num_samples):
        lengths = num_samples.clone()
        for index in range(5):
            layer = getattr(self.encoder.gEncoder, f"conv{index}")
            lengths = torch.div(lengths + 2 * layer.padding[0] - layer.kernel_size[0],
                                layer.stride[0], rounding_mode="floor") + 1
        lengths = lengths - self.crop_frames
        layer = self.downsample[1]
        return (torch.div(lengths - layer.kernel_size[0], layer.stride[0],
                          rounding_mode="floor") + 1).clamp_min(0)


def checkpoint_frame_hz(path):
    match = re.search(r"_(\d+(?:\.\d+)?)hz(?:_|\.)", Path(path).name, re.IGNORECASE)
    if not match:
        raise ValueError("Checkpoint tensors do not store frame rate; provide frame_hz explicitly")
    return float(match.group(1))


def load_vap_encoder(path, frame_hz=None, freeze=True):
    """Strictly load encoder.* tensors, ignoring every non-encoder parameter."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    return encoder_from_state_dict(state,
                                   checkpoint_frame_hz(path) if frame_hz is None else float(frame_hz),
                                   freeze=freeze)


def encoder_from_state_dict(state, frame_hz, prefix="encoder.", freeze=True, crop_frames=0):
    """Rebuild an encoder directly from saved tensors without an external CPC file."""
    encoder_state = {key[len(prefix):]: value for key, value in state.items()
                     if key.startswith(prefix) and torch.is_tensor(value)}
    if not encoder_state:
        raise ValueError("Checkpoint has no legacy encoder.* weights")
    cnn_dim = encoder_state["encoder.gEncoder.conv0.weight"].shape[0]
    hidden_dim = encoder_state["encoder.gAR.baseNet.weight_hh_l0"].shape[1]
    gate_dim = encoder_state["encoder.gAR.baseNet.weight_hh_l0"].shape[0]
    if gate_dim != hidden_dim * 4:
        raise ValueError("Only LSTM CPC checkpoints are supported by this loader")
    layers = sum(bool(re.fullmatch(r"encoder.gAR.baseNet.weight_ih_l\d+", key))
                 for key in encoder_state)
    kernel_size = encoder_state["downsample.1.weight"].shape[-1]
    encoder = VAPCheckpointEncoder(cnn_dim, hidden_dim, layers, kernel_size,
                                   float(frame_hz), freeze=freeze, crop_frames=crop_frames)
    encoder.load_state_dict(encoder_state, strict=True)
    return encoder.eval()
