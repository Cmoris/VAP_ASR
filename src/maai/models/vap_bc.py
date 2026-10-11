import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import torch.nn.functional as F

from .config import VapConfig
from ..encoder import build_audio_encoder
from ..modules import GPT, GPTStereo
from ..objective import ObjectiveVAP

class VapGPT_bc(nn.Module):
    """Voice Activity Projection with Backchannel (VAP-BC) model.
    
    This model extends VapGPT to specifically handle and predict backchannel
    behaviors during conversations.
    """
    

    def __init__(self, conf: Optional[VapConfig] = None):
        """Initialize the VapGPT_bc model.
        
        Args:
            conf (Optional[VapConfig]): Configuration object.
                If None, default VapConfig is used.
        """
        super().__init__()
        if conf is None:
            conf = VapConfig()
        self.conf = conf
        self.sample_rate = conf.sample_rate
        self.frame_hz = conf.frame_hz

        self.temp_elapse_time = []

        # Single channel
        self.ar_channel = GPT(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.channel_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        # Cross channel
        self.ar = GPTStereo(
            dim=conf.dim,
            dff_k=3,
            num_layers=conf.cross_layers,
            num_heads=conf.num_heads,
            dropout=conf.dropout,
            context_limit=conf.context_limit,
        )

        self.objective = ObjectiveVAP(bin_times=conf.bin_times, frame_hz=conf.frame_hz)

        # Outputs
        # Voice activity objective -> x1, x2 -> logits ->  BCE
        self.va_classifier = nn.Linear(conf.dim, 1)
        
        self.vap_head = nn.Linear(conf.dim, self.objective.n_classes)

        self.bc_detect_head = nn.Linear(conf.dim, 1)

        # For Backchannel
        self.bc_head = nn.Linear(conf.dim, 1)
        self.turn_ending_head = nn.Linear(conf.dim, 1)
        self.pause_head = nn.Linear(conf.dim, 1)
        self.silence_head = nn.Linear(conf.dim, 1)

    def load_encoder(self, cpc_model=None, encoder_checkpoint=None):
        """Load and build the audio encoders for both speakers.
        
        Args:
            cpc_model: Pre-trained CPC model to be used as feature extractor.
        """
        if encoder_checkpoint is not None:
            from ..checkpoint_encoder import load_vap_encoder
            self.encoder1 = load_vap_encoder(encoder_checkpoint, frame_hz=self.frame_hz,
                                             freeze=bool(self.conf.freeze_encoder))
            import copy
            self.encoder2 = copy.deepcopy(self.encoder1)
        else:
            self.encoder1 = build_audio_encoder(self.conf, cpc_model=cpc_model)
            self.encoder2 = build_audio_encoder(self.conf, cpc_model=cpc_model)
        self.encoder1 = self.encoder1.eval()
        self.encoder2 = self.encoder2.eval()

        encoder_dim = getattr(self.encoder1, "output_dim", self.conf.dim)
        if encoder_dim != self.conf.dim:
            self.decrease_dimension = nn.Linear(encoder_dim, self.conf.dim)
        
        if self.conf.freeze_encoder == 1:
            print('freeze encoder')
            self.encoder1.freeze()
            self.encoder2.freeze()

    @property
    def horizon_time(self):
        """Get the horizon time for the projection in seconds.
        
        Returns:
            float: Horizon time for the objective.
        """
        return self.objective.horizon_time

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor) -> Tuple[Tensor, Tensor]:
        """Encode the raw audio inputs into feature representations.
        
        Note: Channel swap is applied for temporal consistency.
        
        Args:
            audio1 (torch.Tensor): Audio waveform for speaker 1 (System).
            audio2 (torch.Tensor): Audio waveform for speaker 2 (User).
            
        Returns:
            Tuple[Tensor, Tensor]: Encoded features for the two speakers.
        """
        
        # Channel swap for temporal consistency
        x1 = self.encoder1(audio2)  # speaker 1 (User)
        x2 = self.encoder2(audio1)  # speaker 2 (System)

        if hasattr(self, "decrease_dimension"):
            x1 = torch.relu(self.decrease_dimension(x1))
            x2 = torch.relu(self.decrease_dimension(x2))

        return x1, x2

    def vad_loss(self, vad_output, vad):
        """Compute the Voice Activity Detection (VAD) loss.
        
        Args:
            vad_output: Predicted VAD logits.
            vad: Ground truth VAD labels.
            
        Returns:
            Tensor: Binary cross-entropy loss between predictions and targets.
        """
        return F.binary_cross_entropy_with_logits(vad_output, vad)
    
    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        cache: Optional[dict] = None,
        return_all_frames: bool = False,
        return_logits: bool = False,
        return_features: bool = False,
    ) -> Tuple[dict, dict]:
        """
        Forward pass for the VapGPT_bc model.

        Args:
            x1 (Tensor): Input audio embedded tensor for speaker 1.
            x2 (Tensor): Input audio embedded tensor for speaker 2.
            cache (dict, optional): Cache of past keys/values.

        Returns:
            Tuple[dict, dict]: Model outputs and updated cache.
        """

        if cache is None:
            cache = {}

        o1 = self.ar_channel(x1, past_kv=cache.get("ar1"))
        o2 = self.ar_channel(x2, past_kv=cache.get("ar2"))
        out = self.ar(
            o1["x"],
            o2["x"],
            past_kv1=cache.get("cross1"),
            past_kv2=cache.get("cross2"),
            past_kv1_c=cache.get("cross1_c"),
            past_kv2_c=cache.get("cross2_c"),
        )

        new_cache = {
            "ar1": (o1["past_k"], o1["past_v"]),
            "ar2": (o2["past_k"], o2["past_v"]),
            "cross1": (out["past_k1"], out["past_v1"]),
            "cross2": (out["past_k2"], out["past_v2"]),
            "cross1_c": (out["past_k1_c"], out["past_v1_c"]),
            "cross2_c": (out["past_k2_c"], out["past_v2_c"]),
        }

        logits = {"logits": self.vap_head(out["x"]),
                  "vad": torch.cat((self.va_classifier(out["x1"]),
                                    self.va_classifier(out["x2"])), dim=-1)}
        for name in ("bc", "bc_detect", "turn_ending", "pause", "silence"):
            logits[name] = getattr(self, f"{name}_head")(out["x"])
        if return_logits:
            if return_features:
                logits["features"] = out["x"]
            return logits, new_cache

        probabilities = {
            f"p_{name}": logits[name].detach().sigmoid().cpu().tolist()[0]
            for name in ("bc", "bc_detect", "turn_ending", "pause", "silence")
        }
        frames = [{name: values[t][0] for name, values in probabilities.items()}
                  for t in range(out["x"].shape[1])]
        ret = frames if return_all_frames else frames[-1]

        return ret, new_cache
