"""Native latent supervision for a known first frame and a future continuation."""
from __future__ import annotations
import torch


def future_velocity_loss(prediction, target, known_prefix=1):
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError('Expected matching B,C,T,H,W velocity tensors')
    if type(known_prefix) is not int or not 0 <= known_prefix < prediction.shape[2]:
        raise ValueError('Known prefix must leave future latent frames to supervise')
    # A separately supplied keyframe is never taken from future target latents.
    # For these causal VAEs, temporal latent zero represents the known frame.
    return torch.nn.functional.mse_loss(prediction[:,:,known_prefix:].float(),
                                        target[:,:,known_prefix:].float())
