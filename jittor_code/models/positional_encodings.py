"""Jittor-native positional encodings used by InterDist models."""

import math
from utils.jittor_compat import jt, nn


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.org_channels = channels
        self.channels = int(math.ceil(channels / 4) * 2)
        inv_freq = 1.0 / (10000 ** (jt.arange(0, self.channels, 2).float32() / self.channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def execute(self, tensor):
        if len(tensor.shape) != 4:
            raise RuntimeError("PositionalEncoding2D expects [B, H, W, C]")
        if self.cached_penc is not None and tuple(self.cached_penc.shape) == tuple(tensor.shape):
            return self.cached_penc

        batch_size, x_len, y_len, orig_ch = tensor.shape
        pos_x = jt.arange(x_len).float32()
        pos_y = jt.arange(y_len).float32()
        sin_inp_x = pos_x.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        sin_inp_y = pos_y.unsqueeze(1) * self.inv_freq.unsqueeze(0)

        emb_x = self._get_emb(sin_inp_x).reshape(x_len, 1, self.channels).expand(x_len, y_len, self.channels)
        emb_y = self._get_emb(sin_inp_y).reshape(1, y_len, self.channels).expand(x_len, y_len, self.channels)
        emb = jt.zeros((x_len, y_len, self.channels * 2), dtype=tensor.dtype)
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels:self.channels * 2] = emb_y

        self.cached_penc = emb.unsqueeze(0)[:, :, :, :orig_ch].repeat(batch_size, 1, 1, 1)
        return self.cached_penc

    @staticmethod
    def _get_emb(sin_inp):
        return jt.stack((jt.sin(sin_inp), jt.cos(sin_inp)), dim=-1).reshape(sin_inp.shape[0], -1)
