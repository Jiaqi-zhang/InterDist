import copy
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _get_clone(module):
    return copy.deepcopy(module)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def modulate(x, shift, scale, batch_first=True):
    if batch_first:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class AdaLNModulation(nn.Module):

    def __init__(self, d_model, ntag=3, nchunks=7, input_dim=None, output_dim=None):
        super(AdaLNModulation, self).__init__()
        self.nchunks = nchunks
        input_dim = d_model * ntag if input_dim is None else input_dim
        output_dim = nchunks * d_model if output_dim is None else output_dim

        self.model = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_dim, output_dim, bias=True),
        )

    def forward(self, cond):
        return self.model(cond).chunk(self.nchunks, dim=-1)


class SelfAttnBlock(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=True):
        super(SelfAttnBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, shift, scale, gate, src_key_padding_mask=None, pre_normalize=True):
        if pre_normalize:
            src_mod = modulate(self.norm(src), shift, scale)
            src_mod = self.attention(src_mod, src_mod, src_mod, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
            src = src + gate.unsqueeze(1) * self.dropout(src_mod)
        else:
            src_mod = self.attention(src, src, src, key_padding_mask=src_key_padding_mask, need_weights=False)[0]
            src = src + gate.unsqueeze(1) * self.dropout(src_mod)
            src = modulate(self.norm(src), shift, scale)
        return src


class CrossAttnBlock(nn.Module):

    def __init__(self, d_model, nhead, dropout, batch_first=True):
        super(CrossAttnBlock, self).__init__()

        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, shift, scale, gate, memory_key_padding_mask=None, pre_normalize=True):
        if pre_normalize:
            tgt_mod = modulate(self.norm(tgt), shift, scale)
            tgt_mod = self.attention(
                query=tgt_mod,
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
            tgt = tgt + gate.unsqueeze(1) * self.dropout(tgt_mod)
        else:
            tgt_mod = self.attention(
                query=tgt,
                key=memory,
                value=memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
            tgt = tgt + gate.unsqueeze(1) * self.dropout(tgt_mod)
            tgt = modulate(self.norm(tgt), shift, scale)
        return tgt


class FFN(nn.Module):

    def __init__(self, d_model, dim_feedforward, dropout):
        super(FFN, self).__init__()        
        self.up = nn.Linear(d_model, dim_feedforward, bias=False)
        self.gate = nn.Linear(d_model, dim_feedforward, bias=False)
        self.down = nn.Linear(dim_feedforward, d_model, bias=False)

        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

    def forward(self, src, shift, scale, gate, pre_normalize=True):
        if pre_normalize:
            src_mod = modulate(self.ffn_norm(src), shift, scale)
            # SwiGLU
            src_mod = self.down(F.silu(self.up(src_mod)) * self.gate(src_mod)) 
            src_mod = self.ffn_dropout(src_mod)
            src = src + gate.unsqueeze(1) * src_mod
        else:
            # SwiGLU
            src_mod = self.down(F.silu(self.up(src)) * self.gate(src))            
            src = src + gate.unsqueeze(1) * self.ffn_dropout(src_mod)
            src = self.ffn_norm(src)
            src = modulate(src, shift, scale)
        return src


class MLP(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


# *************************************************************************************************** #


class TextCondTransformerDecoderLayer(nn.Module):

    def __init__(self, nt, nbp, mask_id, d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True):
        super().__init__()
        self.nt = nt
        self.nbp = nbp
        self.mask_id = mask_id
        self.d_model = d_model
        self.batch_first = batch_first
        assert self.batch_first == True

        self.text_adaLN = AdaLNModulation(d_model, 1, 9)
        self.text_sa = SelfAttnBlock(d_model, nhead, dropout, batch_first)
        self.words_ca = CrossAttnBlock(d_model, nhead, dropout, batch_first)
        self.ffn = FFN(d_model, dim_feedforward, dropout)

    def forward(self, x_ids, sent, words, tgt_key_padding_mask):
        '''
        x_ids: (bs, seq_len*5, 3*embed_dim=latent_dim)
        sent: (bs, latent_dim)
        words: (bs, 77, latent_dim)
        tgt_key_padding_mask: (bs, seq_len*5)
        '''

        # ======> AdaLN modulation
        cond_scale = self.text_adaLN(sent)  # (bs, d_model)*12
        
        # ======> self-attention
        x_ids = self.text_sa(
            x_ids,
            cond_scale[0],
            cond_scale[1],
            cond_scale[2],
            src_key_padding_mask=tgt_key_padding_mask,
        )
        
        # ======> words cross-attention
        x_ids = self.words_ca(
            x_ids,
            words,
            cond_scale[3],
            cond_scale[4],
            cond_scale[5],
        )

        # ======> FFN
        shift_ffn_c, scale_ffn_c, gate_ffn_c = cond_scale[6:]
        x_ids = self.ffn(x_ids, shift_ffn_c, scale_ffn_c, gate_ffn_c)  # (bs, seq_len*nbp, d_model)
        return x_ids


class MotCondTransformerDecoderLayer(nn.Module):

    def __init__(self, nt, nbp, mask_id, d_model, nhead, dim_feedforward=1024, dropout=0.1, batch_first=True):
        super().__init__()
        self.nt = nt
        self.nbp = nbp
        self.mask_id = mask_id
        self.d_model = d_model
        self.batch_first = batch_first
        assert self.batch_first == True

        # input_dim: latent_dim + embed_dim = embed_dim*4
        self.mot_adaLN = AdaLNModulation(-1, -1, 6, input_dim=d_model*4, output_dim=d_model*6)
        
        self.expert_model = MLP(in_features=d_model, hidden_features=d_model*2)
        self.router_model = MLP(in_features=d_model, hidden_features=d_model*2, out_features=2)
        self.fus_ca = CrossAttnBlock(d_model, nhead, dropout, batch_first)
        self.ffn = FFN(d_model, dim_feedforward, dropout)

    def forward(self, x_ids, cond_tag, tgt_key_padding_mask, eps=1e-6):
        '''
        x_ids: (bs, 3, seq_len*5, embed_dim)
        words: (bs, 77, latent_dim)
        cond_tag: (bs, 3, latent_dim+embed_dim)
        tgt_key_padding_mask: (bs, seq_len*5)
        '''
        
        # ======> AdaLN modulation
        cond_scale = self.mot_adaLN(cond_tag)  # (bs, 3, d_model) * 6

        # ======> motion & distance fusion        
        x_expert = self.expert_model(x_ids)  # (bs, 3, seq, dim)
        mot1_ex, mot2_ex, dist_ex = x_expert[:, 0], x_expert[:, 1], x_expert[:, 2]  # (bs, seq, dim)

        output_lst = []
        mot1, mot2, dist = x_ids[:, 0], x_ids[:, 1], x_ids[:, 2]  # (bs, seq, dim)
        for i, data, cond_mot in zip([0, 1, 2], [mot1, mot2, dist], [[mot2_ex, dist_ex], [mot1_ex, dist_ex], [mot1_ex, mot2_ex]]):

            # data + x_combined -> conditional routing weights
            routing_weights = self.router_model(data).sigmoid()  # (bs, seq, 2)
            routing_weights = routing_weights / (routing_weights.sum(dim=-1, keepdim=True) + eps)
            routing_weights = routing_weights[..., None]  # (bs, seq, 2, 1)

            cond_mot = torch.stack(cond_mot, dim=2)  # (bs, seq, 2, dim)
            cond_mot = cond_mot * routing_weights
            cond_mot = cond_mot.sum(dim=2)  # (bs, seq, dim)

            data_scale = [t[:, i, :] for t in cond_scale]  # 6*(bs, dim)
            data = self.fus_ca(
                data,
                cond_mot,
                data_scale[0],
                data_scale[1],
                data_scale[2],
                memory_key_padding_mask=tgt_key_padding_mask,
            )
            # ======> FFN
            shift_ffn_c, scale_ffn_c, gate_ffn_c = data_scale[3:]
            data = self.ffn(data, shift_ffn_c, scale_ffn_c, gate_ffn_c)  # (bs, seq, dim)
            output_lst.append(data)

        x_ids = torch.stack(output_lst, dim=1)  # (bs, 3, seq, dim)
        return x_ids


class CondTransformerDecoder(nn.Module):

    def __init__(self, text_decoder_layer, mot_decoder_layer, num_layers):
        super().__init__()
        self.text_layers = _get_clones(text_decoder_layer, num_layers)
        self.mot_layers = _get_clones(mot_decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def text_forward(self, x_ids, cond, cond_word, tgt_key_padding_mask: Optional[Tensor] = None):
        for tlayer in self.text_layers:
            x_ids = tlayer(x_ids, cond, cond_word, tgt_key_padding_mask)
        return x_ids

    def fus_forward(self, x_ids, cond_tag, tgt_key_padding_mask: Optional[Tensor] = None):
        for mlayer in self.mot_layers:
            x_ids = mlayer(x_ids, cond_tag, tgt_key_padding_mask)
        return x_ids
