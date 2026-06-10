from utils.jittor_compat import jt
from utils.jittor_compat import nn
import numpy as np
import time
import math
from utils.jittor_compat import F


def pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=True):
    return x


def pad_packed_sequence(x, batch_first=True):
    return x, None


def _lengths_to_list(lengths, max_len):
    if hasattr(lengths, "numpy"):
        lengths = lengths.numpy().tolist()
    return [max(1, min(int(length), int(max_len))) for length in lengths]


def _run_gru_with_lengths(gru, inputs, hidden, lengths, return_sequence=False):
    lengths = _lengths_to_list(lengths, inputs.shape[1])
    max_len = max(lengths)
    seq_outs, last_outs = gru(inputs[:, :max_len], hidden, lengths=lengths)
    return seq_outs if return_sequence else None, last_outs


class TorchCompatibleGRU(nn.Module):
    """Single-layer GRU with PyTorch-compatible parameter names for checkpoints."""

    def __init__(self, input_size, hidden_size, batch_first=True, bidirectional=False):
        super(TorchCompatibleGRU, self).__init__()
        if not batch_first:
            raise NotImplementedError("InterX evaluator only uses batch_first=True")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.weight_ih_l0 = nn.Parameter(jt.randn((hidden_size * 3, input_size)) * 0.02)
        self.weight_hh_l0 = nn.Parameter(jt.randn((hidden_size * 3, hidden_size)) * 0.02)
        self.bias_ih_l0 = nn.Parameter(jt.zeros((hidden_size * 3,)))
        self.bias_hh_l0 = nn.Parameter(jt.zeros((hidden_size * 3,)))
        if bidirectional:
            self.weight_ih_l0_reverse = nn.Parameter(jt.randn((hidden_size * 3, input_size)) * 0.02)
            self.weight_hh_l0_reverse = nn.Parameter(jt.randn((hidden_size * 3, hidden_size)) * 0.02)
            self.bias_ih_l0_reverse = nn.Parameter(jt.zeros((hidden_size * 3,)))
            self.bias_hh_l0_reverse = nn.Parameter(jt.zeros((hidden_size * 3,)))

    def _step(self, x_t, h_t, weight_ih, weight_hh, bias_ih, bias_hh):
        hidden = self.hidden_size
        gates_i = jt.matmul(x_t, weight_ih.transpose(0, 1)) + bias_ih
        gates_h = jt.matmul(h_t, weight_hh.transpose(0, 1)) + bias_hh

        i_r, i_z, i_n = gates_i[:, :hidden], gates_i[:, hidden:2 * hidden], gates_i[:, 2 * hidden:]
        h_r, h_z, h_n = gates_h[:, :hidden], gates_h[:, hidden:2 * hidden], gates_h[:, 2 * hidden:]

        reset_gate = jt.sigmoid(i_r + h_r)
        update_gate = jt.sigmoid(i_z + h_z)
        new_gate = jt.tanh(i_n + reset_gate * h_n)
        return new_gate + update_gate * (h_t - new_gate)

    def _run_direction(self, inputs, h0, reverse=False, lengths=None):
        seq_len = inputs.shape[1]
        h_t = h0
        outputs = []
        indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        lengths_var = None
        if lengths is not None:
            lengths_var = jt.array(np.asarray(lengths, dtype=np.int32))
        if reverse:
            params = (
                self.weight_ih_l0_reverse,
                self.weight_hh_l0_reverse,
                self.bias_ih_l0_reverse,
                self.bias_hh_l0_reverse,
            )
        else:
            params = (self.weight_ih_l0, self.weight_hh_l0, self.bias_ih_l0, self.bias_hh_l0)

        for t in indices:
            h_new = self._step(inputs[:, t, :], h_t, *params)
            if lengths_var is not None:
                active = (lengths_var > t).unsqueeze(1)
                h_t = jt.where(active, h_new, h_t)
                outputs.append(jt.where(active, h_t, jt.zeros_like(h_t)))
            else:
                h_t = h_new
                outputs.append(h_t)
        if reverse:
            outputs = outputs[::-1]
        return jt.stack(outputs, dim=1), h_t

    def execute(self, inputs, hidden=None, lengths=None):
        batch_size = inputs.shape[0]
        if hidden is None:
            hidden = jt.zeros((self.num_directions, batch_size, self.hidden_size), dtype=inputs.dtype)

        forward_out, forward_last = self._run_direction(inputs, hidden[0], reverse=False, lengths=lengths)
        last_states = [forward_last]
        if self.bidirectional:
            backward_out, backward_last = self._run_direction(inputs, hidden[1], reverse=True, lengths=lengths)
            output = jt.cat([forward_out, backward_out], dim=-1)
            last_states.append(backward_last)
        else:
            output = forward_out
        return output, jt.stack(last_states, dim=0)


class ContrastiveLoss(jt.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=3.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def execute(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = jt.mean((1 - label) * jt.pow(euclidean_distance, 2) + (label) * jt.pow(jt.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


def init_weight(m):
    weighted_layers = [nn.Linear]
    for layer_name in ("Conv1d", "ConvTranspose1d"):
        layer_type = getattr(nn, layer_name, None)
        if layer_type is not None:
            weighted_layers.append(layer_type)
    if isinstance(m, tuple(weighted_layers)):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def reparameterize(mu, logvar):
    s_var = jt.exp(logvar * 0.5)
    eps = jt.randn(s_var.shape, dtype=s_var.dtype)
    return eps * s_var + mu


# batch_size, dimension and position
# output: (batch_size, dim)
def positional_encoding(batch_size, dim, pos):
    assert batch_size == pos.shape[0]
    positions_enc = np.array([[pos[j] / np.power(10000, (i - i % 2) / dim) for i in range(dim)] for j in range(batch_size)], dtype=np.float32)
    positions_enc[:, 0::2] = np.sin(positions_enc[:, 0::2])
    positions_enc[:, 1::2] = np.cos(positions_enc[:, 1::2])
    return jt.from_numpy(positions_enc).float()


def get_padding_mask(batch_size, seq_len, cap_lens):
    cap_lens = cap_lens.numpy().tolist()
    mask_2d = jt.ones((batch_size, seq_len, seq_len), dtype=jt.float32)
    for i, cap_len in enumerate(cap_lens):
        mask_2d[i, :, :cap_len] = 0
    return mask_2d.bool(), 1 - mask_2d[:, :, 0].clone()


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=300):
        super(PositionalEncoding, self).__init__()

        pe = jt.zeros(max_len, d_model)
        position = jt.arange(0, max_len, dtype=jt.float).unsqueeze(1)
        div_term = jt.exp(jt.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = jt.sin(position * div_term)
        pe[:, 1::2] = jt.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def execute(self, pos):
        return self.pe[pos]


class MovementConvEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def execute(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)


class MovementConvDecoder(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvDecoder, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose1d(input_size, hidden_size, 4, 2, 1),
            # nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(hidden_size, output_size, 4, 2, 1),
            # nn.Dropout(0.2),
            nn.LeakyReLU(0.2),
        )
        self.out_net = nn.Linear(output_size, output_size)

        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def execute(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return self.out_net(outputs)


class TextVAEDecoder(nn.Module):

    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextVAEDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2))

        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.positional_encoder = PositionalEncoding(hidden_size)

        self.output = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, output_size))

        self.output.apply(init_weight)
        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)

    def get_init_hidden(self, latent):
        hidden = self.z2init(latent)
        hidden = jt.split(hidden, self.hidden_size, dim=-1)
        return list(hidden)

    def execute(self, inputs, last_pred, hidden, p):
        h_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        h_in = h_in + pos_enc
        for i in range(self.n_layers):
            # print(h_in.shape)
            hidden[i] = self.gru[i](h_in, hidden[i])
            h_in = hidden[i]
        pose_pred = self.output(h_in)
        return pose_pred, hidden


class TextDecoder(nn.Module):

    def __init__(self, text_size, input_size, output_size, hidden_size, n_layers):
        super(TextDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.emb = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2))

        self.gru = nn.ModuleList([nn.GRUCell(hidden_size, hidden_size) for i in range(self.n_layers)])
        self.z2init = nn.Linear(text_size, hidden_size * n_layers)
        self.positional_encoder = PositionalEncoding(hidden_size)

        self.mu_net = nn.Linear(hidden_size, output_size)
        self.logvar_net = nn.Linear(hidden_size, output_size)

        self.emb.apply(init_weight)
        self.z2init.apply(init_weight)
        self.mu_net.apply(init_weight)
        self.logvar_net.apply(init_weight)

    def get_init_hidden(self, latent):

        hidden = self.z2init(latent)
        hidden = jt.split(hidden, self.hidden_size, dim=-1)

        return list(hidden)

    def execute(self, inputs, hidden, p):
        # print(inputs.shape)
        x_in = self.emb(inputs)
        pos_enc = self.positional_encoder(p).to(inputs.device).detach()
        x_in = x_in + pos_enc

        for i in range(self.n_layers):
            hidden[i] = self.gru[i](x_in, hidden[i])
            h_in = hidden[i]
        mu = self.mu_net(h_in)
        logvar = self.logvar_net(h_in)
        z = reparameterize(mu, logvar)
        return z, mu, logvar, hidden


class AttLayer(nn.Module):

    def __init__(self, query_dim, key_dim, value_dim):
        super(AttLayer, self).__init__()
        self.W_q = nn.Linear(query_dim, value_dim)
        self.W_k = nn.Linear(key_dim, value_dim, bias=False)
        self.W_v = nn.Linear(key_dim, value_dim)

        self.softmax = nn.Softmax(dim=1)
        self.dim = value_dim

        self.W_q.apply(init_weight)
        self.W_k.apply(init_weight)
        self.W_v.apply(init_weight)

    def execute(self, query, key_mat):
        '''
        query (batch, query_dim)
        key (batch, seq_len, key_dim)
        '''
        # print(query.shape)
        query_vec = self.W_q(query).unsqueeze(-1)  # (batch, value_dim, 1)
        val_set = self.W_v(key_mat)  # (batch, seq_len, value_dim)
        key_set = self.W_k(key_mat)  # (batch, seq_len, value_dim)

        weights = jt.matmul(key_set, query_vec) / np.sqrt(self.dim)

        co_weights = self.softmax(weights)  # (batch, seq_len, 1)
        values = val_set * co_weights  # (batch, seq_len, value_dim)
        pred = values.sum(dim=1)  # (batch, value_dim)
        return pred, co_weights

    def short_cut(self, querys, keys):
        return self.W_q(querys), self.W_k(keys)


class TextEncoderBiGRU(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, device):
        super(TextEncoderBiGRU, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = TorchCompatibleGRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(jt.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def execute(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = _lengths_to_list(cap_lens, input_embs.shape[1])
        gru_seq, gru_last = _run_gru_with_lengths(self.gru, input_embs, hidden, cap_lens, return_sequence=True)

        gru_last = jt.cat([gru_last[0], gru_last[1]], dim=-1)
        gru_seq = pad_packed_sequence(gru_seq, batch_first=True)[0]
        forward_seq = gru_seq[..., :self.hidden_size]
        backward_seq = gru_seq[..., self.hidden_size:].clone()

        # Concate the forward and backward word embeddings
        for i, length in enumerate(cap_lens):
            backward_seq[i:i + 1, :length] = jt.flip(backward_seq[i:i + 1, :length].clone(), dims=[1])
        gru_seq = jt.cat([forward_seq, backward_seq], dim=-1)

        return gru_seq, gru_last


class TextEncoderBiGRUCo(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = TorchCompatibleGRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, output_size))

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(jt.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def execute(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = _lengths_to_list(cap_lens, input_embs.shape[1])
        _, gru_last = _run_gru_with_lengths(self.gru, input_embs, hidden, cap_lens)

        gru_last = jt.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionEncoderBiGRUCo(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = TorchCompatibleGRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.LayerNorm(hidden_size), nn.LeakyReLU(0.2), nn.Linear(hidden_size, output_size))

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(jt.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def execute(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = _lengths_to_list(m_lens, input_embs.shape[1])
        _, gru_last = _run_gru_with_lengths(self.gru, input_embs, hidden, cap_lens)

        gru_last = jt.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)


class MotionLenEstimatorBiGRU(nn.Module):

    def __init__(self, word_size, pos_size, hidden_size, output_size):
        super(MotionLenEstimatorBiGRU, self).__init__()

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = TorchCompatibleGRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        nd = 512
        self.output = nn.Sequential(nn.Linear(hidden_size * 2, nd), nn.LayerNorm(nd), nn.LeakyReLU(0.2), nn.Linear(nd, nd // 2), nn.LayerNorm(nd // 2), nn.LeakyReLU(0.2),
                                    nn.Linear(nd // 2, nd // 4), nn.LayerNorm(nd // 4), nn.LeakyReLU(0.2), nn.Linear(nd // 4, output_size))
        # self.linear2 = nn.Linear(hidden_size, output_size)

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(jt.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def execute(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.numpy().tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = jt.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output(gru_last)
