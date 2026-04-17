import numpy as np

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from models.mask_transformer.tools import *
from models.mask_transformer.cond_transformer import AdaLNModulation, modulate
from models.mask_transformer.cond_transformer import CondTransformerDecoder
from models.mask_transformer.cond_transformer import TextCondTransformerDecoderLayer, MotCondTransformerDecoderLayer
from positional_encodings.torch_encodings import PositionalEncoding2D


class SpaTempPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(SpaTempPositionalEncoding, self).__init__()
        self.positional_encoding = PositionalEncoding2D(d_model)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (bs, part, seqen, input_feats)
            x = x + self.positional_encoding(x)  # (bs, part, seqen, input_feats)
            x = x.squeeze(1)  # (bs, seqen, input_feats)
        else:
            x = x + self.positional_encoding(x)  # (bs, part, seqen, input_feats)
        return x


class InputProcess(nn.Module):

    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = self.poseEmbedding(x)  # [bs, seqlen, d]
        return x


class OutputProcess(nn.Module):

    def __init__(self, out_feats, latent_dim, nt=3):
        super().__init__()

        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu

        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.adaLN_mod = AdaLNModulation(-1, -1, nchunks=2, input_dim=latent_dim * nt + latent_dim, output_dim=latent_dim * 2)

        self.poseFinal = nn.Linear(latent_dim, out_feats)  #Bias!

    def forward(self, hidden_states: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # hidden_states: (b, 3, seqlen*5, embed_dim)
        # cond: (b, 3, latent_dim + embed_dim)

        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)

        bs, nt, seqlen, dim = hidden_states.shape
        hidden_states = hidden_states.reshape(bs * nt, seqlen, dim)
        shift, scale = self.adaLN_mod(cond.reshape(bs * nt, -1))
        hidden_states = modulate(self.LayerNorm(hidden_states), shift, scale)
        hidden_states = hidden_states.reshape(bs, nt, seqlen, dim)
        output = self.poseFinal(hidden_states)  # [bs, nt, seqlen, out_feats]
        return output


class MaskTransformer(nn.Module):

    def __init__(
        self,
        embed_dim,
        device,
        num_tokens=8192,
        cond_mode='text',
        latent_dim=256,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        clip_dim=512,
        cond_drop_prob=0.1,
        clip_version=None,
        position_embedding='sine',
    ):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.nt = 3
        self.nbp = 5
        self.embed_dim = embed_dim
        self.device = device
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob
        self.num_tokens = num_tokens

        assert self.latent_dim == self.embed_dim * self.nt, 'latent_dim should be equal to embed_dim*nt'
        '''
        Preparing Networks
        '''
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        # _num_tokens = num_tokens + 2  # two dummy tokens, one for masking, one for padding
        self.mask_id = num_tokens
        self.pad_id = num_tokens + 1
        self.token_emb = nn.Embedding(num_tokens + 2, self.embed_dim)  # one for masking, one for padding
        self.input_process = InputProcess(self.embed_dim, self.embed_dim)
        self.output_process = OutputProcess(out_feats=num_tokens, latent_dim=self.embed_dim)
        self.tags_token = nn.Parameter(torch.zeros(3, self.embed_dim))

        # position_embedding='learned' | 'sine'
        self.position_enc = SpaTempPositionalEncoding(self.latent_dim)
        text_decoder_layer = TextCondTransformerDecoderLayer(
            self.nt,
            self.nbp,
            self.mask_id,
            d_model=self.latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
        )
        mot_decoder_layer = MotCondTransformerDecoderLayer(
            self.nt,
            self.nbp,
            self.mask_id,
            d_model=self.embed_dim,
            nhead=self.embed_dim // 64,
            dim_feedforward=ff_size,
            dropout=dropout,
            batch_first=True,
        )
        self.seqCondTransDecoder = CondTransformerDecoder(text_decoder_layer, mot_decoder_layer, num_layers)

        self.initialize_weights()
        '''
        Preparing frozen weights
        '''
        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)
        self.noise_schedule = cosine_schedule

    def initialize_weights(self):

        def __init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    module.bias.data.zero_()
                if module.weight is not None:
                    module.weight.data.fill_(1.0)

        self.apply(__init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.seqCondTransDecoder.text_layers:
            nn.init.constant_(block.text_adaLN.model[-1].weight, 0)
            nn.init.constant_(block.text_adaLN.model[-1].bias, 0)

        for block in self.seqCondTransDecoder.mot_layers:
            nn.init.constant_(block.mot_adaLN.model[-1].weight, 0)
            nn.init.constant_(block.mot_adaLN.model[-1].bias, 0)

        nn.init.constant_(self.output_process.adaLN_mod.model[-1].weight, 0)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].bias, 0)
        return

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training
        # Added support for cpu
        if str(self.device) != "cpu":
            clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
            # Date 0707: It's necessary, only unecessary when load directly to gpu. Disable if need to run on cpu

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)

        with torch.no_grad():
            word_emb = self.clip_model.token_embedding(text).type(self.clip_model.dtype)
            word_emb = word_emb + self.clip_model.positional_embedding.type(self.clip_model.dtype)
            word_emb = word_emb.permute(1, 0, 2)  # NLD -> LND
            word_emb = self.clip_model.transformer(word_emb)
            word_emb = self.clip_model.ln_final(word_emb).permute(1, 0, 2).float()
            feat_clip_text = self.clip_model.encode_text(text).float()

        return feat_clip_text, word_emb

    def mask_cond(self, cond, force_mask=False):
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            if len(cond.shape) == 2:
                mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            else:
                mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1, 1)
            return cond * (1. - mask)
        else:
            return cond

    def trans_forward(self, ids_lst, cond, cond_word, padding_mask, force_mask=False):
        # m1_ids, m2_ids, d_ids = ids_lst
        cond = self.mask_cond(cond, force_mask=force_mask)
        cond_word = self.mask_cond(cond_word, force_mask=force_mask)

        cond = self.cond_emb(cond)  #(b, latent_dim)
        cond_word = self.cond_emb(cond_word)
        cond_word = self.position_enc(cond_word)  # (b, 77, latent_dim)

        # ======> 1. token embedding <======
        bs, seqlen = ids_lst[0].shape
        bs = bs // self.nbp
        x_ids = torch.cat(ids_lst, dim=1)  # (b*j*3, seqlen)
        x_ids = self.token_emb(x_ids)
        x_ids = self.input_process(x_ids)  # (b*j*3, seqlen, embed_dim)

        x_ids = x_ids.reshape(bs, self.nbp, self.nt, seqlen, -1)  # (b, 5, 3, seqlen, embed_dim)
        x_ids = x_ids.permute(0, 3, 1, 2, 4).reshape(bs, seqlen, self.nbp, -1)
        x_ids = self.position_enc(x_ids)  # (b, seqlen, 5, 3*embed_dim)
        x_ids = x_ids.reshape(bs, seqlen*self.nbp, -1)

        padding_mask = padding_mask[:, :, None].expand(-1, -1, self.nbp).reshape(bs, -1)  # (bs, seq_len*5)
        tag_tokens = self.tags_token[None, :, :].expand(bs, -1, -1)  # (bs, 3, embed_dim)

        # ======> 2. text condition injection <======
        x_ids = x_ids + tag_tokens.reshape(bs, 1, -1)        
        x_ids = self.seqCondTransDecoder.text_forward(x_ids, cond, cond_word, tgt_key_padding_mask=padding_mask)

        # ======> 3. motion and distance fusion <======
        x_ids = x_ids.reshape(bs, seqlen * self.nbp, self.nt, -1)
        x_ids = x_ids.permute(0, 2, 1, 3).reshape(bs * self.nt, seqlen * self.nbp, -1)
        x_ids = self.position_enc(x_ids)  # (b*3, seqlen*5, embed_dim)
        x_ids = x_ids.reshape(bs, self.nt, seqlen * self.nbp, -1)

        cond_tag = cond.unsqueeze(1).expand(-1, self.nt, -1)  # (b, 3, latent_dim)
        cond_tag = torch.cat((cond_tag, tag_tokens), dim=-1)  # (bs, 3, latent_dim + embed_dim)
        x_ids = self.seqCondTransDecoder.fus_forward(x_ids, cond_tag, tgt_key_padding_mask=padding_mask)

        # ======> 4. output process <======
        logits = self.output_process(x_ids, cond_tag)  # (bs, nt, seqlen*nbp, num_tokens)
        logits = logits.reshape(bs, self.nt, seqlen, self.nbp, -1)
        logits = logits.permute(0, 1, 3, 2, 4)  # (b, 3, 5, seqlen, num_tokens)
        logits_lst = [logits[:, i].reshape(bs * self.nbp, seqlen, -1) for i in range(self.nt)]
        return logits_lst

    def _mask_tokens(self, x_ids, mask):
        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)

        # ! num of local motion tokens == num of global motion tokens
        rand_id = torch.randint_like(x_ids, high=self.num_tokens)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
        x_ids = torch.where(mask_mid, self.mask_id, x_ids)
        return x_ids

    def _prepare_tokens(self, ids, bs, device, ntokens, non_pad_mask):
        # Positions that are PADDED are ALL FALSE
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        # ======> 1. Random 50% masking, no prediction
        rand_time = uniform((bs, ), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # ======> 2. prepare input
        # Positions to be MASKED must also be NON-PADDED
        mask &= non_pad_mask

        # non_pre_mask, True: need to predict, False: fill with mask_token, do not predict directly
        labels = torch.where(mask, ids, self.mask_id)

        # The difference is that,
        # in the first line 50% of the tokens are directly masked and not predicted,
        # while in the second line 50% of the tokens are given the ground truth, the remaining 50% are predicted, and masking is applied in a 0.1/0.1/0.8 ratio.
        # motion_ids = labels.clone()
        motion_ids = ids.clone()

        # ======> 3. predict
        # For places where non_pre_mask is True, split into (10%, 10%, 80%)
        motion_ids = self._mask_tokens(motion_ids, mask)
        return motion_ids, labels

    def _prepare_tokens_part(self, ids, non_pad_mask):
        '''
        ids: (bs, 5, seqlen)
        non_pad_mask: (bs, seqlen)
        '''
        bs, nbp, ntokens = ids.shape
        device = ids.device

        # Positions that are PADDED are ALL FALSE
        non_pad_mask = non_pad_mask[:, None, :].repeat(1, nbp, 1)  #(b, 5, t)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        # ======> 1. Random 50% mask, no prediction
        rand_time = uniform((bs, ), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        # Positions to be MASKED are ALL TRUE
        mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # ======> 2. prepare input
        # Positions to be MASKED must also be NON-PADDED
        mask = mask[:, None, :].repeat(1, nbp, 1)  # (b, 5, t)
        mask &= non_pad_mask
        labels = torch.where(mask, ids, self.mask_id)
        
        # The difference is that,
        # in the first line 50% of the tokens are directly masked and not predicted,
        # while in the second line 50% of the tokens are given the ground truth, the remaining 50% are predicted, and masking is applied in a 0.1/0.1/0.8 ratio.
        # motion_ids = labels.clone()
        motion_ids = ids.clone()

        # ======> 3. predict
        # Where non_pre_mask is True, split into (10%, 10%, 80%)
        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask[:, 0, :], 0.1)
        mask_rid = mask_rid[:, None, :].repeat(1, nbp, 1)  # (b, 5, t)

        # ! num of local motion tokens == num of global motion tokens
        rand_id = torch.randint_like(motion_ids, high=self.num_tokens)
        motion_ids = torch.where(mask_rid, rand_id, motion_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = mask[:, 0, :] & ~mask_rid[:, 0, :]
        mask_mid = get_mask_subset_prob(mask_mid, 0.88)
        mask_mid = mask_mid[:, None, :].repeat(1, nbp, 1)  # (b, 5, t)
        motion_ids = torch.where(mask_mid, self.mask_id, motion_ids)
        return motion_ids, labels

    def forward(self, m1_ids, m2_ids, d_ids, y, m_lens):
        '''
        :param ids_mot: (b, n)
        :param ids_glbm: (b, res_length)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''
        m1_ids = m1_ids.reshape(m1_ids.shape[0], self.nbp, -1)
        m2_ids = m2_ids.reshape(m2_ids.shape[0], self.nbp, -1)
        d_ids = d_ids.reshape(d_ids.shape[0], self.nbp, -1)

        bs, _, ntokens = m1_ids.shape
        device = m1_ids.device

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                # cond_word_vector: (bs, 77, 768)
                cond_vector, cond_word_vector = self.encode_text(y)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            cond_word_vector = torch.zeros(bs, 77, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        non_pad_mask = lengths_to_mask(m_lens, ntokens)  #(b, t)
        ids_lst, labels_lst = [], []
        for ids in [m1_ids, m2_ids, d_ids]:
            ids, labels = self._prepare_tokens_part(ids, non_pad_mask)
            ids_lst.append(ids.reshape(-1, ntokens))
            labels_lst.append(labels.reshape(-1, ntokens))
        logits_lst = self.trans_forward(ids_lst, cond_vector, cond_word_vector, ~non_pad_mask, force_mask)

        ce_loss, acc = 0, 0
        for labels, logits in zip(labels_lst, logits_lst):
            logits = logits.permute(0, 2, 1)  # (b, n+t, num_token) -> (b, num_token, n+t)
            _ce_loss, _pred_id, _acc = cal_performance(logits, labels, ignore_index=self.mask_id)
            ce_loss = ce_loss + _ce_loss
            acc = acc + _acc
        ce_loss, acc = ce_loss / len(labels_lst), acc / len(labels_lst)

        for i in range(len(logits_lst)):
            logits_lst[i] = logits_lst[i].reshape(-1, self.nbp, logits_lst[i].shape[1], logits_lst[i].shape[2])
        return ce_loss, acc, logits_lst

    def forward_with_cond_scale(self, ids_lst, cond_vector, cond_word_vector, padding_mask, cond_scale=3, force_mask=False):
        '''
        motion_ids: (b, n+t), n is the length of global token, t is the length of local token
        
        logits: based on text condition
        aux_logits: masked all text condition
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale        
        '''

        if force_mask:
            return self.trans_forward(ids_lst, cond_vector, cond_word_vector, padding_mask, force_mask=True)

        logits_lst = self.trans_forward(ids_lst, cond_vector, cond_word_vector, padding_mask)
        if cond_scale == 1:
            return logits_lst

        aux_logits_lst = self.trans_forward(ids_lst, cond_vector, cond_word_vector, padding_mask, force_mask=True)

        scaled_logits_lst = []
        for logits, aux_logits in zip(logits_lst, aux_logits_lst):
            scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
            scaled_logits_lst.append(scaled_logits)

        return scaled_logits_lst

    @torch.no_grad()
    @eval_decorator
    def generate(
            self,
            conds,
            m_lens,
            timesteps: int,  # 18
            cond_scale: int,  # kit: 2, t2m: 4
            max_len=None,
            temperature=1,
            topk_filter_thres=0.9,
            gsample=False,
            force_mask=False):
        device = next(self.parameters()).device

        # ! When generating, the length of the motion must be provided; 
        # ! therefore, during the generation stage a length-prediction model is needed to predict the motion length.
        seq_len = max(m_lens) if max_len is None else max_len
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, cond_word_vector = self.encode_text(conds)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
            cond_word_vector = torch.zeros(batch_size, 77, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        part_padding_mask = padding_mask[:, None, :].expand(-1, self.nbp, -1)  #(b, 5, t)
        part_padding_mask = part_padding_mask.reshape(-1, part_padding_mask.shape[2])  #(b*5, t)

        # Start from all tokens being masked
        ids_lst, scores_lst = [], []
        for _ in range(3):
            ids = torch.where(part_padding_mask, self.pad_id, self.mask_id).to(device)
            scores = torch.where(part_padding_mask, 1e5, 0.).to(device)
            ids_lst.append(ids)  # (b*5, t)
            scores_lst.append(scores)  # (b*5, t)
        starting_temperature = temperature

        part_m_lens = m_lens[:, None].expand(-1, self.nbp).reshape(-1)  # (b*5, )

        # Generate in multiple passes; in each pass, mask out tokens with lower scores according to a cosine-shaped probability curve.
        for timestep in torch.linspace(0, 1, timesteps, device=device):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep).to(device)  # Tensor
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * part_m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            is_mask_lst = []
            for k, scores in enumerate(scores_lst):
                sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
                ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
                is_mask = (ranks < num_token_masked.unsqueeze(-1))
                ids_lst[k] = torch.where(is_mask, self.mask_id, ids_lst[k])
                is_mask_lst.append(is_mask)
            '''
            Preparing input
            '''
            logits_lst = self.forward_with_cond_scale(ids_lst, cond_vector=cond_vector, cond_word_vector=cond_word_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)

            for k, logits in enumerate(logits_lst):
                # print(logits.shape, self.num_tokens)
                # clean low prob token
                filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
                '''
                Update ids
                '''
                temperature = starting_temperature
                # temperature is annealed, gradually reducing temperature as well as randomness
                if gsample:  # use gumbel_softmax sampling
                    pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
                else:  # use multinomial sampling
                    probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                    pred_ids = Categorical(probs).sample()  # (b, seqlen)

                ids_lst[k] = torch.where(is_mask_lst[k], pred_ids, ids_lst[k])
                '''
                Updating scores
                '''
                probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
                scores_lst[k] = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
                scores_lst[k] = scores_lst[k].squeeze(-1)  # (b, seqlen)
                # We do not want to re-mask the previously kept tokens, or pad tokens
                scores_lst[k] = scores_lst[k].masked_fill(~is_mask_lst[k], 1e5)

        for i in range(len(ids_lst)):
            ids_lst[i] = torch.where(part_padding_mask, self.pad_id, ids_lst[i])
            ids_lst[i] = ids_lst[i].reshape(-1, self.nbp, ids_lst[i].shape[1])
        return ids_lst
    
    @torch.no_grad()
    @eval_decorator
    def edit(
            self,
            conds,
            motion1_ids,
            motion2_ids, 
            dist_ids,
            edit_mask,
            m_lens,
            timesteps: int,  # 18
            cond_scale: int,  # kit: 2, t2m: 4
            max_len=None,
            temperature=1,
            topk_filter_thres=0.9,
            gsample=False,
            force_mask=False):
        device = next(self.parameters()).device

        seq_len = max(m_lens) if max_len is None else max_len
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, cond_word_vector = self.encode_text(conds)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
            cond_word_vector = torch.zeros(batch_size, 77, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        part_padding_mask = padding_mask[:, None, :].expand(-1, self.nbp, -1)  #(b, 5, t)
        part_padding_mask = part_padding_mask.reshape(-1, part_padding_mask.shape[2])  #(b*5, t)
        part_edit_mask = edit_mask.reshape(-1, part_padding_mask.shape[1])
        
        motion1_ids = motion1_ids.reshape(-1, part_padding_mask.shape[1])
        motion1_ids = torch.where(part_edit_mask, self.mask_id, motion1_ids)
        motion1_ids = torch.where(part_padding_mask, self.pad_id, motion1_ids)
        motion1_scores = torch.where(part_edit_mask, 0., 1e5)
        motion1_scores = torch.where(part_padding_mask, 1e5, motion1_scores)

        motion2_ids = motion2_ids.reshape(-1, part_padding_mask.shape[1])
        motion2_ids = torch.where(part_edit_mask, self.mask_id, motion2_ids)
        motion2_ids = torch.where(part_padding_mask, self.pad_id, motion2_ids)
        motion2_scores = torch.where(part_edit_mask, 0., 1e5)
        motion2_scores = torch.where(part_padding_mask, 1e5, motion2_scores)
        
        dist_ids = dist_ids.reshape(-1, part_padding_mask.shape[1])
        dist_ids = torch.where(part_edit_mask, self.mask_id, dist_ids)
        dist_ids = torch.where(part_padding_mask, self.pad_id, dist_ids)
        dist_scores = torch.where(part_edit_mask, 0., 1e5)
        dist_scores = torch.where(part_padding_mask, 1e5, dist_scores)

        ids_lst = [motion1_ids, motion2_ids, dist_ids]
        scores_lst = [motion1_scores, motion2_scores, dist_scores]
        starting_temperature = temperature
        part_m_lens = edit_mask.sum(dim=2).reshape(-1)  # (b*5, )

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep).to(device)  # Tensor
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * part_m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            is_mask_lst = []
            for k, scores in enumerate(scores_lst):
                sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
                ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
                is_mask = (ranks < num_token_masked.unsqueeze(-1))
                ids_lst[k] = torch.where(is_mask, self.mask_id, ids_lst[k])
                is_mask_lst.append(is_mask)
            '''
            Preparing input
            '''
            logits_lst = self.forward_with_cond_scale(ids_lst, cond_vector=cond_vector, cond_word_vector=cond_word_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)

            for k, logits in enumerate(logits_lst):
                # print(logits.shape, self.num_tokens)
                # clean low prob token
                filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
                '''
                Update ids
                '''
                temperature = starting_temperature

                # temperature is annealed, gradually reducing temperature as well as randomness
                if gsample:  # use gumbel_softmax sampling
                    pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
                else:  # use multinomial sampling
                    probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                    pred_ids = Categorical(probs).sample()  # (b, seqlen)

                ids_lst[k] = torch.where(is_mask_lst[k], pred_ids, ids_lst[k])
                '''
                Updating scores
                '''
                probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
                scores_lst[k] = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
                scores_lst[k] = scores_lst[k].squeeze(-1)  # (b, seqlen)
                # We do not want to re-mask the previously kept tokens, or pad tokens
                scores_lst[k] = scores_lst[k].masked_fill(~is_mask_lst[k], 1e5)

        for i in range(len(ids_lst)):
            ids_lst[i] = torch.where(part_padding_mask, self.pad_id, ids_lst[i])
            ids_lst[i] = ids_lst[i].reshape(-1, self.nbp, ids_lst[i].shape[1])
        return ids_lst
    
    @torch.no_grad()
    @eval_decorator
    def react(
            self,
            conds,
            motion1_ids,
            m_lens,
            timesteps: int,  # 18
            cond_scale: int,  # kit: 2, t2m: 4
            max_len=None,
            temperature=1,
            topk_filter_thres=0.9,
            gsample=False,
            force_mask=False):
        device = next(self.parameters()).device

        seq_len = max(m_lens) if max_len is None else max_len
        batch_size = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, cond_word_vector = self.encode_text(conds)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
            cond_word_vector = torch.zeros(batch_size, 77, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        
        part_padding_mask = padding_mask[:, None, :].expand(-1, self.nbp, -1)  #(b, 5, t)
        part_padding_mask = part_padding_mask.reshape(-1, part_padding_mask.shape[2])  #(b*5, t)

        # Start from all tokens being masked
        ids_lst, scores_lst = [], []
        for i in range(3):
            if i == 0:
                motion1_ids = motion1_ids.reshape(-1, part_padding_mask.shape[1])
                ids = torch.where(part_padding_mask, self.pad_id, motion1_ids).to(device)
                scores = torch.where(part_padding_mask, 1e5, 1e5).to(device)
            else:
                ids = torch.where(part_padding_mask, self.pad_id, self.mask_id).to(device)
                scores = torch.where(part_padding_mask, 1e5, 0.).to(device)
            ids_lst.append(ids)  # (b*5, t)
            scores_lst.append(scores)  # (b*5, t)
        starting_temperature = temperature

        part_m_lens = m_lens[:, None].expand(-1, self.nbp).reshape(-1)  # (b*5, )

        for timestep in torch.linspace(0, 1, timesteps, device=device):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep).to(device)  # Tensor
            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * part_m_lens).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            is_mask_lst = []
            for k, scores in enumerate(scores_lst):
                if k == 0: 
                    is_mask_lst.append([])
                    continue
                sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
                ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
                is_mask = (ranks < num_token_masked.unsqueeze(-1))
                ids_lst[k] = torch.where(is_mask, self.mask_id, ids_lst[k])
                is_mask_lst.append(is_mask)
            '''
            Preparing input
            '''
            logits_lst = self.forward_with_cond_scale(ids_lst, cond_vector=cond_vector, cond_word_vector=cond_word_vector, padding_mask=padding_mask, cond_scale=cond_scale, force_mask=force_mask)

            for k, logits in enumerate(logits_lst):
                if k == 0: 
                    continue
                filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
                '''
                Update ids
                '''
                temperature = starting_temperature
                # temperature is annealed, gradually reducing temperature as well as randomness
                if gsample:  # use gumbel_softmax sampling
                    pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
                else:  # use multinomial sampling
                    probs = F.softmax(filtered_logits / temperature, dim=-1)  # (b, seqlen, ntoken)
                    pred_ids = Categorical(probs).sample()  # (b, seqlen)

                ids_lst[k] = torch.where(is_mask_lst[k], pred_ids, ids_lst[k])
                '''
                Updating scores
                '''
                probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
                scores_lst[k] = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
                scores_lst[k] = scores_lst[k].squeeze(-1)  # (b, seqlen)
                # We do not want to re-mask the previously kept tokens, or pad tokens
                scores_lst[k] = scores_lst[k].masked_fill(~is_mask_lst[k], 1e5)

        for i in range(len(ids_lst)):
            ids_lst[i] = torch.where(part_padding_mask, self.pad_id, ids_lst[i])
            ids_lst[i] = ids_lst[i].reshape(-1, self.nbp, ids_lst[i].shape[1])
        return ids_lst
