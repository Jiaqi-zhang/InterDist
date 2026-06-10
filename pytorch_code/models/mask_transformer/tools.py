import math
import torch
import torch.nn.functional as F
from einops import rearrange, einsum


# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask  #(b, len)


# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)


# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def eval_decorator(fn):

    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def l2norm(t):
    return F.normalize(t, dim=-1)


# tensor helpers


# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    subset_mask = torch.bernoulli(mask, p=prob) & mask
    return subset_mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids == special_id)
    return mask


# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


# classifier free guidance functions


def uniform(shape, device=None):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob


# sampling helpers


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres=0.9, dim=1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim=dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    return probs

# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)


def scale_cosine_schedule(t, scale):
    return torch.clip(scale * torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)


# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs, ), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low


def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        # cross_entropy + dice
        loss_ce = F.cross_entropy(pred, labels, ignore_index=ignore_index)
        loss_dice = cal_dice_loss(pred, labels, ignore_index=ignore_index)
        loss = (loss_ce + loss_dice) * 0.5

    return loss


def cal_dice_loss(pred, target, ignore_index=None, smooth=1e-5):
    """  
    Efficient Dice Loss calculation for multi-class segmentation.  
    
    Args:  
        pred (torch.Tensor): Model output [batch_size, num_classes, num_features]  
        target (torch.Tensor): Ground truth labels [batch_size, num_features]  
        num_classes (int): Number of classes  
        ignore_index (int, optional): Index to ignore in loss calculation  
        smooth (float, optional): Smoothing factor to prevent division by zero  
    
    Returns:  
        torch.Tensor: Mean Dice Loss  
    """
    num_classes = pred.shape[1]
    # Apply softmax to predictions (already in the correct shape)
    pred = F.softmax(pred, dim=1)

    # Create mask for ignored indices if specified
    if ignore_index is not None:
        mask = target != ignore_index
        target = target.masked_fill(~mask, 0)

    # Create one-hot encoded target tensor
    target_one_hot = F.one_hot(target, num_classes=num_classes).float().permute(0, 2, 1)

    # Compute intersection and union
    intersection = torch.sum(pred * target_one_hot, dim=-1)
    pred_sum = torch.sum(pred, dim=-1)
    target_sum = torch.sum(target_one_hot, dim=-1)

    # Apply mask if ignore_index is specified
    if ignore_index is not None:
        intersection = intersection[mask.any(dim=1)]
        pred_sum = pred_sum[mask.any(dim=1)]
        target_sum = target_sum[mask.any(dim=1)]

    # Calculate Dice coefficient
    dice_coef = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)

    # Convert to Dice Loss and return mean
    return torch.mean(1 - dice_coef)


def cal_performance_two(
    logits_first,
    labels_first,
    logits_second,
    labels_second,
    mask_id,
    split_gap,
    smoothing=0.,
    tk=1,
):
    ignore_index_first = mask_id // split_gap
    ignore_index_second = mask_id % split_gap
    new_ignore_index_second = split_gap

    # Calculate loss for the first set of logits
    loss_first = cal_loss(logits_first, labels_first, ignore_index=ignore_index_first, smoothing=smoothing)

    # Calculate loss for the second set of logits using the combined mask
    mask_ignore_second = (labels_first == ignore_index_first) & (labels_second == ignore_index_second)

    loss_second = cal_loss(
        logits_second,
        labels_second.masked_fill(mask_ignore_second, new_ignore_index_second),
        ignore_index=new_ignore_index_second,
        smoothing=smoothing,
    )
    loss = loss_first + loss_second

    # Get predictions
    pred_id_first_k = torch.topk(logits_first, k=tk, dim=1).indices
    pred_id_first = pred_id_first_k[:, 0]  # (bs, 55)
    pred_id_second_k = torch.topk(logits_second, k=tk, dim=1).indices
    pred_id_second = pred_id_second_k[:, 0]  # (bs, 55)
    pred_id = pred_id_first * split_gap + pred_id_second

    labels = labels_first * split_gap + labels_second
    mask = labels.ne(mask_id)
    n_correct = (pred_id == labels).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc
