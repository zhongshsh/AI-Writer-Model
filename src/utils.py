import random
import numpy as np
import oneflow as flow
import oneflow.nn as nn
from oneflow.nn import functional as F


def top_k_logits(logits, k):
    v, ix = flow.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


def top_p_probs(probs, p):
    out = probs.clone()

    sorted_probs, sorted_indices = flow.sort(out, descending=True)
    cumulative_probs = flow.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    out[indices_to_remove] = 0

    return out


# top-p + top-k + pow&ratio sampling
def sample_logits(
    logits,
    pos,
    temperature=1.0,
    top_k=None,
    top_p=None,
    min_p_pow=None,
    min_p_ratio=None,
):
    logits = logits[:, pos, :] / temperature
    probs = F.softmax(logits, dim=-1)

    if min_p_ratio is not None:
        limit = flow.pow(flow.max(probs), min_p_pow) * min_p_ratio
        logits[probs < limit] = -float("Inf")

    if top_k is not None:
        logits = top_k_logits(logits, top_k)

    probs = F.softmax(logits, dim=-1)

    if top_p is not None:
        probs[0] = top_p_probs(probs[0], top_p)

    probs = probs[0]
    p = probs.numpy().astype(np.float64)
    p /= p.sum()
    ix = np.random.choice(np.arange(probs.shape[0]), size=1, p=p)

    # sample = np.random.multinomial(n=1, pvals=p)
    # ix = np.argmax(sample)

    return int(ix)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    flow.manual_seed(seed)
    # flow.cuda.manual_seed_all(seed)
