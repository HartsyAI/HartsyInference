"""Independent float64 score/merge and block-tail references from exact stored operands."""
import math


def dense(q, k, v, scale, lengths=None, causal=False, query_offset=0, bias=None):
    import torch
    q, k, v = q.cpu().double(), k.cpu().double(), v.cpu().double()
    group = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
    scores = q @ k.transpose(-1, -2) * scale
    visible = torch.ones(scores.shape, dtype=torch.bool)
    if lengths is not None:
        visible &= torch.arange(k.shape[2])[None, None, None, :] < torch.tensor(lengths)[:, None, None, None]
    if causal:
        origin = torch.as_tensor(query_offset).reshape(-1, 1, 1, 1)
        visible &= torch.arange(k.shape[2])[None, None, None, :] <= origin + torch.arange(q.shape[2])[None, None, :, None]
    if bias is not None:
        bias = bias.cpu()
        if bias.dtype == torch.bool:
            visible &= bias
        else:
            scores += bias.double()
            visible &= ~torch.isneginf(scores)
    scores = scores.masked_fill(~visible, -math.inf)
    lse = torch.logsumexp(scores, -1)
    p = torch.exp(scores - lse.masked_fill(torch.isneginf(lse), 0)[..., None])
    return p @ v, lse


def merge(prefix_output, prefix_lse, q, k, v, scale, mask):
    import torch
    q, k, v = q.cpu().double(), k.cpu().double(), v.cpu().double()
    group = q.shape[1] // k.shape[1]
    scores = q @ k.repeat_interleave(group, 1).transpose(-1, -2) * scale
    allowed = torch.tensor([[bool(bits & (1 << key)) for key in range(k.shape[2])] for bits in mask])
    scores.masked_fill_(~allowed, -math.inf)
    logits = torch.cat([prefix_lse.cpu().double()[..., None], scores], -1)
    maximum = logits.amax(-1, keepdim=True)
    weights = torch.exp(logits - maximum.masked_fill(torch.isneginf(maximum), 0))
    numerator = weights[..., :1] * prefix_output.cpu().double() + weights[..., 1:] @ v.repeat_interleave(group, 1)
    denominator = weights.sum(-1, keepdim=True)
    return numerator / denominator.clamp_min(1e-300)


def sol(q, k, v, scale, spec, key_bias=None):
    import torch
    q, k, v = q.cpu().double(), k.cpu().double(), v.cpu().double()
    b, h, t, d = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError('General Sol reference requires equal MHA geometry')
    blocks = (t + 63) // 64
    lengths = [min(64, t - i*64) for i in range(blocks)]
    if 'block_len' in spec:
        lengths = [max(1, min(n, supplied)) for n, supplied in zip(lengths, spec['block_len'], strict=True)]
    means_q = torch.stack([q[:, :, i*64:i*64+n].mean(2) for i, n in enumerate(lengths)], 2)
    means_k = torch.stack([k[:, :, i*64:i*64+n].mean(2) for i, n in enumerate(lengths)], 2)
    sums_v = torch.stack([v[:, :, i*64:i*64+n].sum(2) for i, n in enumerate(lengths)], 2)
    center = means_k.mean(2, keepdim=True)
    centered = means_k - center
    log2scale = scale * math.log2(math.e)
    proxy = means_q @ centered.transpose(-1, -2) * log2scale
    variance = centered.square().mean(2)
    thresholds = spec.get('tau', 1) * torch.sqrt((means_q.square() * variance[:, :, None]).sum(-1) * log2scale**2 + 1e-6)
    output = torch.zeros_like(q)
    bias = None
    if key_bias is not None:
        raw_bias = key_bias.cpu()
        if raw_bias.dtype == torch.bool:
            raw_bias = torch.where(raw_bias, 0., -math.inf)
        bias = raw_bias.double().reshape(-1, t).expand(b, t)
    sinks = spec.get('sink_blocks', [0, 0])
    sink_queries = spec.get('sink_q', [0, 0])
    for batch in range(b):
        for head in range(h):
            for block in range(blocks):
                ratio = spec.get('topk_ratio', 0)
                if ratio > 0:
                    candidates = [j for j in range(blocks) if not sinks[0] <= j < sinks[1]]
                    count = max(0, min(len(candidates)-1, max(1, round(len(candidates) * ratio))))
                    cutoff = sorted((float(proxy[batch, head, block, j]) for j in candidates), reverse=True)[count-1] if count else math.inf
                    selected = {j for j in candidates if float(proxy[batch, head, block, j]) >= cutoff}
                else:
                    selected = {j for j in range(blocks) if proxy[batch, head, block, j] > thresholds[batch, head, block]}
                selected |= {j for j in range(blocks) if abs(block-j) <= 1 or sinks[0] <= j < sinks[1]}
                if sink_queries[0] <= block < sink_queries[1]: selected = set(range(blocks))
                for row in range(block*64, block*64 + lengths[block]):
                    logits, values, multiplicities = [], [], []
                    for key_block in range(blocks):
                        if key_block in selected:
                            keys = k[batch, head, key_block*64:key_block*64+lengths[key_block]] - center[batch, head, 0]
                            scores = keys @ q[batch, head, row] * log2scale
                            if bias is not None:
                                scores += bias[batch,
                                    key_block*64:key_block*64+lengths[key_block]] * math.log2(math.e)
                            logits.extend(scores.tolist())
                            values.extend(v[batch, head, key_block*64:key_block*64+lengths[key_block]])
                            multiplicities.extend([1] * lengths[key_block])
                        elif spec.get('tail', True):
                            logits.append(float(proxy[batch, head, block, key_block]))
                            values.append(sums_v[batch, head, key_block])
                            multiplicities.append(lengths[key_block])
                    scores = torch.tensor(logits, dtype=torch.float64)
                    maximum = scores.max()
                    weights = torch.exp2(scores - (maximum if torch.isfinite(maximum) else 0))
                    output[batch, head, row] = (weights[:, None] * torch.stack(values)).sum(0) / (
                        weights * torch.tensor(multiplicities)).sum().clamp_min(1e-300)
    if 'coarse_gate' in spec:
        coarse = torch.softmax(means_q @ means_k.transpose(-1, -2) * scale, -1) @ (
            sums_v / torch.tensor(lengths)[None, None, :, None])
        gate = torch.as_tensor(spec['coarse_gate'], dtype=torch.float64)
        gate = gate.expand(b, t, h, d).permute(0, 2, 1, 3)
        for block in range(blocks):
            start, end = block*64, block*64+lengths[block]
            output[:, :, start:end] += gate[:, :, start:end] * coarse[:, :, block:block+1]
    return output
