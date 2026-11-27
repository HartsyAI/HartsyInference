using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol.Budget;

/// <summary>One parent latent becomes <c>n</c> child latents (upstream <c>RefineStage</c>): a cube's eight cells, or a cell's two points. The child prediction starts as the nearest repeat of the parent and a small
/// adaLN-zero transformer over the parent's own children (width 128, 4 heads, 2 blocks, one sequence of <c>n</c> per parent) adds a correction through a learned output projection. Its inputs are the repeated parent
/// latent, the noisy state of each child through a skip projection, the child's position relative to the parent cube, a learned child slot and, in the second stage of the s = 4 cascade, the hidden state of the first.
/// The matrix products run on the backend; the normalisations, the gates and the attention over a handful of children run on the host.</summary>
internal sealed class BudgetRefineStage
{
    private const float NormEps = 1e-6f;

    private readonly BudgetAdapterWeights _w;
    private readonly string _prefix;
    private readonly int _n;

    internal BudgetRefineStage(BudgetAdapterWeights weights, int children)
    {
        _w = weights;
        _n = children;
        _prefix = $"bp_refiners.{(children == 2 ? 2 : 4)}";
    }

    /// <summary>The tensors a forward pass hands to the backend.</summary>
    internal IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (string name in new[] { "embed", "embed_zt", "pos_proj", "out" })
        {
            yield return _w[$"{_prefix}.{name}.weight"];
            yield return _w[$"{_prefix}.{name}.bias"];
        }

        for (int b = 0; b < BudgetAdapterWeights.RefineDepth; b++)
            foreach (string name in new[] { "qkv", "proj", "mlp.0", "mlp.2", "ada" })
            {
                yield return _w[$"{_prefix}.blocks.{b}.{name}.weight"];
                yield return _w[$"{_prefix}.blocks.{b}.{name}.bias"];
            }
    }

    /// <summary>The children of <paramref name="rows"/> parents.</summary>
    /// <param name="parent">The parents' latent prediction, <c>rows × 16</c>.</param>
    /// <param name="cond">The conditioning hidden state of each condition group, <c>(rows / condRepeat) × C</c>; each group serves <paramref name="condRepeat"/> adjacent rows.</param>
    /// <param name="relative">The children's offsets inside the parent, <c>n × 3</c>.</param>
    /// <param name="noisy">The children's noisy latent, <c>rows × n × 16</c>.</param>
    /// <param name="carry">The hidden state carried from a first stage, <c>rows × 128</c>, or empty.</param>
    /// <returns>The child latents <c>rows × n × 16</c> and the hidden states <c>rows × n × 128</c>.</returns>
    internal (float[] Latent, float[] Hidden) Forward(IBackend backend, ReadOnlySpan<float> parent, int rows, ReadOnlySpan<float> cond, int condDim, int condRepeat,
        ReadOnlySpan<float> relative, ReadOnlySpan<float> noisy, ReadOnlySpan<float> carry)
    {
        int n = _n, w = BudgetAdapterWeights.RefineWidth, lat = 16, tokens = rows * n;
        float[] repeated = new float[tokens * lat];
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
                parent.Slice(r * lat, lat).CopyTo(repeated.AsSpan((r * n + j) * lat, lat));

        // h = embed(rep) + slots + pos_proj(fourier3d(rel)) [+ embed_zt(z_t)] [+ carry]
        float[] h;
        using (Tensor repT = BudgetHost.Make(repeated, 1, tokens, lat))
        using (Tensor embedded = BudgetHost.Linear(backend, repT, _w[$"{_prefix}.embed.weight"], _w[$"{_prefix}.embed.bias"]))
            h = embedded.AsReadOnlySpan<float>().ToArray();
        float[] positional = PositionTable(backend, relative);
        ReadOnlySpan<float> slots = _w[$"{_prefix}.slots"].AsReadOnlySpan<float>();
        for (int i = 0; i < tokens; i++)
        {
            int j = i % n;
            for (int c = 0; c < w; c++)
                h[i * w + c] += slots[j * w + c] + positional[j * w + c];
        }

        using (Tensor ztT = BudgetHost.Make(noisy, 1, tokens, lat))
        using (Tensor skip = BudgetHost.Linear(backend, ztT, _w[$"{_prefix}.embed_zt.weight"], _w[$"{_prefix}.embed_zt.bias"]))
        {
            ReadOnlySpan<float> z = skip.AsReadOnlySpan<float>();
            for (int i = 0; i < h.Length; i++)
                h[i] += z[i];
        }

        if (!carry.IsEmpty)
            for (int i = 0; i < tokens; i++)
                for (int c = 0; c < w; c++)
                    h[i * w + c] += carry[(i / n) * w + c];

        int groups = rows / condRepeat;
        for (int b = 0; b < BudgetAdapterWeights.RefineDepth; b++)
            Block(backend, h, rows, groups, condRepeat, cond, condDim, b);

        // out = rep + out(out_norm(h)), the norm an RMS norm with a learned gain.
        ReadOnlySpan<float> gain = _w[$"{_prefix}.out_norm.weight"].AsReadOnlySpan<float>();
        float[] normed = new float[h.Length];
        for (int i = 0; i < tokens; i++)
        {
            float sq = 0f;
            for (int c = 0; c < w; c++)
                sq += h[i * w + c] * h[i * w + c];
            float inv = 1f / MathF.Sqrt(sq / w + NormEps);
            for (int c = 0; c < w; c++)
                normed[i * w + c] = h[i * w + c] * inv * gain[c];
        }

        float[] latent;
        using (Tensor normT = BudgetHost.Make(normed, 1, tokens, w))
        using (Tensor projected = BudgetHost.Linear(backend, normT, _w[$"{_prefix}.out.weight"], _w[$"{_prefix}.out.bias"]))
        {
            ReadOnlySpan<float> o = projected.AsReadOnlySpan<float>();
            latent = new float[tokens * lat];
            for (int i = 0; i < latent.Length; i++)
                latent[i] = repeated[i] + o[i];
        }

        return (latent, h);
    }

    // pos_proj(fourier3d(rel_pos)): the sines then cosines of the offsets at 4 octaves (24 features), projected to the width.
    private float[] PositionTable(IBackend backend, ReadOnlySpan<float> relative)
    {
        const int octaves = 4;
        float[] features = new float[_n * 3 * 2 * octaves];
        float pi = (float)Math.PI;
        for (int j = 0; j < _n; j++)
            for (int axis = 0; axis < 3; axis++)
                for (int k = 0; k < octaves; k++)
                {
                    float angle = relative[j * 3 + axis] * (MathF.Pow(2f, k) * pi);
                    features[j * 24 + axis * 8 + k] = MathF.Sin(angle);
                    features[j * 24 + axis * 8 + octaves + k] = MathF.Cos(angle);
                }

        using Tensor input = BudgetHost.Make(features, 1, _n, 24);
        using Tensor projected = BudgetHost.Linear(backend, input, _w[$"{_prefix}.pos_proj.weight"], _w[$"{_prefix}.pos_proj.bias"]);
        return projected.AsReadOnlySpan<float>().ToArray();
    }

    // One adaLN-zero block over every parent's children, in place on h (rows × n × width).
    private void Block(IBackend backend, float[] h, int rows, int groups, int condRepeat, ReadOnlySpan<float> cond, int condDim, int index)
    {
        int n = _n, w = BudgetAdapterWeights.RefineWidth, heads = BudgetAdapterWeights.RefineHeads, hd = w / heads, tokens = rows * n;
        string p = $"{_prefix}.blocks.{index}";

        // shift1, scale1, gate1, shift2, scale2, gate2 = ada(silu(cond)), one set per condition group.
        float[] ada;
        using (Tensor condT = BudgetHost.Make(cond, 1, groups, condDim))
        using (Tensor act = BudgetHost.Activate(backend, condT, static (b, o, i) => b.Silu(o, i)))
        using (Tensor modulation = BudgetHost.Linear(backend, act, _w[$"{p}.ada.weight"], _w[$"{p}.ada.bias"]))
            ada = modulation.AsReadOnlySpan<float>().ToArray();

        float[] Modulated(int shift, int scale)
        {
            float[] m = new float[tokens * w];
            Parallel.For(0, tokens, i =>
            {
                int g = i / n / condRepeat;
                float mean = 0f;
                for (int c = 0; c < w; c++)
                    mean += h[i * w + c];
                mean /= w;
                float variance = 0f;
                for (int c = 0; c < w; c++)
                {
                    float d = h[i * w + c] - mean;
                    variance += d * d;
                }

                float inv = 1f / MathF.Sqrt(variance / w + NormEps);
                for (int c = 0; c < w; c++)
                    m[i * w + c] = (h[i * w + c] - mean) * inv * (1f + ada[g * 6 * w + scale * w + c]) + ada[g * 6 * w + shift * w + c];
            });

            return m;
        }

        // attention branch
        float[] attended;
        using (Tensor norm1 = BudgetHost.Make(Modulated(0, 1), 1, tokens, w))
        using (Tensor qkvT = BudgetHost.Linear(backend, norm1, _w[$"{p}.qkv.weight"], _w[$"{p}.qkv.bias"]))
            attended = Attend(qkvT.AsReadOnlySpan<float>().ToArray(), rows, n, heads, hd);
        using (Tensor attT = BudgetHost.Make(attended, 1, tokens, w))
        using (Tensor projT = BudgetHost.Linear(backend, attT, _w[$"{p}.proj.weight"], _w[$"{p}.proj.bias"]))
        {
            ReadOnlySpan<float> proj = projT.AsReadOnlySpan<float>();
            for (int i = 0; i < tokens; i++)
            {
                int g = i / n / condRepeat;
                for (int c = 0; c < w; c++)
                    h[i * w + c] += ada[g * 6 * w + 2 * w + c] * proj[i * w + c];
            }
        }

        // MLP branch: Linear, tanh GELU, Linear
        using (Tensor norm2 = BudgetHost.Make(Modulated(3, 4), 1, tokens, w))
        using (Tensor hidden = BudgetHost.Linear(backend, norm2, _w[$"{p}.mlp.0.weight"], _w[$"{p}.mlp.0.bias"]))
        using (Tensor act = BudgetHost.Activate(backend, hidden, static (b, o, i) => b.Gelu(o, i)))
        using (Tensor down = BudgetHost.Linear(backend, act, _w[$"{p}.mlp.2.weight"], _w[$"{p}.mlp.2.bias"]))
        {
            ReadOnlySpan<float> mlp = down.AsReadOnlySpan<float>();
            for (int i = 0; i < tokens; i++)
            {
                int g = i / n / condRepeat;
                for (int c = 0; c < w; c++)
                    h[i * w + c] += ada[g * 6 * w + 5 * w + c] * mlp[i * w + c];
            }
        }
    }

    // Softmax attention over each parent's n children, per head: qkv is [tokens, 3 · heads · hd] split as (which, head, hd); the heads come back concatenated.
    private static float[] Attend(float[] qkv, int rows, int n, int heads, int hd)
    {
        int w = heads * hd, stride = 3 * w;
        float scale = 1f / MathF.Sqrt(hd);
        float[] output = new float[rows * n * w];
        Parallel.For(0, rows, r =>
        {
            float[] scores = new float[n];
            for (int head = 0; head < heads; head++)
                for (int i = 0; i < n; i++)
                {
                    int qBase = (r * n + i) * stride + head * hd;
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < n; j++)
                    {
                        int kBase = (r * n + j) * stride + w + head * hd;
                        float dot = 0f;
                        for (int d = 0; d < hd; d++)
                            dot += qkv[qBase + d] * qkv[kBase + d];
                        scores[j] = dot * scale;
                        max = MathF.Max(max, scores[j]);
                    }

                    float sum = 0f;
                    for (int j = 0; j < n; j++)
                    {
                        scores[j] = MathF.Exp(scores[j] - max);
                        sum += scores[j];
                    }

                    for (int d = 0; d < hd; d++)
                    {
                        float acc = 0f;
                        for (int j = 0; j < n; j++)
                            acc += scores[j] / sum * qkv[(r * n + j) * stride + 2 * w + head * hd + d];
                        output[(r * n + i) * w + head * hd + d] = acc;
                    }
                }
        });

        return output;
    }
}
