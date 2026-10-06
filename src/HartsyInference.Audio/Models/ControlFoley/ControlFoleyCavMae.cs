using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.ControlFoley;

/// <summary>Visual branch of CAV-MAE-ST (<c>CAVMAEST.forward_feat_v</c>): patch embedding, sin-cos positions and modality
/// embedding, the visual-specific blocks, then the shared blocks with their visual norms, and <c>norm_v</c>.
/// The released <c>cav_mae_st.pth</c> stores every key under <c>module.</c>; the official <c>load_state_dict(strict=False)</c>
/// on an unwrapped model therefore matches nothing and runs with random weights. This port loads the real weights
/// (prefix stripped), the behaviour the checkpoint was trained for.</summary>
public sealed unsafe class ControlFoleyCavMae : IDisposable
{
    private const int FrameChunk = 8;

    private readonly ControlFoleyCavMaeConfig _config;
    private readonly List<Tensor> _owned = [];
    private Block[]? _blocks;
    private Tensor? _patchW, _patchB, _pos, _modality, _normW, _normB;

    private sealed record Block(Tensor Norm1W, Tensor Norm1B, Tensor QkvW, Tensor QkvB, Tensor ProjW, Tensor ProjB, Tensor Norm2W,
        Tensor Norm2B, Tensor Fc1W, Tensor Fc1B, Tensor Fc2W, Tensor Fc2B);

    /// <summary>Creates an unloaded encoder; call <see cref="LoadWeights"/> before encoding.</summary>
    public ControlFoleyCavMae(ControlFoleyCavMaeConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.EmbedDim % config.Heads != 0 || config.ModalitySpecificDepth < 0 || config.ModalitySpecificDepth > config.TotalDepth)
        {
            throw new ArgumentException("Invalid CAV-MAE-ST configuration.", nameof(config));
        }

        _config = config;
    }

    /// <summary>Geometry this instance was built for.</summary>
    public ControlFoleyCavMaeConfig Config => _config;

    /// <summary>Binds the visual tensors of a CAV-MAE-ST state dict. A leading <c>module.</c> (DataParallel) is accepted.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        string p = weights.ContainsKey("module.patch_embed_v.proj.weight") ? "module." : "";
        ControlFoleyCavMaeConfig c = _config;
        int d = c.EmbedDim, hidden = d * c.MlpRatio;
        Tensor conv = ControlFoleyVitOps.Load(weights, p + "patch_embed_v.proj.weight", _owned, d, 3, c.PatchSize, c.PatchSize);
        _patchW = conv.Reshape(new TensorShape(d, 3 * c.PatchSize * c.PatchSize));
        _owned.Add(_patchW);
        _patchB = ControlFoleyVitOps.Load(weights, p + "patch_embed_v.proj.bias", _owned, d);
        Tensor pos = ControlFoleyVitOps.Load(weights, p + "pos_embed_v", _owned, 1, c.Tokens, d);
        Tensor modality = ControlFoleyVitOps.Load(weights, p + "modality_v", _owned, 1, 1, d);
        _pos = pos;
        _modality = modality;
        _normW = ControlFoleyVitOps.Load(weights, p + "norm_v.weight", _owned, d);
        _normB = ControlFoleyVitOps.Load(weights, p + "norm_v.bias", _owned, d);
        _blocks = new Block[c.TotalDepth];
        for (int i = 0; i < c.TotalDepth; i++)
        {
            bool visual = i < c.ModalitySpecificDepth;
            string b = visual ? $"{p}blocks_v.{i}" : $"{p}blocks_u.{i - c.ModalitySpecificDepth}";
            string n1 = visual ? "norm1" : "norm1_v", n2 = visual ? "norm2" : "norm2_v";
            _blocks[i] = new Block(
                ControlFoleyVitOps.Load(weights, $"{b}.{n1}.weight", _owned, d), ControlFoleyVitOps.Load(weights, $"{b}.{n1}.bias", _owned, d),
                ControlFoleyVitOps.Load(weights, $"{b}.attn.qkv.weight", _owned, 3 * d, d),
                ControlFoleyVitOps.Load(weights, $"{b}.attn.qkv.bias", _owned, 3 * d),
                ControlFoleyVitOps.Load(weights, $"{b}.attn.proj.weight", _owned, d, d),
                ControlFoleyVitOps.Load(weights, $"{b}.attn.proj.bias", _owned, d),
                ControlFoleyVitOps.Load(weights, $"{b}.{n2}.weight", _owned, d), ControlFoleyVitOps.Load(weights, $"{b}.{n2}.bias", _owned, d),
                ControlFoleyVitOps.Load(weights, $"{b}.mlp.fc1.weight", _owned, hidden, d),
                ControlFoleyVitOps.Load(weights, $"{b}.mlp.fc1.bias", _owned, hidden),
                ControlFoleyVitOps.Load(weights, $"{b}.mlp.fc2.weight", _owned, d, hidden),
                ControlFoleyVitOps.Load(weights, $"{b}.mlp.fc2.bias", _owned, d));
        }
    }

    /// <summary>Encodes normalised frames <c>[T, 3, S, S]</c> (S = <see cref="ControlFoleyCavMaeConfig.InputSize"/>) to
    /// <c>[T, Tokens, EmbedDim]</c> visual tokens (<c>forward_feat_v</c> per frame).</summary>
    public float[] EncodeFrames(IBackend backend, float[] frames)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(frames);
        if (_blocks is null)
        {
            throw new InvalidOperationException("Call LoadWeights before encoding.");
        }

        ControlFoleyCavMaeConfig c = _config;
        int size = c.InputSize, perFrame = 3 * size * size;
        if (frames.Length == 0 || frames.Length % perFrame != 0)
        {
            throw new ArgumentException($"Expected a multiple of 3x{size}x{size} floats, got {frames.Length}.", nameof(frames));
        }

        int count = frames.Length / perFrame, n = c.Tokens, d = c.EmbedDim;
        float[] result = new float[(long)count * n * d];
        for (int start = 0; start < count; start += FrameChunk)
        {
            int m = Math.Min(FrameChunk, count - start);
            using Tensor patches = Patchify(frames, start, m);
            Tensor x = ControlFoleyOps.Linear(backend, patches, _patchW!, _patchB);
            AddPositions(x);
            foreach (Block blk in _blocks)
            {
                x = Run(backend, x, blk);
            }

            using (x)
            using (Tensor normed = ControlFoleyVitOps.LayerNorm(backend, x, _normW!, _normB!, c.LayerNormEps))
            {
                new ReadOnlySpan<float>((void*)normed.DataPointer, m * n * d).CopyTo(result.AsSpan((int)((long)start * n * d), m * n * d));
            }
        }

        return result;
    }

    /// <summary>Frames to <c>[T, EmbedDim]</c> as <c>generate()</c> feeds the network: token mean of <see cref="EncodeFrames"/>.</summary>
    public float[] EncodePooled(IBackend backend, float[] frames)
    {
        float[] tokens = EncodeFrames(backend, frames);
        int n = _config.Tokens, d = _config.EmbedDim, count = tokens.Length / (n * d);
        float[] pooled = new float[count * d];
        for (int f = 0; f < count; f++)
        {
            for (int t = 0; t < n; t++)
            {
                for (int e = 0; e < d; e++)
                {
                    pooled[f * d + e] += tokens[((long)f * n + t) * d + e];
                }
            }

            for (int e = 0; e < d; e++)
            {
                pooled[f * d + e] /= n;
            }
        }

        return pooled;
    }

    private Tensor Patchify(float[] frames, int start, int count)
    {
        ControlFoleyCavMaeConfig c = _config;
        int size = c.InputSize, p = c.PatchSize, grid = c.GridSize, plane = size * size, patchDim = 3 * p * p;
        Tensor o = new(new TensorShape(count, grid * grid, patchDim), DType.F32);
        float* dst = (float*)o.DataPointer;
        for (int f = 0; f < count; f++)
        {
            long frameBase = (long)(start + f) * 3 * plane;
            for (int gy = 0; gy < grid; gy++)
            {
                for (int gx = 0; gx < grid; gx++)
                {
                    float* row = dst + (((long)f * grid + gy) * grid + gx) * patchDim;
                    for (int ch = 0; ch < 3; ch++)
                    {
                        for (int ky = 0; ky < p; ky++)
                        {
                            new ReadOnlySpan<float>(frames, (int)(frameBase + ch * plane + (gy * p + ky) * size + gx * p), p)
                                .CopyTo(new Span<float>(row + (ch * p + ky) * p, p));
                        }
                    }
                }
            }
        }

        return o;
    }

    private void AddPositions(Tensor x)
    {
        int b = (int)x.Shape[0], n = (int)x.Shape[1], d = (int)x.Shape[2];
        float* xp = (float*)x.DataPointer, pos = (float*)_pos!.DataPointer, mod = (float*)_modality!.DataPointer;
        for (int bi = 0; bi < b; bi++)
        {
            for (int t = 0; t < n; t++)
            {
                float* row = xp + ((long)bi * n + t) * d;
                for (int e = 0; e < d; e++)
                {
                    row[e] = row[e] + pos[t * d + e] + mod[e];
                }
            }
        }
    }

    private Tensor Run(IBackend backend, Tensor x, Block b)
    {
        float eps = _config.LayerNormEps;
        using (Tensor normed = ControlFoleyVitOps.LayerNorm(backend, x, b.Norm1W, b.Norm1B, eps))
        using (Tensor attn = ControlFoleyVitOps.SelfAttention(backend, normed, b.QkvW, b.QkvB, b.ProjW, b.ProjB, _config.Heads))
        {
            ControlFoleyVitOps.AddInPlace(x, attn);
        }

        using (Tensor normed = ControlFoleyVitOps.LayerNorm(backend, x, b.Norm2W, b.Norm2B, eps))
        using (Tensor ff = ControlFoleyVitOps.Mlp(backend, normed, b.Fc1W, b.Fc1B, b.Fc2W, b.Fc2B))
        {
            ControlFoleyVitOps.AddInPlace(x, ff);
        }

        return x;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }

        _owned.Clear();
    }
}
