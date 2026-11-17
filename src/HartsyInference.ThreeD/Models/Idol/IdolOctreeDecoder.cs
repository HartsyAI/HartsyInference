using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol's octree occupancy decoder (upstream <c>OctreeProbabilityFixedlenDecoder</c>): for cell centres <c>[1, L, 3]</c> at an octree
/// resolution, four blocks of cross-attention to the 8192-token latent and an MLP, modulated by an embedding of the resolution, give the
/// logits <c>[1, L, 8]</c> of how each cell's points split among its eight children. <see cref="IdolOctreeSampler"/> descends the tree with them.
/// Borrows <see cref="IdolDecoderWeights"/>.</summary>
public sealed class IdolOctreeDecoder : IDisposable
{
    private const int LevelChannels = 256;
    private const float MaxPeriod = 1024f;
    private const float NormEps = 1e-6f;
    private const float FinalNormEps = 1e-5f;

    private readonly IdolDecoderWeights _weights;
    private readonly IdolDecoderConfig _config;
    private readonly IdolDecoderAttention[] _attention;
    private int _disposed;

    public IdolOctreeDecoder(IdolDecoderWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _weights = weights;
        _config = weights.Config;
        _attention = [.. Enumerable.Range(0, _config.OctreeBlocks)
            .Select(i => new IdolDecoderAttention(weights.Tensors, $"octree.blocks.{i}.cross_attn", cross: true, _config.ModelChannels, _config.Heads))];
    }

    /// <summary>The logits <c>[L, 8]</c> (row-major floats) of the cells whose centres are <paramref name="centres"/> (<c>L × 3</c>, in [0, 1]) at
    /// octree <paramref name="resolution"/> (2 to the level), given the latent <paramref name="latent"/> <c>[1, 8192, 16]</c>.</summary>
    public float[] Logits(IBackend backend, ReadOnlySpan<float> centres, int resolution, Tensor latent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        int cells = IdolAnchorEmbedding.CheckedCount(centres), d = _config.ModelChannels;

        using Tensor points = new(new TensorShape(1, cells, 3), DType.F32);
        centres.CopyTo(points.AsSpan<float>());
        using Tensor embedding = IdolAnchorEmbedding.Build(centres, d);
        using Tensor projected = Linear(backend, points, "octree.in_proj", d);
        using Tensor summed = new(projected.Shape, DType.F32);
        backend.Add(summed, projected, embedding);
        Tensor h = Linear(backend, summed, "octree.input_layer", d);

        using Tensor modulation = LevelModulation(backend, resolution);
        Tensor[] m = Split(backend, modulation, d);
        try
        {
            // m: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
            for (int i = 0; i < _attention.Length; i++)
            {
                Tensor next = Block(backend, h, i, latent, m);
                h.Dispose();
                h = next;
            }

            using Tensor normed = new(h.Shape, DType.F32);
            backend.LayerNormNoAffine(normed, h, FinalNormEps);
            using Tensor logits = Linear(backend, normed, "octree.out_proj", 8);
            return logits.AsReadOnlySpan<float>().ToArray();
        }
        finally
        {
            h.Dispose();
            foreach (Tensor t in m)
                t.Dispose();
        }
    }

    private Tensor Block(IBackend backend, Tensor x, int index, Tensor latent, Tensor[] m)
    {
        using Tensor h1 = Modulated(backend, x, m[1], m[0]);
        using Tensor attended = _attention[index].Forward(backend, h1, latent);
        using Tensor x1 = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(x1, x, attended, m[2]);

        using Tensor h2 = Modulated(backend, x1, m[4], m[3]);
        using Tensor hidden = Linear(backend, h2, $"octree.blocks.{index}.mlp.mlp.0", _config.MlpHidden);
        using Tensor act = new(hidden.Shape, DType.F32);
        backend.Gelu(act, hidden);
        using Tensor down = Linear(backend, act, $"octree.blocks.{index}.mlp.mlp.2", _config.ModelChannels);
        Tensor x2 = new(x.Shape, DType.F32);
        backend.GatedResidualLastDim(x2, x1, down, m[5]);
        return x2;
    }

    private static Tensor Modulated(IBackend backend, Tensor x, Tensor scale, Tensor shift)
    {
        Tensor h = new(x.Shape, DType.F32);
        backend.NormModulate(NormModulateCall.Rows(h, x, NormKind.LayerNorm, NormEps)
            .WithModulation(ModulationSpec.Broadcast(x, scale, shift, ScaleMode.OnePlus)));
        return h;
    }

    // adaLN_modulation(l_embedder(resolution)): a sinusoidal embedding with 2π folded in (cosines first), MLP with SiLU, then SiLU and Linear to six vectors.
    private Tensor LevelModulation(IBackend backend, int resolution)
    {
        int half = LevelChannels / 2;
        using Tensor sinusoid = new(new TensorShape(1, LevelChannels), DType.F32);
        Span<float> s = sinusoid.AsSpan<float>();
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(MaxPeriod) * i / half);
            float angle = resolution * freq * 2f * (float)Math.PI;
            s[i] = MathF.Cos(angle);
            s[half + i] = MathF.Sin(angle);
        }

        using Tensor hidden = Linear(backend, sinusoid, "octree.l_embedder.mlp.0", _config.ModelChannels);
        using Tensor act = new(hidden.Shape, DType.F32);
        backend.Silu(act, hidden);
        using Tensor embedded = Linear(backend, act, "octree.l_embedder.mlp.2", _config.ModelChannels);
        using Tensor act2 = new(embedded.Shape, DType.F32);
        backend.Silu(act2, embedded);
        return Linear(backend, act2, "octree.adaLN_modulation.1", 6L * _config.ModelChannels);
    }

    private static Tensor[] Split(IBackend backend, Tensor modulation, int width)
    {
        Tensor[] parts = new Tensor[6];
        try
        {
            for (int i = 0; i < parts.Length; i++)
            {
                parts[i] = new Tensor(new TensorShape(1, width), DType.F32);
                backend.SliceLastDim(parts[i], modulation, i * width);
            }
        }
        catch
        {
            foreach (Tensor? t in parts)
                t?.Dispose();
            throw;
        }

        return parts;
    }

    private Tensor Linear(IBackend backend, Tensor x, string layer, long outDim)
    {
        Tensor o = new(new TensorShape(1, x.Shape[x.Shape.Rank - 2], outDim), DType.F32);
        backend.Linear(o, x, _weights[$"{layer}.weight"], _weights[$"{layer}.bias"]);
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (IdolDecoderAttention a in _attention)
            a.Dispose();
    }
}
