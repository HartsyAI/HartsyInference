using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.ThreeD.Models.TrellisV2;

/// <summary>The parts the dense and sparse TRELLIS.2 flows share: input projection, timestep embedder and the shared adaLN
/// modulation, the block stack, and the final norm and output projection. Borrows <see cref="Trellis2FlowWeights"/>.</summary>
internal sealed class Trellis2FlowCore : IDisposable
{
    /// <summary>Epsilon of the final <c>F.layer_norm</c>.</summary>
    internal const float FinalNormEps = 1e-5f;

    private readonly Trellis2FlowConfig _config;
    private readonly Tensor _inW, _inB, _t0W, _t0B, _t2W, _t2B, _modW, _modB, _outW, _outB;
    private readonly Trellis2FlowBlock[] _blocks;
    private int _disposed;

    internal Trellis2FlowCore(Trellis2FlowWeights weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _config = weights.Config;
        _inW = weights["input_layer.weight"];
        _inB = weights["input_layer.bias"];
        _t0W = weights["t_embedder.mlp.0.weight"];
        _t0B = weights["t_embedder.mlp.0.bias"];
        _t2W = weights["t_embedder.mlp.2.weight"];
        _t2B = weights["t_embedder.mlp.2.bias"];
        _modW = weights["adaLN_modulation.1.weight"];
        _modB = weights["adaLN_modulation.1.bias"];
        _outW = weights["out_layer.weight"];
        _outB = weights["out_layer.bias"];
        _blocks = new Trellis2FlowBlock[_config.NumBlocks];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new Trellis2FlowBlock(weights, i);
    }

    internal Trellis2FlowConfig Config => _config;

    internal IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in new[] { _inW, _inB, _t0W, _t0B, _t2W, _t2B, _modW, _modB, _outW, _outB })
            yield return t;
        foreach (Trellis2FlowBlock block in _blocks)
        {
            foreach (Tensor t in block.EnumerateWeights())
                yield return t;
        }
    }

    /// <summary><c>input_layer</c> over token rows <c>[1, N, InChannels]</c> to <c>[1, N, C]</c>.</summary>
    internal Tensor Embed(IBackend backend, Tensor tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        Tensor h = new(new TensorShape(1, tokens.Shape[1], _config.ModelChannels), DType.F32);
        backend.Linear(h, tokens, _inW, _inB);
        return h;
    }

    /// <summary>The model-wide modulation <c>[1, 6C]</c> for a model timestep (<c>1000 · t</c>): sinusoidal embedding (cosines then
    /// sines, base 10000), Linear, SiLU, Linear, then SiLU and the shared adaLN Linear.</summary>
    internal Tensor SharedModulation(IBackend backend, float tModel)
    {
        int c = _config.ModelChannels;
        using Tensor freq = new(new TensorShape(1, Trellis2FlowConfig.TimestepFrequencyDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(freq, tModel, 1, Trellis2FlowConfig.TimestepFrequencyDim);
        using Tensor h0 = new(new TensorShape(1, c), DType.F32);
        backend.Linear(h0, freq, _t0W, _t0B);
        using Tensor a0 = new(h0.Shape, DType.F32);
        backend.Silu(a0, h0);
        using Tensor emb = new(new TensorShape(1, c), DType.F32);
        backend.Linear(emb, a0, _t2W, _t2B);
        using Tensor act = new(emb.Shape, DType.F32);
        backend.Silu(act, emb);
        Tensor mod = new(new TensorShape(1, 6L * c), DType.F32);
        backend.Linear(mod, act, _modW, _modB);
        return mod;
    }

    /// <summary>Checks the per-token features against the architecture: required as <c>[1, tokens, ProjInChannels]</c> by a
    /// projection flow, and refused by a plain one so a mismatched caller fails instead of silently dropping them.</summary>
    internal void CheckProjection(Tensor? proj, long tokens)
    {
        if (!_config.HasProjection)
        {
            if (proj is not null)
                throw new ArgumentException("This flow has no projection branch; pass null for the projected features.", nameof(proj));
            return;
        }

        if (proj is null || proj.Shape.Rank != 3 || proj.Shape[0] != 1 || proj.Shape[1] != tokens || proj.Shape[2] != _config.ProjInChannels)
            throw new ArgumentException($"Projected features must be [1, {tokens}, {_config.ProjInChannels}]; got {proj?.Shape.ToString() ?? "null"}.", nameof(proj));
    }

    /// <summary>Runs every block over <paramref name="h"/> and consumes it; <paramref name="proj"/> is the projection mode's per-token features.</summary>
    internal Tensor RunBlocks(IBackend backend, Tensor h, Tensor sharedMod, Tensor cond, Tensor? proj, Trellis2RopeTable rope)
    {
        Tensor x = h;
        foreach (Trellis2FlowBlock block in _blocks)
        {
            Tensor next;
            try
            {
                next = block.Forward(backend, x, sharedMod, cond, proj, rope);
            }
            catch
            {
                x.Dispose();
                throw;
            }

            x.Dispose();
            x = next;
        }

        return x;
    }

    /// <summary>Non-affine layer norm (eps 1e-5) then <c>out_layer</c>, <c>[1, N, C]</c> to <c>[1, N, OutChannels]</c>; consumes <paramref name="h"/>.</summary>
    internal Tensor Project(IBackend backend, Tensor h)
    {
        using Tensor owned = h;
        using Tensor normed = new(h.Shape, DType.F32);
        backend.LayerNormNoAffine(normed, h, FinalNormEps);
        Tensor o = new(new TensorShape(1, h.Shape[1], _config.OutChannels), DType.F32);
        backend.Linear(o, normed, _outW, _outB);
        return o;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Trellis2FlowBlock block in _blocks)
            block.Dispose();
    }
}
