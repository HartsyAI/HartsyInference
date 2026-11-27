using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The video stream of one LTX-2.0 transformer block (<c>LTX2VideoTransformerBlock</c> with its audio and
/// audio↔video parts left out, as SANA-WM's refiner runs it in <c>_forward_video_block</c>), single sample over
/// <c>[S, dim]</c> rows: AdaLN-Single modulated self-attention with a gate, ungated text cross-attention, and the modulated,
/// gated gelu-approximate FFN. The per-block <c>scale_shift_table</c> has six rows (shift, scale and gate for the
/// self-attention, then for the FFN); the cross-attention is not modulated in LTX-2.0.</summary>
/// <remarks>Only the video keys of a block are read, so an audio-video checkpoint loads with its audio weights untouched.
/// The self-attention is <see cref="LtxVideo2PrefixSelfAttention"/>, which makes the block chunk-causal when it is given a
/// stream.</remarks>
internal sealed class LtxVideo2VideoBlock : IStreamingBlock
{
    private const int ModulationRows = 6;

    private readonly int _dim;
    private readonly float _normEps;
    private readonly LtxVideo2PrefixSelfAttention _attn1;
    private readonly LtxVideo2Attention _attn2;

    private Tensor? _table;     // [6, dim]
    private Tensor? _ffW0, _ffB0, _ffW2, _ffB2;

    public LtxVideo2VideoBlock(LtxVideo2Config config)
    {
        _dim = config.InnerDim;
        _normEps = config.NormEps;
        _attn1 = new LtxVideo2PrefixSelfAttention(_dim, config.NumHeads, config.HeadDim, config.QkNormEps);
        _attn2 = new LtxVideo2Attention(_dim, config.CrossAttentionDim, config.NumHeads, config.HeadDim, _dim, config.QkNormEps);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string p)
    {
        _attn1.LoadWeights(w, $"{p}.attn1");
        _attn2.LoadWeights(w, $"{p}.attn2");
        _table = TensorCasts.LoadF32(w, $"{p}.scale_shift_table");
        if (_table.Shape.Rank != 2 || _table.Shape[0] != ModulationRows || _table.Shape[1] != _dim)
            throw new InvalidOperationException($"'{p}.scale_shift_table' is {_table.Shape}; an LTX-2.0 video block has "
                + $"[{ModulationRows}, {_dim}]. A table of nine rows is the LTX-2.3 layout, which LtxVideo2Transformer runs.");
        _ffW0 = w[$"{p}.ff.net.0.proj.weight"]; w.TryGetValue($"{p}.ff.net.0.proj.bias", out _ffB0);
        _ffW2 = w[$"{p}.ff.net.2.weight"]; w.TryGetValue($"{p}.ff.net.2.bias", out _ffB2);
    }

    /// <inheritdoc />
    public long EstimatedWeightBytes
    {
        get
        {
            long total = 0;
            foreach (Tensor t in EnumerateWeights()) total += t.DType.ComputeByteCount(t.ElementCount);
            return total;
        }
    }

    /// <inheritdoc />
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in _attn1.EnumerateWeights()) yield return t;
        foreach (Tensor t in _attn2.EnumerateWeights()) yield return t;
        foreach (Tensor? t in new[] { _table, _ffW0, _ffB0, _ffW2, _ffB2 })
            if (t is not null) yield return t;
    }

    /// <summary>Runs the block over <paramref name="hidden"/>, which it takes over, and returns the new hidden state. With
    /// <paramref name="captureOnly"/> the block stops once its self-attention has staged its keys and values and hands
    /// <paramref name="hidden"/> back unchanged. A failure disposes the hidden state.</summary>
    public Tensor Forward(IBackend backend, Tensor hidden, LtxVideo2ChunkPass pass, LtxVideo2LayerStream? stream, bool captureOnly)
    {
        Tensor current = hidden;
        Tensor[]? mod = null;
        bool done = false;
        try
        {
            // shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            mod = LtxVideo2BlockOps.Modulation(backend, _table!, 0, pass.Temb, ModulationRows, _dim);
            using (Tensor n1 = LtxVideo2BlockOps.NormModulate(backend, current, mod[1], mod[0], _dim, _normEps))
            using (Tensor? attn1 = _attn1.Forward(backend, n1, pass, stream, captureOnly))
            {
                if (attn1 is null)
                {
                    done = true;
                    return current;
                }

                current = LtxVideo2BlockOps.GatedAddInto(backend, current, attn1, mod[2]);
            }

            using (Tensor n2 = LtxVideo2BlockOps.RmsNoAffine(backend, current, _dim, _normEps))
            using (Tensor attn2 = _attn2.Forward(backend, n2, pass.Encoder, null, null, null, null, null, null, null))
                current = LtxVideo2BlockOps.GatedAddInto(backend, current, attn2, pass.Ones);

            using (Tensor n3 = LtxVideo2BlockOps.NormModulate(backend, current, mod[4], mod[3], _dim, _normEps))
            using (Tensor ff = LtxVideo2BlockOps.Ffn(backend, n3, _ffW0!, _ffB0, _ffW2!, _ffB2, _dim))
                current = LtxVideo2BlockOps.GatedAddInto(backend, current, ff, mod[5]);
            done = true;
            return current;
        }
        finally
        {
            if (mod is not null)
            {
                foreach (Tensor t in mod)
                    t.Dispose();
            }

            if (!done)
                current.Dispose();
        }
    }
}
