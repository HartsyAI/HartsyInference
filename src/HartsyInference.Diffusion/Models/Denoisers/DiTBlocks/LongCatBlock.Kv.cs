using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>One block's cached keys and values: token-major <c>[S, heads, headDim]</c>, after the QK norm and the RoPE, as the reference keeps
/// them. The holder disposes them.</summary>
public sealed class LongCatBlockKv : IDisposable
{
    /// <summary>Wraps the two tensors; the instance owns them.</summary>
    public LongCatBlockKv(Tensor k, Tensor v)
    {
        K = k;
        V = v;
    }

    /// <summary>Keys <c>[S, heads, headDim]</c>.</summary>
    public Tensor K { get; }

    /// <summary>Values <c>[S, heads, headDim]</c>.</summary>
    public Tensor V { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        K.Dispose();
        V.Dispose();
    }
}

/// <summary>The KV-cache path of a LongCat block (the reference's <c>forward_with_kv_cache</c> and its <c>return_kv</c> pass). The cache pass runs
/// the conditioning rows through self-attention and the FFN only (no text, no audio), keeping their keys and values. The denoising pass runs the
/// noisy rows with the cache in front of their own keys; its text and audio steps then cover every row.</summary>
public sealed partial class LongCatBlock
{
    /// <summary>The cache pass over the conditioning rows: self-attention and the FFN, with the rows' keys and values kept in <paramref name="kv"/>.
    /// The caller owns <paramref name="kv"/> and the returned stream.</summary>
    public Tensor ForwardCacheBuild(IBackend backend, Tensor hidden, Tensor timeEmbed, in LongCatBlockInputs inputs, out LongCatBlockKv? kv)
    {
        int dim = _config.HiddenSize, hw = inputs.TokensPerFrame;
        int s = checked(inputs.Frames * hw);
        if (hidden.DType != DType.F32 || hidden.Shape.Rank != 2 || hidden.Shape[0] != s || hidden.Shape[1] != dim)
            throw new ArgumentException($"The LongCat stream is F32 [{s}, {dim}]; got {hidden.DType} {hidden.Shape}.", nameof(hidden));
        if (timeEmbed.Shape[0] != inputs.Frames)
            throw new ArgumentException($"The timestep embedding has {timeEmbed.Shape[0]} rows for {inputs.Frames} frames.", nameof(timeEmbed));

        using Tensor modulation = LongCatOps.AdaLnProjection(backend, timeEmbed, _adaWeight!, _adaBias);
        using Tensor normed1 = LongCatOps.NormModulate(backend, hidden, modulation, dim, 0, hw);
        using Tensor attn = CachedSelfAttention(backend, normed1, inputs.Cos, inputs.Sin, null, keep: true, inputs.Segments, out kv);
        using Tensor afterSelf = LongCatOps.GatedResidual(backend, hidden, attn, modulation, 2 * dim, dim, inputs.FrameRowIndex);
        using Tensor normed3 = LongCatOps.NormModulate(backend, afterSelf, modulation, 4 * dim, 3 * dim, hw);
        using Tensor ffn = FeedForward(backend, normed3);
        return LongCatOps.GatedResidual(backend, afterSelf, ffn, modulation, 5 * dim, dim, inputs.FrameRowIndex);
    }

    /// <summary>Self-attention over token-major <paramref name="input"/> rows. Without a cache, the rows attend to their own keys. With one, they
    /// attend to the cached rows followed by their own (the reference's <c>forward_with_kv_cache</c>, without the reference-map masking). When
    /// <paramref name="keep"/> is set, this call's keys and values are copied out in <paramref name="kept"/>. The caller owns the result.</summary>
    private Tensor CachedSelfAttention(IBackend backend, Tensor input, Tensor cos, Tensor sin, LongCatBlockKv? cache, bool keep, AttentionSegment[] segments,
        out LongCatBlockKv? kept)
    {
        kept = null;
        int s = (int)input.Shape[0];
        QkRotation rotation = new(QkRotationKind.InterleavedPairs, cos, sin, s);
        AttentionCallContext context = new() { QueryRotation = rotation, KeyRotation = rotation };
        (Tensor q, Tensor k, Tensor v) = _selfAttention!.ProjectSelf(backend, input, releaseInput: false);
        Tensor? queries = null, keys = null, values = null;
        try
        {
            _selfAttention.RotateQk(backend, q, k, in context);
            if (keep)
                kept = new LongCatBlockKv(LongCatOps.Copy(backend, k), LongCatOps.Copy(backend, v));
            if (cache is null)
            {
                Tensor dense = SegmentedAttention.Run(backend, _selfAttention, q, k, v, segments);
                return _selfAttention.ProjectOutput(backend, dense);
            }

            // One segmented pass over the cached rows then the new rows (SegmentedAttention takes equal-length operands). The cached rows'
            // queries are unused: they take the cached keys as dummy queries and attend to the cached rows only. The new rows attend to all.
            int cached = (int)cache.K.Shape[0], total = cached + s;
            int heads = _config.NumHeads, headDim = _config.HeadDim;
            queries = new Tensor(new TensorShape(total, heads, headDim), DType.F32);
            backend.Concat(queries, [cache.K, q], 0);
            keys = new Tensor(new TensorShape(total, heads, headDim), DType.F32);
            backend.Concat(keys, [cache.K, k], 0);
            values = new Tensor(new TensorShape(total, heads, headDim), DType.F32);
            backend.Concat(values, [cache.V, v], 0);
            AttentionSegment[] pass = [new AttentionSegment(0, cached, 0, cached), new AttentionSegment(cached, s, 0, total)];
            using Tensor all = SegmentedAttention.Run(backend, _selfAttention, queries, keys, values, pass);
            Tensor rows = new(new TensorShape(s, heads * headDim), DType.F32);
            backend.SliceRows(rows, all, cached);
            return _selfAttention.ProjectOutput(backend, rows);
        }
        catch
        {
            kept?.Dispose();
            kept = null;
            throw;
        }
        finally
        {
            queries?.Dispose();
            keys?.Dispose();
            values?.Dispose();
            q.Dispose();
            k.Dispose();
            v.Dispose();
        }
    }


}
