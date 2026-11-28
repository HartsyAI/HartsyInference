using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The KV-cache path of a LongCat block (the reference's <c>forward_with_kv_cache</c> and its <c>return_kv</c> pass). The cache pass runs
/// the conditioning rows through self-attention and the FFN only (no text, no audio), keeping their keys and values. The denoising pass runs the
/// noisy rows with the cache in front of their own keys; its text and audio steps then cover every row.</summary>
public sealed partial class LongCatBlock
{
    /// <summary>The cache pass over the conditioning rows: self-attention and the FFN, with the rows' keys and values stored as layer
    /// <paramref name="layer"/> of <paramref name="store"/>. The caller owns the returned stream.</summary>
    internal Tensor ForwardCacheBuild(IBackend backend, Tensor hidden, Tensor timeEmbed, in LongCatBlockInputs inputs, KvSegmentStore store, int layer)
    {
        int dim = _config.HiddenSize, hw = inputs.TokensPerFrame;
        int s = checked(inputs.Frames * hw);
        if (hidden.DType != DType.F32 || hidden.Shape.Rank != 2 || hidden.Shape[0] != s || hidden.Shape[1] != dim)
            throw new ArgumentException($"The LongCat stream is F32 [{s}, {dim}]; got {hidden.DType} {hidden.Shape}.", nameof(hidden));
        if (timeEmbed.Shape[0] != inputs.Frames)
            throw new ArgumentException($"The timestep embedding has {timeEmbed.Shape[0]} rows for {inputs.Frames} frames.", nameof(timeEmbed));

        using Tensor modulation = LongCatOps.AdaLnProjection(backend, timeEmbed, _adaWeight!, _adaBias);
        using Tensor normed1 = LongCatOps.NormModulate(backend, hidden, modulation, dim, 0, hw, _bf16Residual);
        using Tensor attn = Sub(backend, StoredSelfAttention(backend, normed1, in inputs, store, layer));
        using Tensor afterSelf = Residual(backend, LongCatOps.GatedResidual(backend, hidden, attn, modulation, 2 * dim, dim, inputs.FrameRowIndex));
        using Tensor normed3 = LongCatOps.NormModulate(backend, afterSelf, modulation, 4 * dim, 3 * dim, hw, _bf16Residual);
        using Tensor ffn = Sub(backend, FeedForward(backend, normed3));
        return Residual(backend, LongCatOps.GatedResidual(backend, afterSelf, ffn, modulation, 5 * dim, dim, inputs.FrameRowIndex));
    }

    /// <summary>Self-attention of the rows over themselves; their rotated keys and values go to <paramref name="store"/>.</summary>
    private Tensor StoredSelfAttention(IBackend backend, Tensor input, in LongCatBlockInputs inputs, KvSegmentStore store, int layer)
    {
        (Tensor q, Tensor k, Tensor v) = ProjectRotated(backend, input, inputs.Cos, inputs.Sin);
        using (q)
        using (k)
        using (v)
        {
            Tensor keys = HeadMajor(backend, k), values;
            try
            {
                values = HeadMajor(backend, v);
            }
            catch
            {
                keys.Dispose();
                throw;
            }
            store.Set(layer, 0, keys, values);
            Tensor flat = LongCatOps.RoundOwned(backend, SegmentedAttention.Run(backend, _selfAttention!, q, k, v, inputs.Segments), _bf16Residual);
            return Bf16Host(backend) ? OutputBf16(backend, flat) : _selfAttention!.ProjectOutput(backend, flat);
        }
    }

    /// <summary>Self-attention of the rows over the cached rows followed by themselves (the reference's <c>forward_with_kv_cache</c>). The cached
    /// rows are keys and values only. With several speakers, <paramref name="speakerMap"/> is the reference map of these rows against the cache's
    /// first frame. The caller owns the result.</summary>
    private Tensor CachedSelfAttention(IBackend backend, Tensor input, Tensor cos, Tensor sin, in LongCatPastKv past, LongCatSpeakerPass? speakers,
        int tokensPerFrame, out float[]? speakerMap)
    {
        speakerMap = null;
        int s = (int)input.Shape[0];
        (Tensor q, Tensor k, Tensor v) = ProjectRotated(backend, input, cos, sin);
        Tensor? queries = null, keys = null, values = null;
        Tensor attended;
        try
        {
            if (speakers is not null)
            {
                using Tensor referenceKeys = new(new TensorShape(1, _config.NumHeads, tokensPerFrame, _config.HeadDim), DType.F32);
                backend.CopyStrided(referenceKeys, past.Keys, StridedCopySpec.Gather(StridedView.Of(past.Keys.Shape).Slice(2, 0, tokensPerFrame)));
                speakerMap = speakers.ReferenceMap(backend, _selfAttention!, q, 0, referenceKeys);
            }
            queries = HeadMajor(backend, q);
            keys = WithPast(backend, past.Keys, k);
            values = WithPast(backend, past.Values, v);
            attended = _selfAttention!.AttendHeadMajorOperands(backend, queries, keys, values, default, OperandOwnership.None);
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
        Tensor merged = LongCatOps.RoundOwned(backend, _selfAttention.MergeHeads(backend, attended, new TensorShape(s, _config.HiddenSize)), _bf16Residual);
        return Bf16Host(backend) ? OutputBf16(backend, merged) : _selfAttention.ProjectOutput(backend, merged);
    }

    /// <summary>The rows' token-major queries, keys and values, queries and keys rotated. The caller owns all three.</summary>
    private (Tensor Q, Tensor K, Tensor V) ProjectRotated(IBackend backend, Tensor input, Tensor cos, Tensor sin)
    {
        QkRotation rotation = new(QkRotationKind.InterleavedPairs, cos, sin, (int)input.Shape[0]);
        AttentionCallContext context = new() { QueryRotation = rotation, KeyRotation = rotation };
        (Tensor q, Tensor k, Tensor v) = ProjectSelfTiled(backend, input);
        try
        {
            _selfAttention!.RotateQk(backend, q, k, in context);
            q = LongCatOps.RoundOwned(backend, q, _bf16Residual);
            k = LongCatOps.RoundOwned(backend, k, _bf16Residual);
            return (q, k, v);
        }
        catch
        {
            q.Dispose();
            k.Dispose();
            v.Dispose();
            throw;
        }
    }

    /// <summary>Head-major <c>[1, heads, past + S, headDim]</c>: <paramref name="past"/> followed by the token-major rows of <paramref name="now"/>.</summary>
    private Tensor WithPast(IBackend backend, Tensor past, Tensor now)
    {
        using Tensor rows = HeadMajor(backend, now);
        Tensor joined = new(new TensorShape(1, _config.NumHeads, past.Shape[2] + rows.Shape[2], _config.HeadDim), DType.F32);
        try
        {
            backend.Concat(joined, [past, rows], 2);
        }
        catch
        {
            joined.Dispose();
            throw;
        }
        return joined;
    }

    /// <summary>The token-major rows as head-major <c>[1, heads, S, headDim]</c>.</summary>
    private Tensor HeadMajor(IBackend backend, Tensor rows)
    {
        int heads = _config.NumHeads, headDim = _config.HeadDim, s = (int)(rows.ElementCount / _config.HiddenSize);
        Tensor headMajor = new(new TensorShape(1, heads, s, headDim), DType.F32);
        try
        {
            backend.Permute0213(headMajor, rows, s, heads, headDim);
        }
        catch
        {
            headMajor.Dispose();
            throw;
        }
        return headMajor;
    }
}

/// <summary>One block's cached keys and values, head-major <c>[1, heads, S, headDim]</c>, borrowed from the <see cref="LongCatKvCache"/>.</summary>
internal readonly record struct LongCatPastKv(Tensor Keys, Tensor Values);
