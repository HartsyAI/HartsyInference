using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The KV cache of one conditioning pass (the reference's <c>kv_cache_dict</c>): every block's rotated keys and values over
/// <see cref="Tokens"/> conditioning rows, head-major and pinned on the backend that built it. The holder disposes it.</summary>
public sealed class LongCatKvCache : IDisposable
{
    private readonly KvSegmentStore _store;

    internal LongCatKvCache(KvSegmentStore store, IBackend backend, int frames, int gridH, int gridW)
    {
        _store = store;
        Backend = backend;
        Frames = frames;
        GridH = gridH;
        GridW = gridW;
    }

    /// <summary>The backend that built the cache and holds its tensors.</summary>
    internal IBackend Backend { get; }

    /// <summary>The conditioning frames.</summary>
    internal int Frames { get; }

    /// <summary>Token rows of one frame.</summary>
    internal int GridH { get; }

    /// <summary>Token columns of one frame.</summary>
    internal int GridW { get; }

    /// <summary>Blocks the cache spans.</summary>
    public int Layers => _store.Layers;

    /// <summary>The cached rows per block: the conditioning frames times the tokens per frame.</summary>
    public int Tokens => Frames * GridH * GridW;

    /// <summary>Bytes the cache takes.</summary>
    public long Bytes => _store.Bytes;

    /// <summary>Bytes a cache over <paramref name="condFrames"/> latent frames of <paramref name="latentHeight"/> × <paramref name="latentWidth"/> will take.</summary>
    public static long EstimateBytes(LongCatConfig config, int condFrames, int latentHeight, int latentWidth)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentOutOfRangeException.ThrowIfLessThan(condFrames, 1);
        LongCatTransformer.RequireTiling(config, condFrames, latentHeight, latentWidth);
        long tokens = checked((long)condFrames * (latentHeight / config.PatchSize.H) * (latentWidth / config.PatchSize.W));
        return KvSegmentStore.EstimateBytes(config.Depth, tokens, config.HiddenSize, DType.F32);
    }

    /// <summary>The keys of block <paramref name="layer"/>, <c>[1, heads, Tokens, headDim]</c>, borrowed.</summary>
    internal Tensor Keys(int layer) => _store.Keys(layer, 0);

    /// <summary>The values of block <paramref name="layer"/>, <c>[1, heads, Tokens, headDim]</c>, borrowed.</summary>
    internal Tensor Values(int layer) => _store.Values(layer, 0);

    /// <inheritdoc/>
    public void Dispose() => _store.Dispose();
}

/// <summary>The KV-cache path of the LongCat DiT (the reference's <c>return_kv</c> pass and <c>kv_cache_dict</c> denoising). The cache pass takes
/// the conditioning latents (timesteps zero) through the conditioning attention plan and the FFN, with no text or audio, as the pipeline's
/// <c>_cache_clean_latents</c> does, and keeps every block's keys and values. The denoising pass takes only the noisy latents: their rope rows
/// follow the cache, their attention covers the cache and themselves, and their text and audio steps cover every row.</summary>
public sealed partial class LongCatTransformer
{
    /// <summary>The cache pass over conditioning latents <c>[1, C, F, H, W]</c>, where <c>F</c> is <paramref name="conditioning"/>'s conditioning frame
    /// count. The reference's <c>_cache_clean_latents</c>: zero timesteps, <c>skip_crs_attn</c> and <c>return_kv</c>.</summary>
    public LongCatKvCache BuildKvCache(IBackend backend, Tensor condLatent, in LongCatConditioning conditioning)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(condLatent);
        if (!_loaded)
            throw new InvalidOperationException("The transformer has no weights: LoadWeights was never called, or it failed.");
        if (condLatent.Shape.Rank != 5 || condLatent.Shape[0] != 1 || condLatent.Shape[1] != _config.InChannels)
            throw new ArgumentException($"The conditioning latent must be [1, {_config.InChannels}, F, H, W], got {condLatent.Shape}.", nameof(condLatent));
        int frames = (int)condLatent.Shape[2], height = (int)condLatent.Shape[3], width = (int)condLatent.Shape[4];
        if (frames != conditioning.NumCondLatents)
            throw new ArgumentException($"The conditioning latent has {frames} frames, but the conditioning says {conditioning.NumCondLatents}.", nameof(condLatent));
        RequireTiling(frames, height, width);
        RequireNoMask(conditioning);
        int dim = _config.HiddenSize;
        int gridH = height / _config.PatchSize.H, gridW = width / _config.PatchSize.W, hw = gridH * gridW;

        using Tensor timeEmbed = TimestepEmbedding(backend, new float[] { 0f }, frames);
        Tensor? cur = null;
        KvSegmentStore store = new(backend, _blocks.Length, tokenAxis: 2);
        try
        {
            cur = WanDitOps.Patchify(backend, condLatent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchBias);
            (Tensor cos, Tensor sin) = RopeFor(frames, gridH, gridW, conditioning);
            AttentionSegment[] segments = conditioning.AttentionPlan(hw, frames);
            LongCatBlockInputs inputs = new(frames, hw, conditioning.NumCondLatents, cos, sin, segments, FrameRowIndex(frames, hw));
            for (int i = 0; i < _blocks.Length; i++)
            {
                BeforeBlockForward?.Invoke(i);
                Tensor next = _blocks[i].ForwardCacheBuild(backend, cur, timeEmbed, in inputs, store, i);
                cur.Dispose();
                cur = next;
            }
        }
        catch
        {
            cur?.Dispose();
            store.Dispose();
            throw;
        }
        cur.Dispose();
        return new LongCatKvCache(store, backend, frames, gridH, gridW);
    }

    /// <summary>The denoising pass over the noisy latents <c>[1, C, N, H, W]</c> with <paramref name="cache"/> in front of them. The conditioning
    /// frames are not in <paramref name="noisyLatent"/>: <paramref name="conditioning"/> still says how many there were, and the rope rows
    /// start after the cache. <paramref name="frameTimesteps"/> is one timestep for all noisy frames, or one per noisy frame.
    /// <paramref name="audioTokens"/> are the noisy frames' audio tokens: exactly <c>N·contextTokens</c> rows, the last N frames of the projection.</summary>
    public Tensor ForwardCached(IBackend backend, Tensor noisyLatent, Tensor encoder, ReadOnlySpan<float> frameTimesteps,
        in LongCatConditioning conditioning, LongCatKvCache cache, int validTextTokens = -1, Tensor? audioTokens = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(noisyLatent);
        ArgumentNullException.ThrowIfNull(cache);
        if (!_loaded)
            throw new InvalidOperationException("The transformer has no weights: LoadWeights was never called, or it failed.");
        if (!ReferenceEquals(cache.Backend, backend))
            throw new ArgumentException("The cache was built on another backend.", nameof(cache));
        if (cache.Layers != _blocks.Length)
            throw new ArgumentException($"The cache has {cache.Layers} blocks; the transformer has {_blocks.Length}.", nameof(cache));
        if (noisyLatent.Shape.Rank != 5 || noisyLatent.Shape[0] != 1 || noisyLatent.Shape[1] != _config.InChannels)
            throw new ArgumentException($"The noisy latent must be [1, {_config.InChannels}, N, H, W], got {noisyLatent.Shape}.", nameof(noisyLatent));

        int frames = (int)noisyLatent.Shape[2], height = (int)noisyLatent.Shape[3], width = (int)noisyLatent.Shape[4];
        RequireTiling(frames, height, width);
        RequireNoMask(conditioning);
        conditioning.Validate(conditioning.NumCondLatents + frames);
        if (frameTimesteps.Length != 1 && frameTimesteps.Length != frames)
            throw new ArgumentException($"The timesteps hold {frameTimesteps.Length} values; expected 1 or {frames} (one per noisy frame).", nameof(frameTimesteps));
        if (_audioProjection is not null && audioTokens is null)
            throw new ArgumentException("This avatar checkpoint needs the noisy frames' audio tokens.", nameof(audioTokens));
        if (_audioProjection is null && audioTokens is not null)
            throw new ArgumentException("This checkpoint has no audio projection, so audio tokens do not apply.", nameof(audioTokens));
        if (audioTokens is not null)
        {
            int ctx = _config.AudioContextTokens, outDim = _config.AudioOutputDim;
            if (audioTokens.Shape.Rank != 2 || audioTokens.Shape[0] != (long)frames * ctx || audioTokens.Shape[1] != outDim)
                throw new ArgumentException($"The audio tokens must be [{frames}·{ctx}, {outDim}] (the noisy frames only), got {audioTokens.Shape}.", nameof(audioTokens));
        }

        int dim = _config.HiddenSize, gridH = height / _config.PatchSize.H, gridW = width / _config.PatchSize.W, hw = gridH * gridW;
        // The cached keys carry the rope positions of their own grid, so the same row count on another grid is not the same cache.
        if (cache.Frames != conditioning.NumCondLatents || cache.GridH != gridH || cache.GridW != gridW)
            throw new ArgumentException($"The cache holds {cache.Frames} frames of {cache.GridH}×{cache.GridW} tokens; the pass needs "
                + $"{conditioning.NumCondLatents} of {gridH}×{gridW}.", nameof(cache));
        int rows = frames * hw;

        // The rope rows of the noisy frames: the full table covers the conditioning frames then the noisy ones; the noisy part starts at the cache.
        (Tensor fullCos, Tensor fullSin) = RopeFor(conditioning.NumCondLatents + frames, gridH, gridW, conditioning);
        using Tensor cos = new(new TensorShape(rows, fullCos.Shape[1]), DType.F32);
        using Tensor sin = new(new TensorShape(rows, fullSin.Shape[1]), DType.F32);
        backend.SliceRows(cos, fullCos, cache.Tokens);
        backend.SliceRows(sin, fullSin, cache.Tokens);

        using Tensor timeEmbed = TimestepEmbedding(backend, frameTimesteps, frames);
        using Tensor text = CaptionEmbedding(backend, encoder, validTextTokens);
        Tensor? cur = null;
        try
        {
            cur = WanDitOps.Patchify(backend, noisyLatent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchBias);
            AttentionSegment[] segments = [AttentionSegment.Dense(rows)];
            LongCatBlockInputs inputs = new(frames, hw, 0, cos, sin, segments, FrameRowIndex(frames, hw));
            for (int i = 0; i < _blocks.Length; i++)
            {
                BeforeBlockForward?.Invoke(i);
                Tensor next = _blocks[i].Forward(backend, cur, text, timeEmbed, in inputs, new LongCatPastKv(cache.Keys(i), cache.Values(i)),
                    Probe, Probe is null ? "" : $"block{i}", audioTokens);
                cur.Dispose();
                cur = next;
            }

            using Tensor finalModulation = LongCatOps.AdaLnProjection(backend, timeEmbed, _finalAdaW!, _finalAdaB);
            using Tensor modulated = LongCatOps.NormModulate(backend, cur, finalModulation, dim, 0, hw);
            cur.Dispose();
            cur = null;
            using Tensor projected = new(new TensorShape((long)rows, _finalW!.Shape[0]), DType.F32);
            backend.Linear(projected, modulated, _finalW!, _finalB);
            Probe?.Invoke("final_layer", projected);
            return WanDitOps.Unpatchify(projected, _config.OutChannels, frames / _config.PatchSize.T, gridH, gridW, _config.PatchSize);
        }
        finally
        {
            cur?.Dispose();
        }
    }

    /// <summary>The rope tables of the first <paramref name="totalFrames"/> frames under <paramref name="conditioning"/>'s frame indices, cached.</summary>
    private (Tensor Cos, Tensor Sin) RopeFor(int totalFrames, int gridH, int gridW, in LongCatConditioning conditioning)
    {
        int roperef = conditioning.NumRefLatents > 0 ? conditioning.RefImgIndex + 1 : 0;
        LongCatConditioning captured = conditioning;
        return _caches.RopeCosSin((totalFrames, gridH, gridW, roperef), () => BuildRope(totalFrames, gridH, gridW, captured));
    }

    /// <summary>The latent grid must tile the patch size exactly, in time and space.</summary>
    private void RequireTiling(int frames, int height, int width) => RequireTiling(_config, frames, height, width);

    /// <inheritdoc cref="RequireTiling(int, int, int)"/>
    internal static void RequireTiling(LongCatConfig config, int frames, int height, int width)
    {
        if (frames % config.PatchSize.T != 0 || height % config.PatchSize.H != 0 || width % config.PatchSize.W != 0)
            throw new ArgumentException($"The latent grid {frames}×{height}×{width} must tile the patch size {config.PatchSize}.");
    }

    /// <summary>The cached path has no mask-range attention; a non-zero mask range would be silently ignored, so it is refused.</summary>
    private static void RequireNoMask(in LongCatConditioning conditioning)
    {
        if (conditioning.MaskFrameRange != 0)
            throw new ArgumentException("MaskFrameRange is not implemented on the KV-cache path; use 0 (the reference default).", nameof(conditioning));
    }
}
