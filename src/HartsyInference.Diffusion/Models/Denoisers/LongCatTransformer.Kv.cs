using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The KV cache of one conditioning pass: one <see cref="LongCatBlockKv"/> per block (the reference's <c>kv_cache_dict</c>), over
/// <see cref="Tokens"/> conditioning rows. The holder disposes it.</summary>
public sealed class LongCatKvCache : IDisposable
{
    private bool _disposed;

    internal LongCatKvCache(LongCatBlockKv[] blocks, int tokens)
    {
        Blocks = blocks;
        Tokens = tokens;
    }

    /// <summary>The per-block keys and values, in block order.</summary>
    public IReadOnlyList<LongCatBlockKv> Blocks { get; }

    /// <summary>The cached rows per block: the conditioning frames times the tokens per frame.</summary>
    public int Tokens { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        foreach (LongCatBlockKv block in Blocks)
            block.Dispose();
    }
}

/// <summary>The KV-cache path of the LongCat DiT (the reference's <c>return_kv</c> pass and <c>kv_cache_dict</c> denoising). The cache pass takes
/// the conditioning latents (timesteps zero) through self-attention and the FFN, with no text or audio, as the pipeline's
/// <c>_cache_clean_latents</c> does. The denoising pass takes only the noisy latents: their rope rows follow the cache, their attention covers
/// the cache and themselves, and their text and audio steps cover every row.</summary>
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
        int dim = _config.HiddenSize;
        int gridH = height / _config.PatchSize.H, gridW = width / _config.PatchSize.W, hw = gridH * gridW;

        using Tensor timeEmbed = TimestepEmbedding(backend, new float[] { 0f }, frames);
        Tensor hidden = WanDitOps.Patchify(backend, condLatent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchBias);
        (Tensor cos, Tensor sin) = RopeFor(frames, gridH, gridW, conditioning);
        AttentionSegment[] segments = conditioning.AttentionPlan(hw, frames);
        LongCatBlockInputs inputs = new(frames, hw, conditioning.NumCondLatents, cos, sin, segments, FrameRowIndex(frames, hw));

        LongCatBlockKv?[] blocks = new LongCatBlockKv?[_blocks.Length];
        Tensor cur = hidden;
        try
        {
            for (int i = 0; i < _blocks.Length; i++)
            {
                Tensor next = _blocks[i].ForwardCacheBuild(backend, cur, timeEmbed, in inputs, out LongCatBlockKv? kv);
                cur.Dispose();
                cur = next;
                blocks[i] = kv ?? throw new InvalidOperationException($"Block {i} produced no cache.");
            }
        }
        catch
        {
            cur.Dispose();
            foreach (LongCatBlockKv? kv in blocks)
                kv?.Dispose();
            throw;
        }
        cur.Dispose();
        return new LongCatKvCache(blocks.Select(static b => b!).ToArray(), frames * hw);
    }

    /// <summary>The denoising pass over the noisy latents <c>[1, C, N, H, W]</c> with <paramref name="cache"/> in front of them. The conditioning
    /// frames are not in <paramref name="noisyLatent"/>: <paramref name="conditioning"/> still says how many there were, and the rope rows
    /// start after the cache. <paramref name="audioTokens"/> are the noisy frames' audio tokens (the last <c>N</c> frames of the projection).</summary>
    public Tensor ForwardCached(IBackend backend, Tensor noisyLatent, Tensor encoder, ReadOnlySpan<float> frameTimesteps,
        in LongCatConditioning conditioning, LongCatKvCache cache, int validTextTokens = -1, Tensor? audioTokens = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(noisyLatent);
        ArgumentNullException.ThrowIfNull(cache);
        if (!_loaded)
            throw new InvalidOperationException("The transformer has no weights: LoadWeights was never called, or it failed.");
        if (cache.Blocks.Count != _blocks.Length)
            throw new ArgumentException($"The cache has {cache.Blocks.Count} blocks; the transformer has {_blocks.Length}.", nameof(cache));
        if (noisyLatent.Shape.Rank != 5 || noisyLatent.Shape[0] != 1 || noisyLatent.Shape[1] != _config.InChannels)
            throw new ArgumentException($"The noisy latent must be [1, {_config.InChannels}, N, H, W], got {noisyLatent.Shape}.", nameof(noisyLatent));
        if (_audioProjection is not null && audioTokens is null)
            throw new ArgumentException("This avatar checkpoint needs the noisy frames' audio tokens.", nameof(audioTokens));
        if (_audioProjection is null && audioTokens is not null)
            throw new ArgumentException("This checkpoint has no audio projection, so audio tokens do not apply.", nameof(audioTokens));

        int frames = (int)noisyLatent.Shape[2], height = (int)noisyLatent.Shape[3], width = (int)noisyLatent.Shape[4];
        int dim = _config.HiddenSize, gridH = height / _config.PatchSize.H, gridW = width / _config.PatchSize.W, hw = gridH * gridW;
        if (cache.Tokens != conditioning.NumCondLatents * hw)
            throw new ArgumentException($"The cache holds {cache.Tokens} rows; the conditioning needs {conditioning.NumCondLatents * hw}.", nameof(cache));
        int rows = frames * hw;

        // The rope rows of the noisy frames: the full table covers the conditioning frames then the noisy ones; the noisy part starts at the cache.
        (Tensor fullCos, Tensor fullSin) = RopeFor(conditioning.NumCondLatents + frames, gridH, gridW, conditioning);
        using Tensor cos = new(new TensorShape(rows, fullCos.Shape[1]), DType.F32);
        using Tensor sin = new(new TensorShape(rows, fullSin.Shape[1]), DType.F32);
        backend.SliceRows(cos, fullCos, cache.Tokens);
        backend.SliceRows(sin, fullSin, cache.Tokens);

        using Tensor timeEmbed = TimestepEmbedding(backend, frameTimesteps, frames);
        using Tensor text = CaptionEmbedding(backend, encoder, validTextTokens);
        Tensor cur = WanDitOps.Patchify(backend, noisyLatent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchBias);
        AttentionSegment[] segmentsHolder = [AttentionSegment.Dense(rows)];
        LongCatBlockInputs inputs = new(frames, hw, 0, cos, sin, segmentsHolder, FrameRowIndex(frames, hw));
        for (int i = 0; i < _blocks.Length; i++)
        {
            BeforeBlockForward?.Invoke(i);
            Tensor next = _blocks[i].Forward(backend, cur, text, timeEmbed, in inputs, Probe, Probe is null ? "" : $"block{i}", audioTokens,
                cache.Blocks[i]);
            cur.Dispose();
            cur = next;
        }

        using Tensor finalModulation = LongCatOps.AdaLnProjection(backend, timeEmbed, _finalAdaW!, _finalAdaB);
        using Tensor modulated = LongCatOps.NormModulate(backend, cur, finalModulation, dim, 0, hw);
        cur.Dispose();
        using Tensor projected = new(new TensorShape((long)rows, _finalW!.Shape[0]), DType.F32);
        backend.Linear(projected, modulated, _finalW!, _finalB);
        Probe?.Invoke("final_layer", projected);
        return WanDitOps.Unpatchify(projected, _config.OutChannels, frames / _config.PatchSize.T, gridH, gridW, _config.PatchSize);
    }

    /// <summary>The rope tables of the first <paramref name="totalFrames"/> frames under <paramref name="conditioning"/>'s frame indices, cached.</summary>
    private (Tensor Cos, Tensor Sin) RopeFor(int totalFrames, int gridH, int gridW, in LongCatConditioning conditioning)
    {
        int roperef = conditioning.NumRefLatents > 0 ? conditioning.RefImgIndex + 1 : 0;
        LongCatConditioning captured = conditioning;
        return _caches.RopeCosSin((totalFrames, gridH, gridW, roperef), () => BuildRope(totalFrames, gridH, gridW, captured));
    }
}
