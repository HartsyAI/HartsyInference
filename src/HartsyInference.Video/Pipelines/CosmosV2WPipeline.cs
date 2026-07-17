using System.Runtime.CompilerServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Requests;
using HartsyInference.LLM.Sampling;
using HartsyInference.LLM.Transformer;
using HartsyInference.Video.Encoding;
using HartsyInference.Video.Models.Cosmos;
using HartsyInference.Video.Tokenizers;

namespace HartsyInference.Video.Pipelines;

/// <summary>Cosmos-Predict1 Video2World autoregressive pipeline: T5-11B cross-attention context (encoded by the
/// caller) + a discrete-token video prefix → prefill → per-token KV-cached AR sampling (halts at
/// <c>num_token_video_latent</c>; no EOS) → detokenize → DV-decoder render → RGB frames. Mirrors the
/// <see cref="LtxVideoPipeline"/> / <c>WanVideoPipeline</c> structure (pre-encoded conditioning in, streamed
/// <see cref="VideoFrame"/> out) but the loop is next-token AR rather than diffusion denoising.
///
/// <para>First ship is the <b>DV-decoder-only</b> path (fastest to e2e, ~7 GB less VRAM than the optional 7B
/// diffusion decoder — deferred). The pipeline does not own its components (transformer / tokenizer are shared,
/// passed in). <b>No guardrails</b> are applied — see the Cosmos NOTICE; the user is responsible for content
/// review.</para></summary>
public sealed class CosmosV2WPipeline : IDisposable
{
    private readonly IBackend _backend;
    private readonly CosmosArTransformer _transformer;
    private readonly CosmosDvTokenizer _tokenizer;
    private int _disposed;

    /// <summary>Creates the pipeline over a loaded backbone + DV tokenizer (both caller-owned).</summary>
    public CosmosV2WPipeline(IBackend backend, CosmosArTransformer transformer, CosmosDvTokenizer tokenizer)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    }

    /// <summary>Autoregressively samples the full <c>T·H·W</c> token sequence for the latent grid, starting from
    /// <paramref name="prefixTokens"/> (the tokenized image/video conditioning, 1 or 2 latent frames' worth) and
    /// cross-attending <paramref name="context"/> <c>[1, ctxLen, context_dim]</c>. Halts when the sequence reaches
    /// <c>N = T·H·W</c> (no EOS). Returns all <c>N</c> token ids. Defaults: temperature 0.6, top-p 0.9.</summary>
    public int[] GenerateTokens(Tensor? context, int ctxLen, ReadOnlySpan<int> prefixTokens,
        int latentT, int latentH, int latentW, SamplingOptions? sampling = null, Action<GenerationProgress>? progress = null)
    {
        ThrowIfDisposed();
        int n = latentT * latentH * latentW;
        int prefixLen = prefixTokens.Length;
        if (prefixLen < 1) throw new ArgumentException("V2W requires at least one prefix (conditioning) token.", nameof(prefixTokens));
        if (prefixLen >= n) throw new ArgumentException($"prefix length {prefixLen} must be < sequence length {n}.", nameof(prefixTokens));

        sampling ??= new SamplingOptions { Temperature = 0.6f, TopP = 0.9f };
        SamplerChain sampler = SamplerChain.FromOptions(sampling);
        int vocab = _transformer.Config.VocabSize;

        _transformer.SetGrid(latentT, latentH, latentW);
        using FixedKvCache cache = new(_transformer.Config.NumLayers, 1, _transformer.Config.NumKvHeads,
            _transformer.Config.HeadDim, n);

        List<int> tokens = new(n);
        for (int i = 0; i < prefixLen; i++) tokens.Add(prefixTokens[i]);

        long start = System.Diagnostics.Stopwatch.GetTimestamp();

        // Prefill: the last prefix position's logits predict the first generated token.
        int firstNext;
        using (Tensor hidden = _transformer.Forward(_backend, prefixTokens, 0, cache, context, ctxLen))
        using (Tensor logits = _transformer.ProjectLogits(_backend, hidden, prefixLen))
            firstNext = sampler.Next(LastRow(logits, prefixLen, vocab), tokens);
        tokens.Add(firstNext);

        // Per-token decode: feed the last token at its own position, sample the next.
        Span<int> one = stackalloc int[1];
        while (tokens.Count < n)
        {
            one[0] = tokens[^1];
            int posStart = cache.CurrentLength;   // == tokens.Count - 1
            int next;
            using (Tensor hidden = _transformer.Forward(_backend, one, posStart, cache, context, ctxLen))
            using (Tensor logits = _transformer.ProjectLogits(_backend, hidden, 1))
                next = sampler.Next(LastRow(logits, 1, vocab), tokens);
            tokens.Add(next);
            if (progress is not null && (tokens.Count % 256 == 0 || tokens.Count == n))
            {
                double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - start) * 1000.0 / System.Diagnostics.Stopwatch.Frequency;
                progress(new GenerationProgress(tokens.Count - prefixLen, n - prefixLen, ms));
            }
        }
        return tokens.ToArray();
    }

    /// <summary>Renders sampled token ids <c>[N]</c> to an RGB video tensor <c>[1, 3, T_pix, H_pix, W_pix]</c> in
    /// <c>[-1, 1]</c> via the DV tokenizer's decoder path.</summary>
    public Tensor DecodeTokens(ReadOnlySpan<int> tokens, int latentT, int latentH, int latentW)
    {
        ThrowIfDisposed();
        int n = latentT * latentH * latentW;
        if (tokens.Length != n) throw new ArgumentException($"expected {n} tokens for [{latentT},{latentH},{latentW}], got {tokens.Length}.");
        using Tensor codes = new Tensor(new TensorShape(1, n), DType.I32);
        unsafe
        {
            int* cp = (int*)codes.DataPointer;
            for (int i = 0; i < n; i++) cp[i] = tokens[i];
        }
        return _tokenizer.Decode(_backend, codes, latentT, latentH, latentW);
    }

    /// <summary>End-to-end: samples tokens then streams the decoded RGB frames (pull-based → memory bounded; pair
    /// with an <see cref="IVideoEncoder"/>). <paramref name="pixelFrames"/>/<paramref name="height"/>/
    /// <paramref name="width"/> are the output clip dimensions (default 33×640×1024).</summary>
    public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(Tensor? context, int ctxLen,
        int[] prefixTokens, int pixelFrames = 33, int height = 640, int width = 1024,
        SamplingOptions? sampling = null, Action<GenerationProgress>? progress = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ThrowIfDisposed();
        (int lt, int lh, int lw) = _tokenizer.LatentGrid(pixelFrames, height, width);
        int[] tokens = GenerateTokens(context, ctxLen, prefixTokens, lt, lh, lw, sampling, progress);

        using Tensor rgb = DecodeTokens(tokens, lt, lh, lw);
        int tPix = (int)rgb.Shape[2], hPix = (int)rgb.Shape[3], wPix = (int)rgb.Shape[4];
        for (int f = 0; f < tPix; f++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return new VideoFrame(f, wPix, hPix, FrameToRgb(rgb, f, hPix, wPix));
            await Task.Yield();
        }
    }

    private static unsafe byte[] FrameToRgb(Tensor video, int frame, int h, int w)
    {
        // video is [1, 3, T, H, W] in [-1, 1]; emit interleaved RGB24 for the frame.
        int t = (int)video.Shape[2];
        long frameStride = (long)h * w;
        long channelStride = (long)t * frameStride;
        float* vp = (float*)video.DataPointer;
        byte[] rgb = new byte[frameStride * 3];
        for (int c = 0; c < 3; c++)
        {
            float* plane = vp + c * channelStride + (long)frame * frameStride;
            for (long i = 0; i < frameStride; i++)
            {
                float v = (plane[i] + 1f) * 127.5f;
                rgb[i * 3 + c] = (byte)Math.Clamp((int)MathF.Round(v), 0, 255);
            }
        }
        return rgb;
    }

    private static unsafe Span<float> LastRow(Tensor logits, int t, int vocab)
    {
        float* p = (float*)logits.DataPointer;
        return new Span<float>(p + (long)(t - 1) * vocab, vocab);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(CosmosV2WPipeline));
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
    }
}
