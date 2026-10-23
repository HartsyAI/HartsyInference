using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Fibo 2.0 denoiser (ComfyUI <c>fibo2.Fibo2</c>) for one prompt, with optional reference-image latents for editing:
/// batch 1; a prompt is never padded, so there are no padded gist masks. The text arrives as one <c>[1, L, bundles · capFeatDim]</c> tensor: bundle 0 feeds the Perceiver, and
/// bundle <c>j + 1</c> feeds the j-th injection block. Latents are <c>[1, InChannels, H, W]</c> with 2x2 patchify.</summary>
public sealed unsafe class Fibo2Transformer : IDisposable
{
    private readonly Fibo2Config _config;
    private readonly Fibo2Block[] _refiners;
    private readonly Fibo2Block[] _layers;
    private readonly Fibo2PerceiverLayer[] _perceiver;
    private readonly ZImageRope _imageRope;
    private readonly ZImageRope _streamRope;
    private readonly int _patchVolume;
    private readonly List<Tensor> _ownedCasts = [];

    private Tensor? _xEmbedderWeight, _xEmbedderBias;
    private Tensor? _timeMlp1Weight, _timeMlp1Bias, _timeMlp2Weight, _timeMlp2Bias;
    private Tensor? _latentQueries, _perceiverNormOut;
    private Tensor? _finalAdaLnWeight, _finalAdaLnBias, _finalLinearWeight, _finalLinearBias;
    private int _timeMlpWidth;
    private int _disposed;

    /// <summary>Creates the denoiser for <paramref name="config"/>; call <see cref="LoadWeights"/> before <see cref="Forward"/>.</summary>
    public Fibo2Transformer(Fibo2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        config.Validate();
        _config = config;
        _patchVolume = config.InChannels * config.PatchSize * config.PatchSize;
        _refiners = new Fibo2Block[config.NumRefinerLayers];
        for (int i = 0; i < _refiners.Length; i++)
            _refiners[i] = new Fibo2Block(config, hasCrossAttention: false);
        _layers = new Fibo2Block[config.NumLayers];
        for (int i = 0; i < _layers.Length; i++)
            _layers[i] = new Fibo2Block(config, hasCrossAttention: Array.IndexOf(config.InjectionLayers, i) >= 0);
        _perceiver = new Fibo2PerceiverLayer[config.PerceiverLayers];
        for (int i = 0; i < _perceiver.Length; i++)
            _perceiver[i] = new Fibo2PerceiverLayer(config);
        _imageRope = new ZImageRope(config.AxesDims, config.RopeTheta);
        _streamRope = new ZImageRope(config.AxesDims, config.RopeTheta);
    }

    /// <summary>Loads every tensor under its ComfyUI-native name. The weights are borrowed; the split pieces are owned.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        int dim = _config.Dim;
        _xEmbedderWeight = Require(weights, "x_embedder.weight", dim, _patchVolume);
        _xEmbedderBias = Require(weights, "x_embedder.bias", dim);
        _timeMlp1Weight = Require(weights, "t_embedder.mlp.0.weight", -1, Fibo2Config.TimestepEmbeddingDim);
        _timeMlp1Bias = Require(weights, "t_embedder.mlp.0.bias");
        _timeMlp2Weight = Require(weights, "t_embedder.mlp.2.weight", Fibo2Config.TimestepEmbeddingDim, -1);
        _timeMlp2Bias = Require(weights, "t_embedder.mlp.2.bias", Fibo2Config.TimestepEmbeddingDim);
        _timeMlpWidth = (int)_timeMlp1Weight.Shape[0];
        // Both are read through raw float pointers (the queries in PerceiverForward), so a half-precision checkpoint is copied to F32.
        _latentQueries = EnsureOwnedF32(Require(weights, "perceiver.latent_queries", _config.PerceiverQueries, dim));
        _perceiverNormOut = EnsureOwnedF32(weights["perceiver.norm_out.weight"]);
        _finalAdaLnWeight = Require(weights, "final_layer.adaLN_modulation.1.weight", dim, Fibo2Config.TimestepEmbeddingDim);
        _finalAdaLnBias = Require(weights, "final_layer.adaLN_modulation.1.bias", dim);
        _finalLinearWeight = Require(weights, "final_layer.linear.weight", _patchVolume, dim);
        _finalLinearBias = Require(weights, "final_layer.linear.bias", _patchVolume);

        for (int i = 0; i < _refiners.Length; i++)
            _refiners[i].LoadWeights(weights, $"noise_refiner.{i}");
        for (int i = 0; i < _layers.Length; i++)
            _layers[i].LoadWeights(weights, $"layers.{i}");
        for (int i = 0; i < _perceiver.Length; i++)
            _perceiver[i].LoadWeights(weights, $"perceiver.layers.{i}");
    }

    /// <summary>Predicts the velocity for <paramref name="latent"/> <c>[1, C, H, W]</c> at <paramref name="sigma"/> (1 is pure
    /// noise) given the text bundles <c>[1, L, bundles · capFeatDim]</c> in F32. Returns a new <c>[1, C, H, W]</c> tensor.</summary>
    /// <param name="referenceLatents">Edit conditioning: the source and reference images as normalised FLUX.2 latents
    /// <c>[1, C, h, w]</c> (any size that is a multiple of the patch), in picture order. Their tokens follow the image tokens,
    /// are modulated as clean tokens (the timestep-0 embedding) and sit on RoPE frames <c>MaxGist + 2 + k</c>; the noise
    /// refiners see them too. The velocity is read from the image tokens only. Null or empty is plain text-to-image.</param>
    /// <param name="textMask">Must be null. ComfyUI's Fibo 2.0 has no gist mask: it runs one prompt at its true length (the
    /// pipeline encodes unpadded), so nothing is padded.</param>
    public Tensor Forward(IBackend backend, Tensor latent, float sigma, Tensor context, IReadOnlyList<Tensor>? referenceLatents = null,
        Tensor? textMask = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latent);
        ArgumentNullException.ThrowIfNull(context);
        if (textMask is not null)
            throw new NotSupportedException("Fibo 2.0 has no padded gist mask: prompts are encoded at their true length, so a mask would be dropped.");
        if (latent.Shape.Rank != 4 || latent.Shape[0] != 1 || latent.Shape[1] != _config.InChannels)
            throw new ArgumentException($"Fibo2 latent must be [1, {_config.InChannels}, H, W], got {latent.Shape}.", nameof(latent));
        int patch = _config.PatchSize;
        int height = (int)latent.Shape[2];
        int width = (int)latent.Shape[3];
        if (height == 0 || width == 0 || height % patch != 0 || width % patch != 0)
            throw new ArgumentException($"Fibo2 latent {height}x{width} must be a positive multiple of the patch size {patch}.", nameof(latent));
        List<(int Height, int Width)> referenceGrids = ValidateReferences(referenceLatents, patch);
        int bundleCount = 1 + _config.InjectionLayers.Length;
        if (context.Shape.Rank != 3 || context.Shape[0] != 1 || context.Shape[1] < 1
            || context.Shape[2] != (long)bundleCount * _config.CapFeatDim || context.DType != DType.F32)
            throw new ArgumentException($"Fibo2 context must be F32 [1, L, {bundleCount * _config.CapFeatDim}], got {context.Shape} {context.DType}.",
                nameof(context));

        int gridH = height / patch;
        int gridW = width / patch;
        int imageLen = gridH * gridW;
        int gistCount = _config.GistCount((int)context.Shape[1]);

        Tensor cleanEmbedding = TimestepEmbedding(backend, 0f);
        Tensor noisyEmbedding = TimestepEmbedding(backend, sigma * _config.TimestepScale);
        Tensor[] bundles = SplitBundles(context, bundleCount);
        try
        {
            Tensor image = EmbedLatent(backend, latent);
            int referenceLen = 0;
            if (referenceLatents is { Count: > 0 })
            {
                // ComfyUI embeds the image and reference tokens as one sequence; the embedder is per token, so each is embedded alone.
                foreach (Tensor reference in referenceLatents)
                {
                    Tensor embedded = EmbedLatent(backend, reference);
                    referenceLen += (int)embedded.Shape[1];
                    Tensor joined = DiTUtils.ConcatAlongSeqDim(image, embedded);
                    image.Dispose();
                    embedded.Dispose();
                    image = joined;
                }
            }

            image = RunRefiners(backend, image, cleanEmbedding, noisyEmbedding, gridH, gridW, referenceGrids);
            Tensor gist = PerceiverForward(backend, bundles[0], gistCount);

            using (Tensor streamIds = referenceGrids.Count == 0
                ? Fibo2PositionIds.Combined(_config, gistCount, gridH, gridW)
                : Fibo2PositionIds.CombinedWithReferences(_config, gistCount, gridH, gridW, referenceGrids))
                _streamRope.Precompute(streamIds);
            Tensor stream = DiTUtils.ConcatAlongSeqDim(gist, image);
            gist.Dispose();
            image.Dispose();
            stream = RunLayers(backend, stream, cleanEmbedding, noisyEmbedding, gistCount, imageLen, referenceGrids, bundles);

            (Tensor gistTokens, Tensor imageTokens) = DiTUtils.SplitAlongSeqDim(stream, gistCount);
            stream.Dispose();
            gistTokens.Dispose();
            if (referenceLen > 0)
            {
                // Only the image tokens are projected; the reference tokens are context.
                (Tensor imageOnly, Tensor referenceTokens) = DiTUtils.SplitAlongSeqDim(imageTokens, imageLen);
                imageTokens.Dispose();
                referenceTokens.Dispose();
                imageTokens = imageOnly;
            }
            Tensor projected = FinalLayer(backend, imageTokens, noisyEmbedding, imageLen);
            imageTokens.Dispose();

            Tensor unpacked = Flux2LatentOps.Unpack(projected, gridH, gridW);
            projected.Dispose();
            Tensor velocity = Flux2LatentOps.Unpatchify(unpacked, _config.InChannels, patch);
            unpacked.Dispose();
            return velocity;
        }
        finally
        {
            cleanEmbedding.Dispose();
            noisyEmbedding.Dispose();
            foreach (Tensor bundle in bundles)
                bundle.Dispose();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Fibo2Block block in _refiners)
            block.Dispose();
        foreach (Fibo2Block block in _layers)
            block.Dispose();
        foreach (Fibo2PerceiverLayer layer in _perceiver)
            layer.Dispose();
        foreach (Tensor cast in _ownedCasts)
            cast.Dispose();
    }

    /// <summary>F32 form of a borrowed weight. A half-precision source is copied, and the copy is kept until <see cref="Dispose"/>.</summary>
    private Tensor EnsureOwnedF32(Tensor stored)
    {
        Tensor f32 = TensorCasts.EnsureF32(stored);
        if (!ReferenceEquals(f32, stored))
            _ownedCasts.Add(f32);
        return f32;
    }

    /// <summary>Embeds a latent <c>[1, C, H, W]</c> into <c>[1, (H/p)·(W/p), dim]</c> tokens in row-major order.</summary>
    private Tensor EmbedLatent(IBackend backend, Tensor latent)
    {
        int tokens = (int)(latent.Shape[2] / _config.PatchSize * (latent.Shape[3] / _config.PatchSize));
        Tensor patched = Flux2LatentOps.Patchify(latent, _patchVolume, _config.PatchSize);
        Tensor packed = Flux2LatentOps.Pack(patched);
        patched.Dispose();
        Tensor embedded = new(new TensorShape(1, tokens, _config.Dim), DType.F32);
        backend.Linear(embedded, packed, _xEmbedderWeight!, _xEmbedderBias);
        packed.Dispose();
        return embedded;
    }

    /// <summary>Checks the reference latents and returns their token grids <c>(h/p, w/p)</c>; empty without references.</summary>
    private List<(int Height, int Width)> ValidateReferences(IReadOnlyList<Tensor>? referenceLatents, int patch)
    {
        List<(int Height, int Width)> grids = [];
        if (referenceLatents is null)
            return grids;
        for (int k = 0; k < referenceLatents.Count; k++)
        {
            Tensor reference = referenceLatents[k] ?? throw new ArgumentException($"Fibo2 reference latent {k} is null.", nameof(referenceLatents));
            if (reference.Shape.Rank != 4 || reference.Shape[0] != 1 || reference.Shape[1] != _config.InChannels
                || reference.Shape[2] <= 0 || reference.Shape[3] <= 0 || reference.Shape[2] % patch != 0 || reference.Shape[3] % patch != 0
                || reference.DType != DType.F32)
            {
                throw new ArgumentException($"Fibo2 reference latent {k} must be F32 [1, {_config.InChannels}, h, w] with h and w multiples of "
                    + $"{patch}, got {reference.DType} {reference.Shape}.", nameof(referenceLatents));
            }
            grids.Add(((int)reference.Shape[2] / patch, (int)reference.Shape[3] / patch));
        }
        return grids;
    }

    /// <summary>The noise refiners over the image tokens, then any reference tokens. Without references this is global modulation
    /// from the noisy embedding; with them the image tokens take the noisy embedding and the reference tokens the clean one.</summary>
    private Tensor RunRefiners(IBackend backend, Tensor tokens, Tensor cleanEmbedding, Tensor noisyEmbedding, int gridH, int gridW,
        List<(int Height, int Width)> referenceGrids)
    {
        using Tensor ids = referenceGrids.Count == 0
            ? Fibo2PositionIds.Image(_config, gridH, gridW)
            : Fibo2PositionIds.ImageWithReferences(_config, gridH, gridW, referenceGrids);
        _imageRope.Precompute(ids);
        int segments = 1 + referenceGrids.Count;
        int[] lengths = new int[segments];
        int[] groups = new int[segments];
        lengths[0] = gridH * gridW;
        groups[0] = referenceGrids.Count == 0 ? 0 : 1;
        for (int k = 0; k < referenceGrids.Count; k++)
            lengths[1 + k] = referenceGrids[k].Height * referenceGrids[k].Width;
        using TokenSegmentLayout layout = new(lengths, groups);
        layout.Pin(backend);
        IReadOnlyList<Tensor> embeddings = referenceGrids.Count == 0 ? [noisyEmbedding] : [cleanEmbedding, noisyEmbedding];
        Tensor current = tokens;
        foreach (Fibo2Block refiner in _refiners)
        {
            Tensor next = refiner.Forward(backend, current, embeddings, layout, _imageRope, null);
            current.Dispose();
            current = next;
        }
        return current;
    }

    private Tensor RunLayers(IBackend backend, Tensor stream, Tensor cleanEmbedding, Tensor noisyEmbedding, int gistCount,
        int imageLen, List<(int Height, int Width)> referenceGrids, Tensor[] bundles)
    {
        // Gist and reference tokens are clean (group 0); only the image tokens carry the noise level (group 1).
        int segments = 2 + referenceGrids.Count;
        int[] lengths = new int[segments];
        int[] groups = new int[segments];
        lengths[0] = gistCount;
        lengths[1] = imageLen;
        groups[1] = 1;
        for (int k = 0; k < referenceGrids.Count; k++)
            lengths[2 + k] = referenceGrids[k].Height * referenceGrids[k].Width;
        using TokenSegmentLayout layout = new(lengths, groups);
        layout.Pin(backend);
        IReadOnlyList<Tensor> embeddings = [cleanEmbedding, noisyEmbedding];
        Tensor current = stream;
        for (int i = 0; i < _layers.Length; i++)
        {
            int injection = Array.IndexOf(_config.InjectionLayers, i);
            Tensor? text = injection >= 0 ? bundles[injection + 1] : null;
            Tensor next = _layers[i].Forward(backend, current, embeddings, layout, _streamRope, text);
            current.Dispose();
            current = next;
        }
        return current;
    }

    private Tensor PerceiverForward(IBackend backend, Tensor bundle, int count)
    {
        int dim = _config.Dim;
        Tensor queries = new(new TensorShape(1, count, dim), DType.F32);
        new ReadOnlySpan<float>((void*)_latentQueries!.DataPointer, count * dim)
            .CopyTo(new Span<float>((void*)queries.DataPointer, count * dim));

        Tensor current = queries;
        foreach (Fibo2PerceiverLayer layer in _perceiver)
        {
            Tensor next = layer.Forward(backend, current, bundle);
            current.Dispose();
            current = next;
        }
        Tensor normed = new(current.Shape, DType.F32);
        backend.RmsNorm(normed, current, _perceiverNormOut!, _config.NormEps);
        current.Dispose();
        return normed;
    }

    private Tensor TimestepEmbedding(IBackend backend, float timestep)
    {
        Tensor sinusoid = new(new TensorShape(1, Fibo2Config.TimestepEmbeddingDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoid, timestep, 1, Fibo2Config.TimestepEmbeddingDim);
        Tensor hidden = new(new TensorShape(1, _timeMlpWidth), DType.F32);
        backend.Linear(hidden, sinusoid, _timeMlp1Weight!, _timeMlp1Bias);
        sinusoid.Dispose();
        Tensor activated = new(hidden.Shape, DType.F32);
        backend.Silu(activated, hidden);
        hidden.Dispose();
        Tensor embedding = new(new TensorShape(1, Fibo2Config.TimestepEmbeddingDim), DType.F32);
        backend.Linear(embedding, activated, _timeMlp2Weight!, _timeMlp2Bias);
        activated.Dispose();
        return embedding;
    }

    /// <summary>Scale-only adaLN from the noisy embedding (SiLU first), LayerNorm without affine, then the output projection.</summary>
    private Tensor FinalLayer(IBackend backend, Tensor imageTokens, Tensor noisyEmbedding, int imageLen)
    {
        int dim = _config.Dim;
        Tensor activated = new(noisyEmbedding.Shape, DType.F32);
        backend.Silu(activated, noisyEmbedding);
        Tensor scale = new(new TensorShape(1, dim), DType.F32);
        backend.Linear(scale, activated, _finalAdaLnWeight!, _finalAdaLnBias);
        activated.Dispose();

        TensorShape seqShape = new(1, imageLen, dim);
        Tensor normed = new(seqShape, DType.F32);
        backend.LayerNormNoAffine(normed, imageTokens, Fibo2Config.FinalNormEps);
        Tensor modulated = DiTUtils.Modulate(backend, normed, null, scale, seqShape);
        normed.Dispose();
        scale.Dispose();

        Tensor projected = new(new TensorShape(1, imageLen, _patchVolume), DType.F32);
        backend.Linear(projected, modulated, _finalLinearWeight!, _finalLinearBias);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Splits <c>[1, L, count · capFeatDim]</c> along its last dimension into <c>count</c> owned <c>[1, L, capFeatDim]</c>
    /// bundles, in order.</summary>
    private Tensor[] SplitBundles(Tensor context, int count)
    {
        int textLen = (int)context.Shape[1];
        int cap = _config.CapFeatDim;
        int rowWidth = count * cap;
        Tensor[] bundles = new Tensor[count];
        float* src = (float*)context.DataPointer;
        for (int b = 0; b < count; b++)
        {
            bundles[b] = new Tensor(new TensorShape(1, textLen, cap), DType.F32);
            float* dst = (float*)bundles[b].DataPointer;
            for (int row = 0; row < textLen; row++)
                new ReadOnlySpan<float>(src + row * rowWidth + b * cap, cap).CopyTo(new Span<float>(dst + row * cap, cap));
        }
        return bundles;
    }

    private static Tensor Require(IReadOnlyDictionary<string, Tensor> weights, string key, long rows = -1, long cols = -1)
    {
        Tensor tensor = weights.TryGetValue(key, out Tensor? found) ? found : throw new KeyNotFoundException($"Fibo2 checkpoint is missing '{key}'.");
        bool rowsOk = rows < 0 || tensor.Shape[0] == rows;
        bool colsOk = cols < 0 || (tensor.Shape.Rank == 2 && tensor.Shape[1] == cols);
        if (!rowsOk || !colsOk)
            throw new InvalidDataException($"Fibo2 '{key}' has shape {tensor.Shape}, expected rows {rows} and cols {cols} (-1 = any).");
        return tensor;
    }
}
