using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers;
using SharpInference.Diffusion.Models.TextEncoders;
using SharpInference.Diffusion.Models.Vae;

namespace SharpInference.Diffusion.Pipelines;

/// <summary>Owns a fully-wired <see cref="LensPipeline"/> together with the heavy components and
/// factory-allocated resources behind it (transformer, GPT-OSS encoder, Flux.2 VAE decoder, the
/// extracted BN mean/var copies, and any on-disk loaders the factory opened). Produced by
/// <see cref="LensPipelineFactory"/>. <see cref="DiffusionPipelineBase"/> deliberately does NOT own its
/// components — this bundle fills that gap so a caller has a single handle to dispose once it's done
/// generating. Disposing the bundle releases the pipeline, frees the transformer/encoder, disposes the
/// BN copies the factory allocated, and closes any owned loaders (releasing their memory-mapped views).</summary>
public sealed class LensPipelineBundle : IDisposable
{
    private int _disposed;

    /// <summary>The wired pipeline. Use <see cref="LensPipeline.GenerateFromTokens"/> (requires an
    /// attached encoder) or <see cref="LensPipeline.GenerateFromEmbeddings"/>.</summary>
    public LensPipeline Pipeline { get; }

    /// <summary>The Lens DiT backbone.</summary>
    public LensTransformer Transformer { get; }

    /// <summary>The GPT-OSS MoE text encoder, or <c>null</c> when the bundle was built embeddings-only.</summary>
    public LensGptOssEncoder? TextEncoder { get; }

    /// <summary>The Flux.2 semantic VAE decoder (not <see cref="IDisposable"/>; freed with the bundle's GC).</summary>
    public VaeDecoder VaeDecoder { get; }

    /// <summary>The patchified-latent BatchNorm <c>running_mean</c>, cast to an owned F32 <c>[128]</c>.</summary>
    public Tensor BnMean { get; }

    /// <summary>The patchified-latent BatchNorm <c>running_var</c>, cast to an owned F32 <c>[128]</c>.</summary>
    public Tensor BnVar { get; }

    private readonly IDisposable[] _ownedLoaders;

    internal LensPipelineBundle(LensPipeline pipeline, LensTransformer transformer,
        LensGptOssEncoder? textEncoder, VaeDecoder vaeDecoder, Tensor bnMean, Tensor bnVar,
        IDisposable[] ownedLoaders)
    {
        Pipeline = pipeline;
        Transformer = transformer;
        TextEncoder = textEncoder;
        VaeDecoder = vaeDecoder;
        BnMean = bnMean;
        BnVar = bnVar;
        _ownedLoaders = ownedLoaders;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        Pipeline.Dispose();
        Transformer.Dispose();
        TextEncoder?.Dispose();
        BnMean.Dispose();
        BnVar.Dispose();
        for (int i = 0; i < _ownedLoaders.Length; i++)
            _ownedLoaders[i].Dispose();
    }
}
