using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.SanaWm;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Streaming;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.World.Camera;
using HartsyInference.World.Pipelines;
using HartsyInference.World.Pipelines.SanaWm;

namespace HartsyInference.Engine.Worlds;

/// <summary>Loads SANA-WM's components from their checkpoint files, each stage in its own <see cref="WorldPhase"/>: the stage's
/// loaders, modules and converted weights are owned by the phase, and releasing the stage releases the phase, so the next stage
/// starts with the previous one's memory returned. A load that fails releases what it had acquired. The clip pipeline opens
/// one stage at a time; a stream opens the text and image encoders the same way, then holds its DiT and its causal VAE
/// decoder open together.</summary>
internal sealed class SanaWmEngineComponents : ISanaWmComponents, ISanaWmStreamingComponents
{
    private const string LatentsMeanKey = "latents_mean";
    private const string LatentsStdKey = "latents_std";

    // vae/config.json: residual upsamplers, reflect padding, non-causal; the LTX-2 defaults (neither) do not match these weights.
    private static readonly bool[] DecoderUpsampleResidual = [true, true, true];

    private readonly IBackend _backend;
    private readonly WorldComponents _files;
    private readonly WorldPhaseLedger? _ledger;

    /// <summary>Loads from <paramref name="files"/> onto <paramref name="backend"/>; every stage phase reports to
    /// <paramref name="ledger"/>.</summary>
    public SanaWmEngineComponents(IBackend backend, WorldComponents files, WorldPhaseLedger? ledger)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(files);
        bool sanaWm = files.Architecture is WorldArchitecture.SanaWm or WorldArchitecture.SanaWmStreaming;
        if (!sanaWm || files.Vae is null || files.TextEncoder is null || files.Tokenizer is null)
            throw new ArgumentException("SANA-WM needs its DiT, VAE, text encoder and tokenizer files.", nameof(files));
        _backend = backend;
        _files = files;
        _ledger = ledger;
    }

    /// <inheritdoc/>
    public SanaWmStage<ISanaWmTextEncoder> OpenTextEncoder() => Open<ISanaWmTextEncoder>("SANA-WM text encoder", phase =>
    {
        SafeTensorsLoader loader = phase.Own(Load(_files.TextEncoder!));
        Dictionary<string, Tensor> weights = loader.GetAllTensors();
        LlamaStyleEncoder encoder = phase.Own(new LlamaStyleEncoder(LlamaStyleEncoderConfig.Gemma2_2B));
        encoder.LoadWeights(weights);
        phase.OwnRelease("the Gemma weight mapping", weights.Clear);
        // Two encodes promote every projection to a resident weight, which only an explicit free returns.
        phase.OwnRelease("the Gemma device copies", () => _backend.FreeWeights(encoder.EnumerateWeights()));
        GemmaTokenizer tokenizer = phase.Own(new GemmaTokenizer(_files.Tokenizer!));
        return new SanaWmGemmaTextEncoder(_backend, encoder, tokenizer);
    });

    /// <inheritdoc/>
    public SanaWmStage<ISanaWmImageEncoder> OpenImageEncoder() => Open<ISanaWmImageEncoder>("SANA-WM VAE encoder", phase =>
    {
        SafeTensorsLoader loader = phase.Own(Load(_files.Vae!));
        (float[] mean, float[] std) = Statistics(loader);
        // The encoder computes in F32, so its BF16 checkpoint weights are widened once for the stage.
        Dictionary<string, Tensor> weights = Widened(phase, loader, "encoder.", DType.F32);
        LtxVideo2VaeEncoder encoder = new LtxVideo2VaeEncoder(new LtxVideo2VaeEncoderConfig());
        encoder.LoadWeights(weights);
        phase.OwnRelease("the VAE encoder weights", () => DisposeAll(encoder.EnumerateWeights()));
        return new SanaWmLtx2ImageEncoder(_backend, encoder, mean, std);
    });

    /// <inheritdoc/>
    public SanaWmStage<ISanaWmDenoiser> OpenDenoiser(SanaWmTextConditioning text, SanaWmCameraConditioning camera) =>
        Open<ISanaWmDenoiser>("SANA-WM DiT", phase =>
        {
            SanaWmCheckpoint checkpoint = phase.Own(SanaWmCheckpointConverter.Load(_files.Backbone));
            SanaWmDit dit = phase.Own(new SanaWmDit(_backend, checkpoint));
            return phase.Own(new SanaWmDitDenoiser(_backend, dit, text, camera));
        });

    /// <inheritdoc/>
    public SanaWmStage<ISanaWmVideoDecoder> OpenVideoDecoder() => Open<ISanaWmVideoDecoder>("SANA-WM VAE decoder", phase =>
    {
        SafeTensorsLoader loader = phase.Own(Load(_files.Vae!));
        (float[] mean, float[] std) = Statistics(loader);
        DType dtype = VaePrecisionHelper.PreferredVaeDtype(_backend);
        Dictionary<string, Tensor> weights = Widened(phase, loader, "decoder.", dtype);
        LtxVideo2VaeDecoder decoder = new LtxVideo2VaeDecoder(latentChannels: SanaWmPipeline.LatentChannels,
            upsampleResidual: DecoderUpsampleResidual, isCausal: false, latentsMean: mean, latentsStd: std, computeDtype: dtype,
            spatialReflectPad: true);
        decoder.LoadWeights(weights);
        phase.OwnRelease("the VAE decoder weights", () => DisposeAll(decoder.EnumerateWeights()));
        return new SanaWmLtx2VideoDecoder(_backend, decoder);
    });

    /// <inheritdoc/>
    public SanaWmStage<ISanaWmChunkDenoiser> OpenChunkDenoiser(SanaWmTextConditioning text, TokenWindowPolicy softmaxWindow) =>
        Open<ISanaWmChunkDenoiser>("SANA-WM streaming DiT", phase =>
        {
            SanaWmCheckpoint checkpoint = phase.Own(SanaWmCheckpointConverter.Load(_files.Backbone));
            SanaWmDit dit = phase.Own(new SanaWmDit(_backend, checkpoint));
            return phase.Own(new SanaWmDitChunkDenoiser(_backend, dit, checkpoint.Config, text, softmaxWindow));
        });

    /// <inheritdoc/>
    public SanaWmStage<IStreamingVideoDecoder> OpenStreamingDecoder() => Open<IStreamingVideoDecoder>("SANA-WM causal VAE decoder", phase =>
    {
        SafeTensorsLoader loader = phase.Own(Load(_files.Vae!));
        (float[] mean, float[] std) = Statistics(loader);
        DType dtype = VaePrecisionHelper.PreferredVaeDtype(_backend);
        Dictionary<string, Tensor> weights = Widened(phase, loader, "decoder.", dtype);
        // The causal VAE keeps the bidirectional VAE's decoder layout (residual upsamplers, reflect padding) with causal convs.
        LtxVideo2VaeDecoder decoder = new LtxVideo2VaeDecoder(latentChannels: SanaWmPipeline.LatentChannels,
            upsampleResidual: DecoderUpsampleResidual, isCausal: true, latentsMean: mean, latentsStd: std, computeDtype: dtype,
            spatialReflectPad: true);
        decoder.LoadWeights(weights);
        phase.OwnRelease("the VAE decoder weights", () => DisposeAll(decoder.EnumerateWeights()));
        return phase.Own(new LtxVideo2StreamingDecoder(_backend, decoder));
    });

    private SanaWmStage<T> Open<T>(string name, Func<WorldPhase, T> load)
        where T : class
    {
        WorldPhase phase = new WorldPhase(name, _ledger);
        bool opened = false;
        try
        {
            // Released last: freed transients stay reserved in the pool, out of reach of the next stage's weight allocations.
            phase.OwnRelease("the device pool reservations", _backend.TrimMemoryPool);
            T component = load(phase);
            DeviceMemoryTrace.Record(_backend, name, "loaded");
            opened = true;
            return new SanaWmStage<T>(component, () => Release(name, phase));
        }
        finally
        {
            if (!opened)
                phase.Release();
        }
    }

    private void Release(string name, WorldPhase phase)
    {
        try
        {
            DeviceMemoryTrace.Record(_backend, name, "done");
        }
        finally
        {
            phase.Release();
        }

        DeviceMemoryTrace.Record(_backend, name, "released");
    }

    private static SafeTensorsLoader Load(string path)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        try
        {
            loader.Load(path);
            return loader;
        }
        catch
        {
            loader.Dispose();
            throw;
        }
    }

    // The checkpoint's per-channel latent statistics, widened to F32.
    private static (float[] Mean, float[] Std) Statistics(SafeTensorsLoader loader)
    {
        if (!loader.Descriptors.ContainsKey(LatentsMeanKey) || !loader.Descriptors.ContainsKey(LatentsStdKey))
            throw new InvalidDataException($"The SANA-WM VAE '{loader.FilePath}' has no {LatentsMeanKey}/{LatentsStdKey}.");
        return (Floats(loader.GetTensor(LatentsMeanKey)), Floats(loader.GetTensor(LatentsStdKey)));
    }

    private static float[] Floats(Tensor tensor)
    {
        if (tensor.DType == DType.F32)
            return tensor.AsReadOnlySpan<float>().ToArray();
        using Tensor widened = tensor.CastTo(DType.F32);
        return widened.AsReadOnlySpan<float>().ToArray();
    }

    // The weights under prefix cast to dtype. Every cast result is owned by the phase; tensors already at dtype stay borrowed.
    private static Dictionary<string, Tensor> Widened(WorldPhase phase, SafeTensorsLoader loader, string prefix, DType dtype)
    {
        Dictionary<string, Tensor> selected = new(StringComparer.Ordinal);
        foreach (string key in loader.Descriptors.Keys)
        {
            if (key.StartsWith(prefix, StringComparison.Ordinal))
                selected[key] = loader.GetTensor(key);
        }
        Dictionary<string, Tensor> cast = VaePrecisionHelper.CastVaeWeights(selected, dtype);
        List<Tensor> owned = [];
        foreach (KeyValuePair<string, Tensor> entry in cast)
        {
            if (!ReferenceEquals(entry.Value, selected[entry.Key]))
                owned.Add(entry.Value);
        }
        phase.OwnRelease($"the cast {prefix} weights", () => DisposeAll(owned));
        return cast;
    }

    private static void DisposeAll(IEnumerable<Tensor> tensors)
    {
        foreach (Tensor tensor in tensors)
            tensor.Dispose();
    }
}
