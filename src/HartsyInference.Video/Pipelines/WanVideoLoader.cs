using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelHandler.CheckpointConverters;
using HartsyInference.ModelHandler.CheckpointConverters.Utils;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Video.Pipelines;

/// <summary>Production entry point for the Wan video family: checkpoint path(s) → ready-to-run pipeline. Auto-detects
/// the variant from tensor shapes/keys (<see cref="WanConfigDetector"/> — T2V/I2V/TI2V/VACE, fp8 CFG-renorm), picks the
/// z=16/48 VAE decoder+encoder, and wires the Wan2.2 A14B MoE low-noise expert when a second checkpoint is given.
/// Owns the mmap loaders and model components (pipelines borrow them) — dispose the loader after the pipelines are done.
///
/// <para>Exactly one of <see cref="Pipeline"/> / <see cref="VacePipeline"/> is non-null, keyed on the detected
/// architecture. Text conditioning is umT5: pass <see cref="WanVideoLoadOptions.Umt5Path"/> and call
/// <see cref="EncodePrompts"/> (loads the encoder, encodes, frees its VRAM), or encode upstream and feed embeddings
/// directly.</para></summary>
public sealed class WanVideoLoader : IDisposable
{
    private readonly List<SafeTensorsLoader> _loaders = [];
    private readonly List<IDisposable> _components = [];
    private readonly IBackend _backend;
    private readonly string? _umt5Path;
    private int _disposed;

    /// <summary>Detected + override-applied config the pipelines run on.</summary>
    public WanVideoConfig Config { get; private set; } = null!;

    /// <summary>The T2V / I2V / TI2V / MoE pipeline (null when the checkpoint is a VACE DiT).</summary>
    public WanVideoPipeline? Pipeline { get; private set; }

    /// <summary>The VACE control pipeline (null unless the checkpoint has a VACE control branch).</summary>
    public WanVacePipeline? VacePipeline { get; private set; }

    private WanVideoLoader(IBackend backend, string? umt5Path)
    {
        _backend = backend;
        _umt5Path = umt5Path;
    }

    /// <summary>Loads a Wan DiT single-file checkpoint + VAE and constructs the matching pipeline.</summary>
    /// <param name="ditPath">Single-file DiT safetensors (original-Wan or diffusers naming; fp8-scaled handled).</param>
    /// <param name="vaePath">Wan VAE safetensors (z is validated against the DiT's predicted latent width).</param>
    public static WanVideoLoader Load(IBackend backend, string ditPath, string vaePath, WanVideoLoadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        options ??= new WanVideoLoadOptions();
        WanVideoLoader loader = new(backend, options.Umt5Path);
        try
        {
            loader.Build(ditPath, vaePath, options);
            return loader;
        }
        catch (Exception ex)
        {
            Logs.Error($"Failed to load Wan video checkpoint from {ditPath}", ex);
            loader.Dispose();
            throw;
        }
    }

    private void Build(string ditPath, string vaePath, WanVideoLoadOptions options)
    {
        (WanVideoCheckpointConverter.ConvertedWeights conv, SafeTensorsLoader ditLoader) =
            WanVideoCheckpointConverter.LoadAndConvert(ditPath);
        _loaders.Add(ditLoader);

        WanVideoConfig cfg = WanConfigDetector.Detect(conv.Transformer);
        if (options.FlowShift is float shift) cfg = cfg with { FlowShift = shift };
        if (options.CfgRescale is float rescale) cfg = cfg with { CfgRescale = rescale };
        bool moe = options.LowNoiseDitPath is not null;
        if (moe) cfg = cfg with { BoundaryRatio = options.BoundaryRatio };
        Config = cfg;
        Logs.Info($"WanVideoLoader: {WanConfigDetector.Describe(cfg)}");

        (Dictionary<string, Tensor> vaeW, IReadOnlyList<SafeTensorsLoader> vaeLoaders) = LanceCheckpointConverter.LoadVae(vaePath);
        _loaders.AddRange(vaeLoaders);
        IWanVaeDecoder vaeDecoder;
        IWanVaeEncoder vaeEncoder;
        if (cfg.VaeLatentChannels >= 48)
        {
            Wan22VaeDecoder d = new(); d.LoadWeights(vaeW); vaeDecoder = d;
            Wan22VaeEncoder e = new(); e.LoadWeights(vaeW); vaeEncoder = e;
        }
        else
        {
            Wan21VaeDecoder d = new(); d.LoadWeights(vaeW); vaeDecoder = d;
            Wan21VaeEncoder e = new(); e.LoadWeights(vaeW); vaeEncoder = e;
        }
        if (vaeDecoder is IDisposable dd) _components.Add(dd);
        if (vaeEncoder is IDisposable ed) _components.Add(ed);

        if (cfg.VaceLayers.Length > 0)
        {
            WanVaceTransformer vace = new(cfg);
            vace.LoadWeights(conv.Transformer);
            _components.Add(vace);
            VacePipeline = new WanVacePipeline(_backend, vace, vaeDecoder, vaeEncoder, cfg);
            return;
        }

        WanVideoTransformer transformer = new(cfg);
        transformer.LoadWeights(conv.Transformer);
        _components.Add(transformer);

        WanVideoTransformer? transformer2 = null;
        if (moe)
        {
            (WanVideoCheckpointConverter.ConvertedWeights convLow, SafeTensorsLoader lowLoader) =
                WanVideoCheckpointConverter.LoadAndConvert(options.LowNoiseDitPath!);
            _loaders.Add(lowLoader);
            transformer2 = new WanVideoTransformer(cfg);
            transformer2.LoadWeights(convLow.Transformer);
            _components.Add(transformer2);
            Logs.Info($"WanVideoLoader: MoE low-noise expert loaded (boundary={cfg.BoundaryRatio})");
        }

        Pipeline = new WanVideoPipeline(_backend, transformer, vaeDecoder, cfg, vaeEncoder, transformer2);
    }

    /// <summary>Encodes a prompt/negative pair with umT5-XXL (path from <see cref="WanVideoLoadOptions.Umt5Path"/>).
    /// The encoder weights live on the GPU only for the duration of the call. The caller disposes both tensors.</summary>
    public (Tensor PromptEmbeds, Tensor NegativeEmbeds) EncodePrompts(string prompt, string negativePrompt)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (_umt5Path is null || !File.Exists(_umt5Path))
            throw new InvalidOperationException("Set WanVideoLoadOptions.Umt5Path to an umT5-XXL safetensors file to use EncodePrompts.");

        using SafeTensorsLoader umt5Loader = new();
        umt5Loader.Load(_umt5Path);
        Dictionary<string, Tensor> umt5W = CheckpointConvertUtils.ApplyFp8ScaledDequant(umt5Loader.GetAllTensors());
        using T5TextEncoder umt5 = new(T5TextEncoderConfig.Umt5Xxl);
        umt5.LoadWeights(umt5W);
        using T5Tokenizer tokenizer = T5Tokenizer.CreateUmt5(maxLength: 512);
        int[] promptTokens = tokenizer.Encode(prompt);
        int[] negTokens = tokenizer.Encode(negativePrompt);
        Tensor batch = umt5.Encode(_backend, [promptTokens, negTokens],
            [T5Tokenizer.CreateAttentionMask(promptTokens), T5Tokenizer.CreateAttentionMask(negTokens)]);
        try
        {
            Tensor promptEmbeds = CfgHelper.SliceBatchElement(batch, 0, promptTokens.Length, 4096);
            Tensor negEmbeds = CfgHelper.SliceBatchElement(batch, 1, promptTokens.Length, 4096);
            return (promptEmbeds, negEmbeds);
        }
        finally
        {
            batch.Dispose();
            _backend.Sync();
            _backend.FreeWeights(umt5.EnumerateWeights());
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        Pipeline?.Dispose();
        VacePipeline?.Dispose();
        foreach (IDisposable c in _components) c.Dispose();
        _components.Clear();
        foreach (SafeTensorsLoader l in _loaders) l.Dispose();
        _loaders.Clear();
    }
}
