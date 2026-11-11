using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Vision.Dinov2;
using HartsyInference.Vision.Upscale;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>Builds a <see cref="HunyuanPaintPipeline"/> from a local folder in the diffusers layout of the released checkpoints:
/// <c>unet/diffusion_pytorch_model.{safetensors,bin}</c> and <c>vae/diffusion_pytorch_model.{safetensors,bin}</c>. The generation is read from the UNet's tensors
/// (<c>learned_text_clip_albedo</c> means 2.1, a <c>class_embedding</c> means 2.0). Pickle files go through the safe-subset parser only. Dispose the result
/// to release the memory-mapped files.</summary>
public sealed class HunyuanPaintLoader : IDisposable
{
    private readonly List<IDisposable> _owners = [];

    private HunyuanPaintLoader(HunyuanPaintPipeline pipeline, HunyuanPaintConfig config)
    {
        Pipeline = pipeline;
        Config = config;
    }

    /// <summary>The ready pipeline.</summary>
    public HunyuanPaintPipeline Pipeline { get; }

    /// <summary>The configuration read from the checkpoint.</summary>
    public HunyuanPaintConfig Config { get; }

    /// <summary>Loads the pipeline. <paramref name="dinoPath"/> (a <c>model.safetensors</c> of <c>facebook/dinov2-giant</c> or its folder) is required for 2.1;
    /// <paramref name="upscalerPath"/> (Real-ESRGAN x4plus) is optional and used for 2.1 only.</summary>
    public static HunyuanPaintLoader Load(IBackend backend, string folder, string? dinoPath = null, string? upscalerPath = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        List<IDisposable> owners = [];
        try
        {
            Dictionary<string, Tensor> unetWeights = Read(Find(folder, "unet"), owners);
            HunyuanPaintConfig config = Detect(unetWeights);
            HunyuanPaintUnet unet = new(config, generating: true), reference = new(config, generating: false);
            owners.Add(unet);
            owners.Add(reference);
            unet.LoadWeights(unetWeights, "unet");
            reference.LoadWeights(unetWeights, "unet_dual");
            HunyuanPaintPromptTokens prompts = HunyuanPaintPromptTokens.FromWeights(unetWeights, config, "unet");

            // The released VAE is F16 and the VAE kernels read F32: convert once, and release the copies with the loader.
            HunyuanPaintF32Weights vaeWeights = new(Read(Find(folder, "vae"), owners));
            owners.Add(vaeWeights);
            VaeConfig vaeConfig = VaeConfig.Sd15;
            VaeEncoder encoder = new(vaeConfig);
            encoder.LoadWeights(vaeWeights);
            VaeDecoder decoder = new(vaeConfig);
            decoder.LoadWeights(vaeWeights);

            IPaintConditioner? conditioner = null;
            Func<byte[], int, (byte[] Rgb, int Size)>? upscale = null;
            if (config.UseDino)
            {
                if (dinoPath is null)
                    throw new FileNotFoundException("Paint 2.1 conditions on DINOv2-giant; supply the facebook/dinov2-giant model.safetensors (or its folder).");
                Dinov2VisionEncoder dino = new(Dinov2Preset.Giant);
                HunyuanPaintF32Weights dinoWeights = new(Read(File.Exists(dinoPath) ? dinoPath : Path.Combine(dinoPath, "model.safetensors"), owners));
                owners.Add(dinoWeights);
                dino.LoadWeights(dinoWeights);
                HunyuanPaintF32Weights projection = new(unetWeights);
                owners.Add(projection);
                conditioner = new PaintDinoConditioner(dino, projection["unet.image_proj_model_dino.proj.weight"], projection["unet.image_proj_model_dino.proj.bias"],
                    projection["unet.image_proj_model_dino.norm.weight"], projection["unet.image_proj_model_dino.norm.bias"]);
                if (upscalerPath is not null)
                    upscale = BuildUpscaler(backend, upscalerPath, owners);
            }

            HunyuanPaintPipeline pipeline = new(backend, config, unet, reference, prompts, new SdVaeLatentCodec(encoder, decoder), conditioner, upscale);
            HunyuanPaintLoader loader = new(pipeline, config);
            loader._owners.AddRange(owners);
            return loader;
        }
        catch
        {
            for (int i = owners.Count - 1; i >= 0; i--)
                owners[i].Dispose();
            throw;
        }
    }

    /// <summary>Reads the generation from the UNet's tensor names and shapes.</summary>
    public static HunyuanPaintConfig Detect(IReadOnlyDictionary<string, Tensor> unetWeights)
    {
        ArgumentNullException.ThrowIfNull(unetWeights);
        if (unetWeights.ContainsKey("unet.learned_text_clip_albedo"))
            return HunyuanPaintConfig.V21;
        if (unetWeights.TryGetValue("unet.class_embedding.weight", out Tensor? classes))
            return HunyuanPaintConfig.V20 with { CameraClasses = (int)classes.Shape[0] };
        throw new InvalidDataException("This is not a Hunyuan3D Paint UNet: it has neither learned_text_clip_albedo (2.1) nor class_embedding (2.0).");
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        for (int i = _owners.Count - 1; i >= 0; i--)
            _owners[i].Dispose();
        _owners.Clear();
    }

    private static string Find(string folder, string part)
    {
        foreach (string name in new[] { "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin" })
        {
            string path = Path.Combine(folder, part, name);
            if (File.Exists(path))
                return path;
        }
        throw new FileNotFoundException($"No {part}/diffusion_pytorch_model.safetensors or .bin under '{folder}'.");
    }

    private static Dictionary<string, Tensor> Read(string path, List<IDisposable> owners)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            SafeTensorsLoader loader = new();
            loader.Load(path);
            owners.Add(loader);
            return loader.GetAllTensors();
        }
        PytorchPickleLoader pickle = new();
        pickle.Load(path);
        owners.Add(pickle);
        return pickle.GetAllTensors();
    }

    private static Func<byte[], int, (byte[] Rgb, int Size)> BuildUpscaler(IBackend backend, string path, List<IDisposable> owners)
    {
        Dictionary<string, Tensor> weights = RealEsrganConverter.Convert(Read(path, owners));
        RrdbNet net = new(RealEsrganConverter.InferConfig(weights));
        net.LoadWeights(weights);
        UpscalePipeline upscaler = new(backend, net);
        return (rgb, size) =>
        {
            (byte[] output, int width, _) = upscaler.Upscale(rgb, size, size);
            return (output, width);
        };
    }
}
