using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>MiniMax-H3 ("Hailuo 03") recipe — a single-stream packed-token DiT denoising video and stereo audio
/// jointly, with a ViT3D video VAE and a DAC/BigVGAN audio VAE. Components are located by
/// <see cref="MiniMaxH3Assets"/>, which accepts both the vendor folder tree and the flat Comfy/SwarmUI repack. The
/// DiT is loaded without an F32 cast because the bf16 release is larger than host RAM.
/// See <c>docs/Research/MINIMAX_H3.md</c>.</summary>
public sealed class MiniMaxH3Recipe : IVideoRecipe
{
    // ModelScope publishes as MiniMax/*, HuggingFace as MiniMaxAI/*, and the checkpoint may say "Hailuo 03".
    private static readonly string[] _familyIds = { "minimax-h3", "minimax-hailuo-03", "hailuo-03", "hailuo03" };

    /// <inheritdoc/>
    public string Name => "minimax-h3";

    /// <inheritdoc/>
    public bool Matches(string familyId)
    {
        foreach (string id in _familyIds)
        {
            if (string.Equals(familyId, id, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    /// <summary>The reference canvas for 16:9 (768 short edge, 768x1344 area cap, axes rounded to 32) and the
    /// reference default length — 124 frames is the first 17k+5 value at ~5 s / 24 fps.</summary>
    public VideoDefaults Defaults { get; } =
        new VideoDefaults { Steps = 30, CfgScale = 1.0f, Width = 1344, Height = 768, Frames = 124, Fps = 24 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        try
        {
            if (context.Backend is HartsyInference.Cuda.CudaBackend cudaBackend)
            {
                // 66 GB bf16 DiT: caching every F16 cast grows without bound and OOMs the host.
                cudaBackend.CacheWeightCasts = false;
                Logs.Info("[MiniMaxH3Recipe] CacheWeightCasts disabled (bf16-resident DiT).");
            }
            MiniMaxH3Transformer transformer = LoadTransformer(assets.Dit, loaders, out MiniMaxH3Config config);
            MiniMaxH3VideoVaeDecoder videoVae = LoadVideoVae(assets.VideoVae, loaders);
            MiniMaxH3AudioVaeDecoder? audioVae = LoadAudioVae(assets.AudioVae, loaders);

            MiniMaxH3TextEncoder textEncoder = LoadTextEncoder(assets.TextEncoder, loaders);
            MiniMaxH3Pipeline pipeline = new MiniMaxH3Pipeline(context.Backend, transformer, videoVae, audioVae);
            return new MiniMaxH3RecipePipeline(context.Backend, pipeline, config, textEncoder,
                LoadTokenizer(assets.TokenizerDir), loaders);
        }
        catch
        {
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    private static MiniMaxH3Transformer LoadTransformer(string file, List<SafeTensorsLoader> loaders,
        out MiniMaxH3Config config)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(file);
        loaders.Add(loader);
        Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3CheckpointConverter.ThrowIfInt8Convrot(raw);
        // The bf16 DiT is larger than host RAM, so it stays bf16 and the backend casts per call.
        MiniMaxH3CheckpointConverter.ConvertedWeights converted =
            MiniMaxH3CheckpointConverter.Convert(raw, castToF32: false);
        // Norm/rope weights must be F32: CudaBackend.RmsNorm only takes its GPU path when BOTH activation and
        // weight are F32, so a bf16 norm weight silently leaves the intended path. The big linears stay bf16.
        Dictionary<string, Tensor> promotedWeights = new Dictionary<string, Tensor>(converted.Transformer);
        int promoted = 0;
        foreach (string key in promotedWeights.Keys.ToList())
        {
            bool isNorm = key.EndsWith("norm.weight", StringComparison.Ordinal)
                || key.EndsWith("norm1.weight", StringComparison.Ordinal)
                || key.EndsWith("norm2.weight", StringComparison.Ordinal)
                || key.Equals("rope.inv_freq", StringComparison.Ordinal);
            if (isNorm && promotedWeights[key].DType != DType.F32)
            {
                promotedWeights[key] = promotedWeights[key].CastTo(DType.F32);
                promoted++;
            }
        }
        Logs.Info($"[MiniMaxH3Recipe] Promoted {promoted} norm/rope tensors to F32.");
        config = MiniMaxH3Config.Detect(promotedWeights);
        Logs.Info($"[MiniMaxH3Recipe] DiT: {converted.Transformer.Count} tensors, {config.NumLayers} blocks, "
            + $"hidden {config.HiddenSize}, curves={config.UseAdalnCurves}.");
        MiniMaxH3Transformer transformer = new MiniMaxH3Transformer(config);
        transformer.LoadWeights(promotedWeights);
        return transformer;
    }

    private static MiniMaxH3VideoVaeDecoder LoadVideoVae(string file, List<SafeTensorsLoader> loaders)
    {
        string dir = Path.GetDirectoryName(file)!;
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(file);
        loaders.Add(loader);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        string wrapper = Path.Combine(dir, "config.json");
        string source = Path.Combine(dir, "source", "config.json");
        MiniMaxH3VideoVaeConfig config = File.Exists(wrapper)
            ? MiniMaxH3VideoVaeConfig.FromJson(File.ReadAllText(wrapper),
                File.Exists(source) ? File.ReadAllText(source) : null)
            : MiniMaxH3VideoVaeConfig.Detect(weights);
        MiniMaxH3VideoVaeDecoder decoder = new MiniMaxH3VideoVaeDecoder(config);
        decoder.LoadWeights(weights);
        Logs.Info($"[MiniMaxH3Recipe] Video VAE: {weights.Count} tensors.");
        return decoder;
    }

    private static MiniMaxH3AudioVaeDecoder? LoadAudioVae(string? file, List<SafeTensorsLoader> loaders)
    {
        if (file is null)
        {
            return null;
        }
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(file);
        loaders.Add(loader);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3AudioVaeDecoder decoder = new MiniMaxH3AudioVaeDecoder();
        decoder.LoadWeights(weights);
        Logs.Info($"[MiniMaxH3Recipe] Audio VAE: {weights.Count} tensors @ {decoder.SampleRate} Hz.");
        return decoder;
    }

    private static MiniMaxH3TextEncoder LoadTextEncoder(string file, List<SafeTensorsLoader> loaders)
    {
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(file);
        loaders.Add(loader);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3TextEncoder encoder = new MiniMaxH3TextEncoder();
        encoder.LoadWeights(weights);
        Logs.Info($"[MiniMaxH3Recipe] Text encoder: {weights.Count} tensors.");
        return encoder;
    }

    /// <summary>The flat repack ships no tokenizer files; the embedded Qwen BPE is the same base merge table.</summary>
    private static Qwen2Tokenizer LoadTokenizer(string? dir)
    {
        if (dir is null)
        {
            Logs.Info("[MiniMaxH3Recipe] Using the embedded Qwen tokenizer (no vocab.json/merges.txt alongside the checkpoint).");
            return new Qwen2Tokenizer();
        }
        return new Qwen2Tokenizer(Path.Combine(dir, "vocab.json"), Path.Combine(dir, "merges.txt"));
    }
}
