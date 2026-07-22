using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Video.Pipelines;

using HartsyInference.Engine.Features;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>LTX-2.3 recipe (Lightricks, 22B audiovisual; SwarmUI compat class <c>lightricks-ltx-video-2</c>). The
/// bundled single-file/sharded checkpoint carries the dual-stream DiT, the per-modality text connectors, the video
/// VAE, the audio VAE, the vocoder, and — when present — the Gemma-3-12B text tower; a SPLIT LTX-2.3 model ships the
/// DiT alone and the video VAE (<see cref="SideModels.Ltx23VideoVae"/>), audio VAE
/// (<see cref="SideModels.Ltx23AudioVae"/>), and text projection (<see cref="SideModels.Ltx23TextProjection"/>) are
/// resolved as side models, with the standalone Gemma tower (<see cref="SideModels.GemmaLtx2"/>) when the checkpoint
/// omits it. Lifted from the SwarmUI backend's <c>LtxVideo2Loader</c>.</summary>
public sealed class LtxVideo2Recipe : IVideoRecipe
{
    /// <summary>Gemma context length fed to the connectors (padded to a register multiple inside the pipeline).</summary>
    internal const int TokenLength = 256;

    /// <inheritdoc/>
    public string Name => "ltx-video-2";

    /// <inheritdoc/>
    public bool Matches(string familyId) =>
        string.Equals(familyId, "ltx-video-2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "ltx-video2", StringComparison.OrdinalIgnoreCase)
        || string.Equals(familyId, "lightricks-ltx-video-2", StringComparison.OrdinalIgnoreCase);

    /// <summary>LTX-Video 2's official sampling settings: 50 steps at guidance 3.0, 512x320, 25 frames @ 24fps —
    /// the geometry <c>LtxVideo2GenerationTests</c> verified coherent, 22B being too heavy to sample at full 704x480
    /// in a reasonable CLI turnaround (<c>LtxVideo2Config.NumInferenceSteps</c>/<c>GuidanceScale</c>).</summary>
    public VideoDefaults Defaults { get; } = new VideoDefaults { Steps = 50, CfgScale = 3.0f, Width = 512, Height = 320, Frames = 25, Fps = 24 };

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA, image-to-video conditioning, and a VideoRequest.Components Gemma/VAE override are deferred.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try
        {
            AddFile(context.CheckpointPath, loaders, merged);
            LtxVideo2CheckpointConverter.ConvertedWeights conv = LtxVideo2CheckpointConverter.Convert(merged);

            if (conv.Vae.Count == 0)
            {
                Logs.Info("[LtxVideo2Recipe] Split LTX-2.3 model (no bundled VAE) — resolving side files: video VAE, audio VAE, text projection.");
                AddFile(ModelDownloader.EnsureSideModelAsync(SideModels.Ltx23VideoVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult(), loaders, merged);
                AddFile(ModelDownloader.EnsureSideModelAsync(SideModels.Ltx23AudioVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult(), loaders, merged);
                AddFile(ModelDownloader.EnsureSideModelAsync(SideModels.Ltx23TextProjection, onProgress: null, CancellationToken.None).GetAwaiter().GetResult(), loaders, merged);
                conv = LtxVideo2CheckpointConverter.Convert(merged);
            }

            if (conv.Transformer.Count == 0)
            {
                throw new InvalidOperationException($"LTX-2 checkpoint '{context.CheckpointPath}' has no recognized DiT weights after conversion.");
            }
            if (conv.Connectors.Count == 0)
            {
                throw new InvalidOperationException(
                    $"LTX-2 checkpoint '{context.CheckpointPath}' has no text-connector weights — the bundle must include the per-modality embeddings connectors.");
            }
            if (conv.Vae.Count == 0)
            {
                throw new InvalidOperationException($"LTX-2 checkpoint '{context.CheckpointPath}' has no bundled video VAE weights.");
            }
            Logs.Info($"[LtxVideo2Recipe] Converted {conv.Transformer.Count} DiT, {conv.Connectors.Count} connector, {conv.Vae.Count} VAE, "
                + $"{conv.AudioVae.Count} audio-VAE, {conv.Vocoder.Count} vocoder, {conv.TextEncoder.Count} text-encoder keys.");

            LtxVideo2Config config = LtxVideo2Config.V23;
            LtxVideo2Transformer transformer = new LtxVideo2Transformer(config);
            transformer.LoadWeights(conv.Transformer);
            LtxVideo2TextConnectors connectors = new LtxVideo2TextConnectors(config);
            connectors.LoadWeights(conv.Connectors);

            (float[]? videoMean, float[]? videoStd) = ReadStats(conv.Vae, config.InChannels);
            LtxVideo2VaeDecoder vae = new LtxVideo2VaeDecoder(latentsMean: videoMean, latentsStd: videoStd);
            vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.Vae, DType.F32));

            LtxAudioVaeDecoder? audioVae = null;
            LtxAudioVocoder? vocoder = null;
            float[]? audioMean = null;
            float[]? audioStd = null;
            if (conv.AudioVae.Count > 0 && conv.Vocoder.Count > 0)
            {
                // Audio latent stats are stored over the packed feature axis (8 latent ch x 16 mel = 128).
                (audioMean, audioStd) = ReadStats(conv.AudioVae, config.AudioInChannels);
                audioVae = new LtxAudioVaeDecoder();
                audioVae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.AudioVae, DType.F32));
                vocoder = new LtxAudioVocoder();
                vocoder.LoadWeights(conv.Vocoder);
                Logs.Info("[LtxVideo2Recipe] Audio decode wired (VAE + vocoder).");
            }
            else
            {
                Logs.Info("[LtxVideo2Recipe] No bundled audio VAE/vocoder — video-only output.");
            }

            // The standalone Gemma safetensors stays mapped for the pipeline's lifetime: its tensors are mmap views.
            LlamaStyleEncoder gemma = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Gemma3_12B);
            string? gemmaSidePath = null;
            if (conv.TextEncoder.Count > 0)
            {
                gemma.LoadWeights(conv.TextEncoder);
                Logs.Info("[LtxVideo2Recipe] Gemma-3-12B text tower loaded (bundled).");
            }
            else
            {
                gemmaSidePath = ModelDownloader.EnsureSideModelAsync(SideModels.GemmaLtx2, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                SafeTensorsLoader gemmaLoader = new SafeTensorsLoader();
                gemmaLoader.Load(gemmaSidePath);
                loaders.Add(gemmaLoader);
                gemma.LoadWeights(gemmaLoader.GetAllTensors());
                Logs.Info($"[LtxVideo2Recipe] Gemma-3-12B text tower loaded (standalone: {Path.GetFileName(gemmaSidePath)}).");
            }

            GemmaTokenizer tokenizer = new GemmaTokenizer(LocateGemmaTokenizer(context.CheckpointPath, gemmaSidePath), maxLength: TokenLength);

            LtxVideo2Pipeline pipeline = new LtxVideo2Pipeline(context.Backend, transformer, connectors, vae, gemma, config, audioVae, vocoder, audioMean, audioStd);
            Logs.Info($"[LtxVideo2Recipe] LTX-2 ready (text-to-video{(vocoder is not null ? "+audio" : "")}).");
            return new LtxVideo2RecipePipeline(pipeline, config, tokenizer, gemma, transformer, connectors, vocoder, loaders);
        }
        catch (Exception ex)
        {
            Logs.Error("[LtxVideo2Recipe] Construction failed.", ex);
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Merges one safetensors file (or every shard under a directory) into the routing dictionary.</summary>
    private static void AddFile(string path, List<SafeTensorsLoader> loaders, Dictionary<string, Tensor> merged)
    {
        if (File.Exists(path))
        {
            SafeTensorsLoader loader = new SafeTensorsLoader();
            loader.Load(path);
            loaders.Add(loader);
            foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
            {
                merged[kv.Key] = kv.Value;
            }
            return;
        }
        if (Directory.Exists(path))
        {
            string[] shards = Directory.GetFiles(path, "*.safetensors", SearchOption.AllDirectories);
            if (shards.Length == 0)
            {
                throw new FileNotFoundException($"No safetensors shards in: {path}");
            }
            foreach (string shard in shards)
            {
                SafeTensorsLoader loader = new SafeTensorsLoader();
                loader.Load(shard);
                loaders.Add(loader);
                foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
                {
                    merged[kv.Key] = kv.Value;
                }
            }
            return;
        }
        throw new FileNotFoundException($"LTX-2 file not found: {path}");
    }

    /// <summary>Reads the per-channel latent normalization stats from a converted VAE bucket, trying the original
    /// Lightricks names first and the diffusers names second; nulls mean "no denormalization".</summary>
    private static (float[]? Mean, float[]? Std) ReadStats(Dictionary<string, Tensor> vae, int channels)
    {
        Tensor? mean = Find(vae, "per_channel_statistics.mean-of-means", "latents_mean");
        Tensor? std = Find(vae, "per_channel_statistics.std-of-means", "latents_std");
        if (mean is null || std is null)
        {
            return (null, null);
        }
        return (ToFloatArray(mean, channels), ToFloatArray(std, channels));
    }

    /// <summary>First tensor matching any of <paramref name="keys"/>, or null.</summary>
    private static Tensor? Find(Dictionary<string, Tensor> weights, params string[] keys)
    {
        foreach (string key in keys)
        {
            if (weights.TryGetValue(key, out Tensor? tensor))
            {
                return tensor;
            }
        }
        return null;
    }

    /// <summary>Copies the first <paramref name="count"/> elements of a tensor into an F32 array.</summary>
    private static unsafe float[] ToFloatArray(Tensor tensor, int count)
    {
        Tensor f32 = tensor.DType == DType.F32 ? tensor : tensor.CastTo(DType.F32);
        int n = (int)Math.Min(count, f32.Shape.ElementCount);
        float[] result = new float[n];
        float* p = (float*)f32.DataPointer;
        for (int i = 0; i < n; i++)
        {
            result[i] = p[i];
        }
        return result;
    }

    /// <summary>Finds the Gemma SentencePiece model next to the checkpoint, or next to the standalone Gemma tower —
    /// Gemma ships no embedded tokenizer in the engine.</summary>
    private static string LocateGemmaTokenizer(string checkpointPath, string? gemmaSidePath)
    {
        List<string> dirs = new List<string>();
        string? ckptDir = File.Exists(checkpointPath) ? Path.GetDirectoryName(checkpointPath) : checkpointPath;
        if (!string.IsNullOrEmpty(ckptDir))
        {
            dirs.Add(ckptDir);
        }
        if (!string.IsNullOrEmpty(gemmaSidePath))
        {
            string? gemmaDir = Path.GetDirectoryName(gemmaSidePath);
            if (!string.IsNullOrEmpty(gemmaDir) && !dirs.Contains(gemmaDir))
            {
                dirs.Add(gemmaDir);
            }
        }
        foreach (string dir in dirs)
        {
            if (!Directory.Exists(dir))
            {
                continue;
            }
            foreach (string candidate in new[] { "tokenizer.model", "gemma.model", "gemma3.model" })
            {
                string path = Path.Combine(dir, candidate);
                if (File.Exists(path))
                {
                    return path;
                }
            }
            string[] spm = Directory.GetFiles(dir, "*.model");
            if (spm.Length > 0)
            {
                return spm[0];
            }
        }
        throw new FileNotFoundException($"LTX-2 needs the Gemma SentencePiece tokenizer (tokenizer.model) next to the checkpoint (searched: {string.Join(", ", dirs)}).");
    }
}
