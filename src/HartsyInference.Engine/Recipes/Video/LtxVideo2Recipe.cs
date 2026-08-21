using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Runtime;
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

    /// <summary>Models-root-relative folder the shipped workflows keep the latent upsampler in.</summary>
    private const string UpsamplerSubdir = "latent_upscale_models";
    private const string DefaultLatentUpsamplerFile = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";

    private readonly bool _distilled;

    /// <summary>Constructs the recipe for the dev/base LTX-2 families.</summary>
    public LtxVideo2Recipe() : this(distilled: false) { }

    /// <summary>Constructs the recipe. <paramref name="distilled"/> selects the 2.5 distilled sampling contract,
    /// which cannot be detected from a checkpoint — the dev and distilled 2.5 transformers share a model version,
    /// architecture config and tensor keys, so the choice has to arrive as user intent via the model id.</summary>
    public LtxVideo2Recipe(bool distilled)
    {
        _distilled = distilled;
        if (distilled)
        {
            // The shipped template geometry: 1280x736 nominal (the two-stage half-grid snap decodes it at
            // 1280x704), 121 frames = 5 s at 24 fps. Same defaults as dev below — the families differ only in
            // the sampling contract.
            Defaults = new VideoDefaults
            {
                Steps = LtxVideo2Config.V25Distilled.NumInferenceSteps,
                CfgScale = LtxVideo2Config.V25Distilled.GuidanceScale,
                Width = 1280,
                Height = 736,
                Frames = 121,
                Fps = 24,
            };
        }
    }

    /// <inheritdoc/>
    public string Name => _distilled ? LtxVideo2DistilledRouting.DistilledFamilyId : "ltx-video-2";

    /// <inheritdoc/>
    public bool Matches(string familyId) => _distilled
        ? string.Equals(familyId, LtxVideo2DistilledRouting.DistilledFamilyId, StringComparison.OrdinalIgnoreCase)
        : LtxVideo2DistilledRouting.IsDevFamilyId(familyId);

    /// <summary>Dev-family defaults: 20 steps at cfg 4.0, 1280x736, 121 frames @ 24 fps — the geometry Lightricks
    /// ships (their template's 0.9 MP ResolutionSelector output and 5 s clip), at the measured recommended profile
    /// from MODEL_STATUS_VIDEO.md's LTX-2.5 row (quality parity vs ComfyUI at 1280x736 / 20 steps / cfg 4.0). The
    /// distilled 2.5 variant carries the same geometry with its baked 8-step unguided contract (ctor above).</summary>
    public VideoDefaults Defaults { get; private init; } = new VideoDefaults { Steps = 20, CfgScale = 4.0f, Width = 1280, Height = 736, Frames = 121, Fps = 24 };

    /// <inheritdoc/>
    /// <remarks><see cref="VideoFeatures.Lora"/> added 2026-08-20 — the first conditioning this family declares at all (it previously inherited <see cref="IVideoRecipe"/>'s <c>None</c>). NOTE: when the backend reports <c>SupportsQuantized</c> this recipe keeps the DiT resident as PACKED nvfp4, and <c>LoraStack</c> dequant-merges fp8 only — a non-fp8 quantized weight refuses by name with the "use a safetensors build" message, the same boundary HunyuanImage's Q4_K GGUF hits. A LoRA on a resident-nvfp4 LTX-2 build therefore refuses rather than merging; that is intended, not a regression. Image-to-video is still absent and tracked separately: only <c>LtxVideo2VaeDecoder</c> is ported, so there is no encoder to turn an init image into latents.</remarks>
    public VideoFeatures Supports => VideoFeatures.Lora;

    /// <inheritdoc/>
    public IVideoRecipePipeline Construct(RecipeContext context)
    {
        // TODO(E-IMG-4/5): LoRA, image-to-video conditioning, and a VideoRequest.Components Gemma/VAE override are deferred.
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        Dictionary<string, Tensor> merged = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        try
        {
            // An nvfp4 build only stays packed where the backend can consume a packed weight; CPU and Vulkan need the
            // eager unpack. SupportsQuantized is that line — it is exactly "this backend has a packed-weight GEMM path".
            bool residentNvfp4 = context.Backend.Capabilities.SupportsQuantized;
            AddFile(context.CheckpointPath, loaders, merged);
            LtxVideo2CheckpointConverter.ConvertedWeights conv = LtxVideo2CheckpointConverter.Convert(merged, residentNvfp4);

            if (conv.Vae.Count == 0)
            {
                Logs.Info("[LtxVideo2Recipe] Split LTX-2.3 model (no bundled VAE) — resolving side files: video VAE, audio VAE, text projection.");
                AddFile(ModelDownloader.EnsureSideModelAsync(SideModels.Ltx23VideoVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult(), loaders, merged);
                AddFile(ModelDownloader.EnsureSideModelAsync(SideModels.Ltx23AudioVae, onProgress: null, CancellationToken.None).GetAwaiter().GetResult(), loaders, merged);
                AddFile(ModelDownloader.EnsureSideModelAsync(SideModels.Ltx23TextProjection, onProgress: null, CancellationToken.None).GetAwaiter().GetResult(), loaders, merged);
                conv = LtxVideo2CheckpointConverter.Convert(merged, residentNvfp4);
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

            // Metadata from the file that actually carried the DiT; a split bundle's side files have their own.
            IReadOnlyDictionary<string, string>? metadata = loaders
                .FirstOrDefault(l => l.Descriptors.Keys.Any(LtxVideo2CheckpointConverter.IsTransformerKey))?.Metadata;
            LtxVideo2Config config = LtxVideo2VariantDetector.Detect(metadata, conv.Transformer.ContainsKey);
            if (_distilled)
            {
                config = ApplyDistilledContract(config);
                Logs.Info(config.TwoStage
                    ? "[LtxVideo2Recipe] Distilled variant selected — 8-step baked schedule, guidance 1, "
                        + "two-stage default-on (HARTSY_LTX2_TWO_STAGE=0 for single-pass)."
                    : "[LtxVideo2Recipe] Distilled variant selected on a pre-2.5 checkpoint — 8-step baked schedule, "
                        + "guidance 1, single-pass (the x2 latent upsampler is a 2.5 model; no two-stage here).");
            }
            LtxVideo2Transformer transformer = new LtxVideo2Transformer(config);
            // Merge any requested LoRAs BEFORE LoadWeights — device caches are identity-keyed, so merging
            // after would leave layers serving the pre-merge tensors (the Sd3Recipe ordering rule).
            MergedLoraStack? loraStack = LoraApplier.BuildAndApply(
                LoraResolver.Resolve(context.Loras), context.Backend, transformerWeights: conv.Transformer);
            transformer.LoadWeights(conv.Transformer);
            LtxVideo2TextConnectors connectors = new LtxVideo2TextConnectors(config);
            connectors.LoadWeights(conv.Connectors);

            // LTX-2.5's diffusion video decoder is what the official workflows decode with, and it decodes
            // CORRECTLY — verified stage by stage against the ComfyUI reference, and cleaner than the conv decoder
            // it replaces (high-frequency energy 2.06 vs 3.10 on the same latent), which is what the model card
            // claims for it. Every pass now runs over halo-padded temporal chunks sized off free VRAM
            // (LtxVideo25TemporalChunks), exact rather than blended, so the geometry ceiling is gone and
            // 768x512x97f decodes. It stays opt-in on cost, not correctness: ~40x the conv decoder at matched
            // geometry (12.9 s vs 2.878 s at 768x512x97f).
            bool wantDiffusionVae = EnvSwitch.IsEnabled("HARTSY_LTX2_DIFFUSION_VAE", defaultOn: false);
            bool haveConvDecoder = conv.Vae.ContainsKey("decoder.conv_in.conv.weight");
            LtxVideo25DiffusionDecoder? diffusionVae = null;
            if (conv.VaeDiffusionDecoder.Count > 0 && wantDiffusionVae)
            {
                // Pinning the budget pins the chunk PLAN, and decode time is a function of the plan — so an A/B or a
                // reproducible benchmark row needs this, and so does proving a plan change leaves the pixels alone.
                long chunkMb = Math.Max(0, EnvSwitch.GetLong("HARTSY_LTX25_VAE_CHUNK_MB", 0));
                diffusionVae = chunkMb > 0
                    ? new LtxVideo25DiffusionDecoder(new LtxVideo25DiffusionDecoderConfig { ChunkWorkspaceBytes = chunkMb << 20 })
                    : new LtxVideo25DiffusionDecoder();
                diffusionVae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.VaeDiffusionDecoder, DType.F32));
                Logs.Info($"[LtxVideo2Recipe] HARTSY_LTX2_DIFFUSION_VAE set — decoding with the LTX-2.5 diffusion "
                    + $"video decoder ({conv.VaeDiffusionDecoder.Count} tensors). Temporally chunked: no geometry "
                    + "ceiling, but ~13 s at 768x512x97f against the conv decoder's ~3 s.");
            }
            else if (conv.VaeDiffusionDecoder.Count > 0 && haveConvDecoder)
            {
                Logs.Info("[LtxVideo2Recipe] Checkpoint carries the LTX-2.5 diffusion video decoder; using the conv "
                    + "decoder, which is ~40x faster at matched geometry (HARTSY_LTX2_DIFFUSION_VAE=1 to select "
                    + "the diffusion one).");
            }
            else if (conv.VaeDiffusionDecoder.Count > 0)
            {
                throw new InvalidOperationException(
                    $"LTX-2 checkpoint '{context.CheckpointPath}' carries ONLY the LTX-2.5 diffusion video VAE "
                    + $"({conv.VaeDiffusionDecoder.Count} decoder tensors), which decodes correctly but is currently "
                    + "~40x slower at matched geometry. Supply the convolutional VAE "
                    + "(ltx-2.5-video-vae-conv-bf16.safetensors) for the fast path, or set "
                    + "HARTSY_LTX2_DIFFUSION_VAE=1 to use this one.");
            }
            // Gemma 4 (LTX-2.5) vs Gemma 3 (LTX-2.3). `layer_scalar` is the discriminator because it is per-block
            // and Gemma 3 has no counterpart; do NOT probe for a missing v_proj — layer 0 is a sliding layer and
            // has one, so that test would misclassify every Gemma 4 checkpoint as Gemma 3.
            bool isGemma4 = conv.TextEncoder.ContainsKey("model.layers.0.layer_scalar");

            (float[]? videoMean, float[]? videoStd) = ReadStats(conv.Vae, config.InChannels);
            // BF16 where the backend has it (the engine-wide VAE policy — never F16, see VaePrecisionHelper).
            // At 768x512x97f the decode's peak transient set is several full-output-grid tensors; halving their
            // width is what keeps it under the VRAM the DiT's resident prefix leaves behind, so the prefix no
            // longer has to be evicted and re-uploaded around every decode.
            DType vaeDtype = VaePrecisionHelper.PreferredVaeDtype(context.Backend);
            LtxVideo2VaeDecoder vae = new LtxVideo2VaeDecoder(latentsMean: videoMean, latentsStd: videoStd,
                computeDtype: vaeDtype);
            // A diffusion-decoder checkpoint carries no conv decoder keys, so only load one when they are there.
            if (haveConvDecoder)
            {
                vae.LoadWeights(VaePrecisionHelper.CastVaeWeights(conv.Vae, vaeDtype));
            }
            else if (diffusionVae is null)
            {
                throw new InvalidOperationException(
                    $"LTX-2 checkpoint '{context.CheckpointPath}' has no usable video decoder: no conv decoder keys "
                    + "and no diffusion decoder (or HARTSY_LTX2_DIFFUSION_VAE selected one that is not in the checkpoint).");
            }

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
            ILtx2TextTower gemma;
            ILtx2PromptTokenizer tokenizer;
            string? gemmaSidePath = null;
            if (isGemma4)
            {
                // The 49-state harvest's final norm is an unresolved divergence (docs/Research/LTX_2_5.md
                // "Divergence 1"): `model.norm.weight` measures max 600 / mean 20 on the real checkpoint, so
                // norming ONLY the last state makes it outweigh the other 48 by that factor. The verified
                // LTX-2.3 path norms no state and drives the SAME connector, so this matches it.
                Gemma4TextEncoder gemma4 = new Gemma4TextEncoder(
                    Gemma4TextEncoderConfig.Gemma4_12B with { ApplyFinalNormToLastState = false });
                gemma4.LoadWeights(conv.TextEncoder);
                gemma = gemma4;
                // Gemma 4 ships its own tokenizer INSIDE the encoder safetensors as a U8 `tokenizer_json` tensor,
                // so unlike Gemma 3 there is no side file to locate.
                if (!conv.TextEncoder.TryGetValue("tokenizer_json", out Tensor? tokenizerJson))
                {
                    throw new InvalidOperationException(
                        $"LTX-2.5 checkpoint '{context.CheckpointPath}' has a Gemma 4 text tower but no embedded "
                        + "'tokenizer_json' tensor; the Gemma 4 vocabulary ships only inside that file.");
                }
                tokenizer = ReadGemma4Tokenizer(tokenizerJson);
                Logs.Info($"[LtxVideo2Recipe] Gemma-4-12B text tower loaded ({conv.TextEncoder.Count} keys), "
                    + $"conditioning length {Gemma4Tokenizer.LtxMinLength}.");
            }
            else
            {
                LlamaStyleEncoder gemma3 = new LlamaStyleEncoder(LlamaStyleEncoderConfig.Gemma3_12B);
                if (conv.TextEncoder.Count > 0)
                {
                    gemma3.LoadWeights(conv.TextEncoder);
                    Logs.Info("[LtxVideo2Recipe] Gemma-3-12B text tower loaded (bundled).");
                }
                else
                {
                    gemmaSidePath = ModelDownloader.EnsureSideModelAsync(SideModels.GemmaLtx2, onProgress: null, CancellationToken.None).GetAwaiter().GetResult();
                    SafeTensorsLoader gemmaLoader = new SafeTensorsLoader();
                    gemmaLoader.Load(gemmaSidePath);
                    loaders.Add(gemmaLoader);
                    gemma3.LoadWeights(gemmaLoader.GetAllTensors());
                    Logs.Info($"[LtxVideo2Recipe] Gemma-3-12B text tower loaded (standalone: {Path.GetFileName(gemmaSidePath)}).");
                }
                gemma = gemma3;
                tokenizer = new GemmaTokenizer(LocateGemmaTokenizer(context.CheckpointPath, gemmaSidePath), maxLength: TokenLength);
            }

            LtxLatentUpsampler? latentUpsampler = LoadLatentUpsampler(config, loaders);

            LtxVideo2Pipeline pipeline = new LtxVideo2Pipeline(context.Backend, transformer, connectors, vae, gemma, config,
                audioVae, vocoder, audioMean, audioStd, diffusionVae, videoMean, videoStd)
            {
                MinimumTextConditioningLength = tokenizer.MinimumConditioningLength,
                TextEncoderBackend = context.TextEncoderBackendOrDefault,
                VaeBackend = context.VaeBackendOrDefault,
                TwoStage = latentUpsampler is not null,
                LatentUpsampler = latentUpsampler,
            };
            Logs.Info($"[LtxVideo2Recipe] LTX-2 ready (text-to-video{(vocoder is not null ? "+audio" : "")}).");
            return new LtxVideo2RecipePipeline(pipeline, config, tokenizer, gemma, transformer, connectors, vocoder, loaders, loraStack);
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

    /// <summary>The distilled sampling contract over a DETECTED architecture config. The 8-step base schedule is
    /// shared by every 2.x distilled template, but two-stage stays 2.5-only: distilled builds exist for older
    /// generations too (2.0's templates ship a 0.909375-head refine; a 2.3 distilled LoRA is documented), and the
    /// x2 upsampler is a 2.5 model — running it on 2.3 latents is unverified numerics.</summary>
    internal static LtxVideo2Config ApplyDistilledContract(LtxVideo2Config detected) => detected with
    {
        FixedSigmas = LtxVideo2Config.Ltx25DistilledSigmas,
        NumInferenceSteps = LtxVideo2Config.V25Distilled.NumInferenceSteps,
        GuidanceScale = LtxVideo2Config.V25Distilled.GuidanceScale,
        TwoStage = detected.UseKeyframesAbsPosEmbedding && LtxVideo2Config.V25Distilled.TwoStage,
    };

    /// <summary>Loads the LTX-2.5 learned x2 latent upsampler for the two-stage flow, or returns null when the flow
    /// is off. Default ON for the distilled family (<see cref="LtxVideo2Config.V25Distilled"/> carries
    /// <c>TwoStage = true</c>) — <c>HARTSY_LTX2_TWO_STAGE=0</c> is the single-pass kill-switch, and =1 the opt-in
    /// probe elsewhere. Distilled-only either way: the dev checkpoints ship no two-stage reference configuration,
    /// so enabling it there would be guesswork. <c>HARTSY_LTX2_UPSAMPLER</c> names the file; otherwise the shipped
    /// name is resolved under <c>Models/latent_upscale_models/</c> (auto-downloaded when absent).</summary>
    private LtxLatentUpsampler? LoadLatentUpsampler(LtxVideo2Config config, List<SafeTensorsLoader> loaders)
    {
        if (!EnvSwitch.IsEnabled("HARTSY_LTX2_TWO_STAGE", defaultOn: config.TwoStage))
        {
            return null;
        }
        if (!_distilled)
        {
            Logs.Warning("[LtxVideo2Recipe] HARTSY_LTX2_TWO_STAGE is set but this is not the distilled family — "
                + "the two-stage sigma schedule and upsample point are only documented for ltx-2.5-distilled. Running single-pass.");
            return null;
        }
        string? named = Environment.GetEnvironmentVariable("HARTSY_LTX2_UPSAMPLER") is { Length: > 0 } n ? n : null;
        string requested = named ?? DefaultLatentUpsamplerFile;
        // The folder scan is DISCOVERY for the default name only. Falling back to it when the caller named a file
        // would load a different upsampler than the one they asked for, with only an Info line to notice. When
        // nothing was named and nothing is on disk, fetch the shipped upsampler — two-stage is the distilled
        // default, so a missing side file must not turn a working install into a throw.
        string? path = ModelFileLocator.Find(requested, UpsamplerSubdir)
            ?? (named is null ? FindAnyLatentUpsampler() : null);
        if (path is null && named is null)
        {
            Logs.Info($"[LtxVideo2Recipe] Latent upsampler not on disk — downloading {SideModels.Ltx25LatentUpsampler.Repo}/"
                + $"{SideModels.Ltx25LatentUpsampler.RepoPath}.");
            path = ModelDownloader.EnsureSideModelAsync(SideModels.Ltx25LatentUpsampler, onProgress: null, CancellationToken.None)
                .GetAwaiter().GetResult();
        }
        if (path is null)
        {
            throw new FileNotFoundException(
                $"LTX-2.5 two-stage is enabled but the latent upsampler '{requested}' was not found under "
                + $"'{Path.Combine(RepoPaths.ModelsRoot(), UpsamplerSubdir)}'. Download it from "
                + "Lightricks/LTX-2.5 (latent_upscale_models/) or set HARTSY_LTX2_TWO_STAGE=0 for single-pass.");
        }

        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(path);
        loaders.Add(loader);
        LtxLatentUpsampler upsampler = new LtxLatentUpsampler();
        // F32 throughout: Conv3d has no lower-precision path on any backend, so the BF16 file doubles on load.
        upsampler.LoadWeights(VaePrecisionHelper.CastVaeWeights(loader.GetAllTensors(), DType.F32),
            loader.Metadata is not null && loader.Metadata.TryGetValue("config", out string? cfg) ? cfg : null);
        Logs.Info($"[LtxVideo2Recipe] Two-stage enabled: latent upsampler {Path.GetFileName(path)} "
            + $"(in={upsampler.InChannels} mid={upsampler.MidChannels} blocks={upsampler.NumBlocksPerStage} x{upsampler.SpatialScale}).");
        return upsampler;
    }

    /// <summary>Any safetensors under the upsampler folder whose name looks like a latent upscaler — the shipped
    /// file carries a version suffix that a future release will bump.</summary>
    private static string? FindAnyLatentUpsampler()
    {
        string dir = Path.Combine(RepoPaths.ModelsRoot(), UpsamplerSubdir);
        if (!Directory.Exists(dir))
        {
            return null;
        }
        foreach (string file in Directory.EnumerateFiles(dir, "*.safetensors", SearchOption.AllDirectories))
        {
            string name = Path.GetFileName(file);
            if (name.Contains("upscaler", StringComparison.OrdinalIgnoreCase)
                || name.Contains("upsampler", StringComparison.OrdinalIgnoreCase))
            {
                return file;
            }
        }
        return null;
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

    /// <summary>Builds the Gemma 4 tokenizer from the <c>tokenizer_json</c> U8 tensor LTX-2.5 embeds in its text
    /// encoder — a ~32 MB HuggingFace <c>tokenizer.json</c> with no side file anywhere to fall back to.</summary>
    private static unsafe Gemma4Tokenizer ReadGemma4Tokenizer(Tensor tokenizerJson)
    {
        if (tokenizerJson.DType != DType.U8)
        {
            throw new InvalidOperationException(
                $"'tokenizer_json' must be a U8 blob; got {tokenizerJson.DType}.");
        }
        return Gemma4Tokenizer.FromTokenizerJson(
            new ReadOnlySpan<byte>(tokenizerJson.DataPointer, (int)tokenizerJson.ElementCount));
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
