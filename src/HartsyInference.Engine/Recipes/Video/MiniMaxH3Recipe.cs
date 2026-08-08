using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Placement;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Music;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Video.Pipelines;
using MergedLoraStack = HartsyInference.ModelAssets.Lora.LoraStack;

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
    /// <remarks>MiniMax-H3 VAE-encodes the start and end images into keyframe conditioning, and is the first video
    /// family to merge LoRAs — on either build, since an fp8 target is dequantized, merged and requantized.</remarks>
    public VideoFeatures Supports => VideoFeatures.InitImage | VideoFeatures.EndFrame | VideoFeatures.Lora
        | VideoFeatures.ReferenceImages | VideoFeatures.ReferenceVideos | VideoFeatures.ReferenceAudios;
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
        WarnIfPlacementIgnored(context);
        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(context.CheckpointPath);
        List<SafeTensorsLoader> loaders = new List<SafeTensorsLoader>();
        MergedLoraStack? loraStack = null;
        try
        {
            // The 66 GB bf16 build cannot stay device-resident on any consumer card, alone or sharded across two —
            // this is a fixed structural fact of the build, not something live free VRAM changes. The 21 GB fp8
            // build CAN fit resident, but whether it actually should on THIS box right now is a separate, live
            // question — see fitsResidentSingleBackend below.
            bool ditIsHugeBf16Build = new FileInfo(assets.Dit).Length >= 40L << 30;
            if (ditIsHugeBf16Build)
            {
                // Caching its F16 casts grows without bound and OOMs the host; the fp8 build must keep the cache ON
                // or every GEMM re-uploads its weight.
                RecipeBackendFlags.DisableCacheWeightCasts(context, "MiniMaxH3Recipe");
            }
            else
            {
                // A card without native fp8 GEMM dequants every fp8 weight per call, and the cache keeps those F16
                // copies resident until only its own headroom floor (2 GB) is left — but H3's per-block transients
                // need ~5 GB at long geometries, so the cache silently eats the activation budget and OOMs
                // mid-forward with no pre-flight signal. Native-fp8 cards never cast at all, so they keep it on.
                RecipeBackendFlags.DisableCacheWeightCasts(context, "MiniMaxH3Recipe", onlyWithoutNativeFp8Gemm: true);
            }
            // F32, and F16 is not a bug to chase: this DiT's stream genuinely leaves F16 range on real weights —
            // condition_proj already emits 82740 (2 of 5376 text channels overflow to inf before block 0) and the
            // residual reaches 2.7e6 by the last block. BF16 holds the range but falls off the native fp8 GEMM
            // guard (CudaBackend.cs:959 takes fp8/F32/F16 inputs only) for a 46% slowdown. The 2-layer parity
            // test passes both, so it does not gate this — only a real-weight run does.
            DType bodyDType = DType.F32;
            MiniMaxH3Transformer transformer = LoadTransformer(assets.Dit, loaders, bodyDType,
                out MiniMaxH3Config config, out Dictionary<string, Tensor> ditWeights);
            // Must precede LoadWeights: the merge swaps dictionary entries, and device caches are identity-keyed, so a
            // tensor already captured by a layer would keep serving its stale device copy.
            loraStack = ApplyLoras(context, ditWeights);
            transformer.LoadWeights(ditWeights);
            (MiniMaxH3VideoVaeDecoder videoVae, MiniMaxH3VideoVaeEncoder? videoVaeEncoder) =
                LoadVideoVae(assets.VideoVae, loaders);
            (MiniMaxH3AudioVaeDecoder? audioVae, MiniMaxH3AudioVaeEncoder? audioVaeEncoder) =
                LoadAudioVae(assets.AudioVae, loaders);

            MiniMaxH3TextEncoder textEncoder = LoadTextEncoder(assets.TextEncoder, loaders);

            // A real check, not the file-size proxy: the fp8 build normally fits, but a box that's already tight on
            // this card (another tenant, a prior leaked allocation) should stream rather than have PreloadWeights
            // fail and fall back anyway — this just makes that outcome the first attempt instead of a caught
            // exception. VramPlanner's cache-based Resident/Streamed decision doesn't fit here: it returns Resident
            // whenever there's no IStreamingWeightCache, and H3 has none — its "streaming" is a per-call
            // PreloadWeights/FreeWeights toggle, not a windowed cache — so this is a direct comparison of the same
            // weight-bytes-vs-free-VRAM shape rather than a VramPlanner.PlanPhase call. Only gates the SINGLE-backend
            // preload flag below; sharding eligibility is a different, build-size question (see ditIsHugeBf16Build).
            long ditWeightBytes = 0;
            foreach (Tensor t in transformer.EnumerateWeights())
            {
                ditWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
            }
            (long primaryFreeBytes, _) = context.Backend.GetVramInfo();
            bool fitsResidentSingleBackend = !ditIsHugeBf16Build && (primaryFreeBytes <= 0
                || primaryFreeBytes >= ditWeightBytes + MinimumActivationReserveBytes(config));

            // DiT sharding: fp8 build only — the bf16 blocks alone (~64 GB) exceed any 2-consumer-card pool, so
            // sharding buys nothing there and the streaming path stands. 50 homogeneous blocks → the
            // count-proportional plan is byte-accurate. Gated on the build-size class, not
            // fitsResidentSingleBackend — sharding is precisely for when the DiT does NOT fit on one card alone.
            int ditShardSplitBlock = 0;
            IBackend? ditShardBackend = null;
            if (context.DitShardBackend is not null && !ditIsHugeBf16Build)
            {
                long sharedWeightBytes = 0;
                foreach (Tensor t in transformer.EnumerateSharedWeights())
                {
                    sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                (long freeA, _) = context.Backend.GetVramInfo();
                (long freeB, _) = context.DitShardBackend.GetVramInfo();
                ditShardSplitBlock = PlacementPlanner.DitSplitPlan([freeA, freeB], config.NumLayers, sharedWeightBytes)[0];
                ditShardBackend = context.DitShardBackend;
                Logs.Info($"[MiniMaxH3Recipe] DiT sharding enabled: blocks [0,{ditShardSplitBlock}) on the primary "
                    + $"backend, [{ditShardSplitBlock},{config.NumLayers}) on the shard backend.");
            }
            else if (context.DitShardBackend is not null)
            {
                Logs.Warning("[MiniMaxH3Recipe] DiT sharding requested but this is the bf16 build (>40 GB) — "
                    + "its blocks exceed any two-consumer-card pool; running the streaming path unsharded.");
            }

            MiniMaxH3Pipeline pipeline =
                new MiniMaxH3Pipeline(context.Backend, transformer, videoVae, audioVae, fitsResidentSingleBackend)
                {
                    DitShardBackend = ditShardBackend,
                    DitShardSplitBlock = ditShardSplitBlock,
                    VaeBackend = context.VaeBackendOrDefault,
                };
            return new MiniMaxH3RecipePipeline(context.Backend, pipeline, config, textEncoder,
                LoadTokenizer(assets.TokenizerDir), loaders, videoVaeEncoder, audioVaeEncoder, loraStack,
                context.TextEncoderBackendOrDefault, context.VaeBackendOrDefault);
        }
        catch
        {
            loraStack?.Dispose();
            foreach (SafeTensorsLoader loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
    }

    /// <summary>Conservative reserve any H3 forward needs beyond its resident weights, independent of the actual
    /// request's sequence length — the smallest chunk's own transient scratch plus the fixed cuBLAS/RoPE fudge
    /// tail (matches <see cref="MiniMaxH3ActivationEstimate"/>'s seq-independent terms). The seq-scaled cost is a
    /// separate, per-request question that <c>MiniMaxH3RecipePipeline.CheckVramFeasibility</c> checks against live
    /// free VRAM before every generation; this only needs to keep the one-time PRELOAD decision honest.</summary>
    private static long MinimumActivationReserveBytes(MiniMaxH3Config config)
    {
        int inner = config.NumAttentionHeads * config.AttentionHeadDim;
        int ffn = config.FfnHiddenSize;
        int chunkRows = MiniMaxH3ChunkPolicy.DefaultChunkRows;
        long attnChunkBytes = (long)chunkRows * inner * 6L * DType.F32.SizeInBytes;
        long mlpChunkBytes = (long)chunkRows * ffn * 3L * DType.F32.SizeInBytes;
        return Math.Max(attnChunkBytes, mlpChunkBytes) + MiniMaxH3ActivationEstimate.FudgeBytes;
    }

    /// <summary>Warns when placement knobs are configured that H3 cannot use, so the operator learns why they saw
    /// no effect instead of silently paying for an unused second CUDA context (<see cref="RecipeContext.AllBackends"/>
    /// still constructs and gates on every configured backend regardless of whether a recipe consumes it).</summary>
    private static void WarnIfPlacementIgnored(RecipeContext context)
    {
        if (context.CpBackends is { Count: > 0 })
        {
            Logs.Warning("[MiniMaxH3Recipe] Context parallelism is configured but not wired for MiniMax-H3, and is "
                + "also hardware-blocked on an asymmetric pair here — the ~21 GB fp8 DiT replica does not fit a "
                + "12 GB card. This generation runs single-GPU.");
        }
        if (context.CfgParallelBackend is not null)
        {
            Logs.Warning("[MiniMaxH3Recipe] CFG-parallel is configured but structurally inapplicable to MiniMax-H3 — "
                + "it runs CfgScale=1.0 as a single forward pass with no unconditional branch to parallelize.");
        }
        if (context.TextEncoderBackend is not null && !ReferenceEquals(context.TextEncoderBackend, context.Backend))
        {
            Logs.Info("[MiniMaxH3Recipe] Text encoder placed off the primary backend — its weights stay resident "
                + "between generations, since nothing on that device competes with the DiT for the space.");
        }
        if (context.VaeBackend is not null && !ReferenceEquals(context.VaeBackend, context.Backend))
        {
            Logs.Info("[MiniMaxH3Recipe] VAE encode/decode placed off the primary backend — its weights stay "
                + "resident between generations, since nothing on that device competes with the DiT for the space.");
        }
    }

    /// <summary>Builds the DiT but leaves its weights unloaded, handing back the converted dict so a LoRA merge can
    /// land on it first — the merge rewrites entries in place, so it has to happen before the transformer reads them.</summary>
    private static MiniMaxH3Transformer LoadTransformer(string file, List<SafeTensorsLoader> loaders,
        DType bodyDType, out MiniMaxH3Config config, out Dictionary<string, Tensor> transformerWeights)
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
        // BF16 residual stream matches the reference body dtype; CPU has F32-only kernels.
        transformerWeights = promotedWeights;
        return new MiniMaxH3Transformer(config) { BodyDType = bodyDType };
    }

    /// <summary>Merges the request's LoRA stack into the DiT weights. Every H3 module a LoRA can target is a Linear
    /// (<c>qkv_proj</c>, <c>out_proj</c>, <c>fc1</c>/<c>fc2</c>, <c>adaln_proj.linear</c>, the patch/condition
    /// projections), so no convolution path is needed. Both builds are supported: the fp8 targets are requantized
    /// per tensor by <see cref="MergedLoraStack"/>, and the int8-convrot build is already rejected at load.</summary>
    private static MergedLoraStack? ApplyLoras(RecipeContext context, Dictionary<string, Tensor> weights)
    {
        IReadOnlyList<LoraResolver.LoraSpec> specs = LoraResolver.Resolve(context.Loras);
        if (specs.Count == 0)
        {
            return null;
        }
        return LoraApplier.BuildAndApply(specs, context.Backend, transformerWeights: weights);
    }

    /// <summary>The decoder, plus the encoder when the file carries its weights — the vendor VAE ships both halves, but
    /// a decode-only repack would leave keyframe and reference conditioning unavailable rather than failing to load.</summary>
    private static (MiniMaxH3VideoVaeDecoder Decoder, MiniMaxH3VideoVaeEncoder? Encoder) LoadVideoVae(
        string file, List<SafeTensorsLoader> loaders)
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
        MiniMaxH3VideoVaeEncoder? encoder = null;
        if (MiniMaxH3VideoVaeConfig.MatchesEncoder(weights))
        {
            encoder = new MiniMaxH3VideoVaeEncoder(config);
            encoder.LoadWeights(weights);
        }
        Logs.Info($"[MiniMaxH3Recipe] Video VAE: {weights.Count} tensors"
            + (encoder is null ? " (decode only — no keyframe or reference conditioning)." : ", encoder included."));
        return (decoder, encoder);
    }

    /// <summary>The decoder, plus the encoder when the file carries its half — reference audio needs the encoder, but a
    /// decode-only build must still generate soundtracks.</summary>
    private static (MiniMaxH3AudioVaeDecoder? Decoder, MiniMaxH3AudioVaeEncoder? Encoder) LoadAudioVae(
        string? file, List<SafeTensorsLoader> loaders)
    {
        if (file is null)
        {
            return (null, null);
        }
        SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(file);
        loaders.Add(loader);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(loader.GetAllTensors());
        MiniMaxH3AudioVaeDecoder decoder = new MiniMaxH3AudioVaeDecoder();
        decoder.LoadWeights(weights);
        MiniMaxH3AudioVaeEncoder? encoder = null;
        if (MiniMaxH3AudioVaeEncoder.Matches(weights))
        {
            encoder = new MiniMaxH3AudioVaeEncoder();
            encoder.LoadWeights(weights);
        }
        Logs.Info($"[MiniMaxH3Recipe] Audio VAE: {weights.Count} tensors @ {decoder.SampleRate} Hz"
            + (encoder is null ? " (decode only — no reference audio)." : ", encoder included."));
        return (decoder, encoder);
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
