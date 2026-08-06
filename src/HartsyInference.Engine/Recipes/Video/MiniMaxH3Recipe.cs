using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
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
        MergedLoraStack? loraStack = null;
        try
        {
            {
                // Only the 66 GB bf16 DiT needs this — caching its F16 casts grows without bound and OOMs the host.
                // The 21 GB fp8 build must keep the cache ON or every GEMM re-uploads its weight.
                long ditBytes = new FileInfo(assets.Dit).Length;
                if (ditBytes > 40L << 30)
                {
                    RecipeBackendFlags.DisableCacheWeightCasts(context, "MiniMaxH3Recipe");
                }
            }
            // F32, and F16 is not a bug to chase: this DiT's stream genuinely leaves F16 range on real weights —
            // condition_proj already emits 82740 (2 of 5376 text channels overflow to inf before block 0) and the
            // residual reaches 2.7e6 by the last block. BF16 holds the range but falls off the native fp8 GEMM
            // guard (CudaBackend.cs:959 takes fp8/F32/F16 inputs only) for a 46% slowdown. The 2-layer parity
            // test passes both, so it does not gate this — only a real-weight run does.
            DType bodyDType = DType.F32;
            MiniMaxH3Transformer transformer = LoadTransformer(assets.Dit, loaders, bodyDType,
                out MiniMaxH3Config config, out Dictionary<string, Tensor> ditWeights);
            loraStack = ApplyLoras(context, ditWeights);
            transformer.LoadWeights(ditWeights);
            (MiniMaxH3VideoVaeDecoder videoVae, MiniMaxH3VideoVaeEncoder? videoVaeEncoder) =
                LoadVideoVae(assets.VideoVae, loaders);
            (MiniMaxH3AudioVaeDecoder? audioVae, MiniMaxH3AudioVaeEncoder? audioVaeEncoder) =
                LoadAudioVae(assets.AudioVae, loaders);

            MiniMaxH3TextEncoder textEncoder = LoadTextEncoder(assets.TextEncoder, loaders);
            // The 66 GB bf16 build cannot stay device-resident on any consumer card; the 21 GB fp8 build can, and
            // must, or every GEMM re-uploads its weight.
            bool fitsResident = new FileInfo(assets.Dit).Length < 40L << 30;

            // DiT sharding: fp8 build only — the bf16 blocks alone (~64 GB) exceed any 2-consumer-card pool, so
            // sharding buys nothing there and the streaming path stands. 50 homogeneous blocks → the
            // count-proportional plan is byte-accurate.
            int ditShardSplitBlock = 0;
            IBackend? ditShardBackend = null;
            if (context.DitShardBackend is not null && fitsResident)
            {
                long sharedWeightBytes = 0;
                foreach (Tensor t in transformer.EnumerateSharedWeights())
                {
                    sharedWeightBytes += t.DType.ComputeByteCount(t.ElementCount);
                }
                (long freeA, _) = context.Backend.GetVramInfo();
                (long freeB, _) = context.DitShardBackend.GetVramInfo();
                ditShardSplitBlock = PlacementPlanner.DitSplitPlan(freeA, freeB, config.NumLayers, sharedWeightBytes);
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
                new MiniMaxH3Pipeline(context.Backend, transformer, videoVae, audioVae, fitsResident)
                {
                    DitShardBackend = ditShardBackend,
                    DitShardSplitBlock = ditShardSplitBlock,
                };
            return new MiniMaxH3RecipePipeline(context.Backend, pipeline, config, textEncoder,
                LoadTokenizer(assets.TokenizerDir), loaders, videoVaeEncoder, audioVaeEncoder, loraStack);
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
    /// projections), so no convolution path is needed.</summary>
    private static MergedLoraStack? ApplyLoras(RecipeContext context, Dictionary<string, Tensor> weights)
    {
        IReadOnlyList<LoraResolver.LoraSpec> specs = LoraResolver.Resolve(context.Loras);
        if (specs.Count == 0)
        {
            return null;
        }
        // Merging rewrites a base weight in place, which a quantized tensor cannot represent — the delta would have to
        // be re-quantized per tensor. Caught here so the failure names the checkpoint rather than surfacing from
        // inside the merge as an opaque scale-factor complaint.
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            if (kv.Value.DType.IsFp8 || Math.Abs(kv.Value.Fp8ScaleFactor - 1.0f) > 1e-6f)
            {
                throw new UnsupportedModelException(
                    "MiniMax-H3 LoRA needs the bf16 DiT: the pruned fp8 build stores quantized weights, and a LoRA "
                    + "merge has to rewrite them in full precision. Load 'minimax_h3_*_bf16.safetensors' to use LoRAs.");
            }
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
