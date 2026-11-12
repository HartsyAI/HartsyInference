using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Engine.Recipes;

/// <summary>Component loaders shared by the MiniMax-H3 video and image recipes: the DiT, the Qwen text encoder and its
/// tokenizer. Modality-specific loaders (video and audio VAEs, Fun ControlNet, PDD adapters) stay in their recipe.</summary>
internal static class MiniMaxH3Components
{
    /// <summary>Builds the DiT but leaves its weights unloaded, handing back the converted dict so a LoRA merge can land on it first — the merge rewrites entries in place, so it has to happen before the transformer reads them.</summary>
    internal static MiniMaxH3Transformer LoadTransformer(string file, List<IDisposable> loaders,
        DType bodyDType, IEnumerable<IBackend> backends, out MiniMaxH3Config config,
        out Dictionary<string, Tensor> transformerWeights)
    {
        // An NVFP4 build stays packed: its F16 unpack (66 GB) fits neither a card nor most hosts, so there is no eager
        // twin to stand in for, and the unpack picks BF16 for the F32 stream as the fp8 build's GEMMs do.
        CheckpointSource source = CheckpointSource.Open(file,
            CheckpointOpenOptions.ForResidentNvfp4(backends) with { ResidentNvfp4DenseDType = null });
        loaders.Add(source);
        Dictionary<string, Tensor> raw = new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
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
            Tensor current = promotedWeights[key];
            if (isNorm && current.DType != DType.F32)
            {
                // A GGUF build can carry a quantized norm, and CastTo refuses a quantized source by design —
                // decoding a block layout is the dequantizer's job, not a dtype conversion's.
                promotedWeights[key] = current.DType.IsQuantized
                    ? GgufDequantizer.Dequantize(current, DType.F32)
                    : current.CastTo(DType.F32);
                promoted++;
            }
        }
        Logs.Info($"[MiniMaxH3Recipe] Promoted {promoted} norm/rope tensors to F32.");
        config = MiniMaxH3Config.Detect(promotedWeights, source.Header.Metadata);
        Logs.Info($"[MiniMaxH3Recipe] DiT: {converted.Transformer.Count} tensors, {config.NumLayers} blocks, "
            + $"hidden {config.HiddenSize}, curves={config.UseAdalnCurves}.");
        // BF16 residual stream matches the reference body dtype; CPU has F32-only kernels.
        transformerWeights = promotedWeights;
        return new MiniMaxH3Transformer(config) { BodyDType = bodyDType };
    }

    internal static MiniMaxH3TextEncoder LoadTextEncoder(string file, List<IDisposable> loaders)
    {
        // The encoder binds every projection as Nvfp4Linear, which keeps the weight packed and dequantizes one slice
        // at a time inside the forward — the reason a 15.7 GB nvfp4 checkpoint costs a mmap here and not its widened
        // size. That layer reads `.weight_scale`/`.weight_scale_2` as KEYS, so folding them away leaves it nothing to
        // read and the pass widens the weight instead, which is how the published qwen3vl_32b nvfp4-AWQ encoder
        // started materializing in full and taking the DiT's headroom with it. Nothing renames these keys between
        // here and LoadWeights, so the hazard folding exists to prevent cannot arise. int8 and fp8 builds are
        // unaffected: their companions still fold onto QuantInfo, which is where this encoder reads them.
        CheckpointSource source = CheckpointSource.Open(file,
            new CheckpointOpenOptions { KeepNvfp4Companions = true });
        loaders.Add(source);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>(source.Weights, StringComparer.Ordinal);
        MiniMaxH3TextEncoder encoder = new MiniMaxH3TextEncoder();
        encoder.LoadWeights(weights);
        Logs.Info($"[MiniMaxH3Recipe] Text encoder: {weights.Count} tensors.");
        return encoder;
    }

    /// <summary>The flat repack ships no tokenizer files; the embedded Qwen BPE is the same base merge table.</summary>
    internal static Qwen2Tokenizer LoadTokenizer(string? dir)
    {
        if (dir is null)
        {
            Logs.Info("[MiniMaxH3Recipe] Using the embedded Qwen tokenizer (no vocab.json/merges.txt alongside the checkpoint).");
            return new Qwen2Tokenizer();
        }
        return new Qwen2Tokenizer(Path.Combine(dir, "vocab.json"), Path.Combine(dir, "merges.txt"));
    }
}
