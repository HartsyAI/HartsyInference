using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelAssets.CheckpointConverters;

/// <summary>Splits a Hunyuan-GameCraft checkpoint (loaded from <c>.pt</c> via <c>PytorchPickleLoader</c> or a
/// converted safetensors) into the component weight dicts the pipeline loads: the <c>Dit</c> (HunyuanVideo MM-DiT
/// incl. <c>camera_in</c>… no — camera kept separate), the <c>CameraNet</c>, the <c>Vae</c>, and the
/// <c>Llava</c> + <c>Clip</c> text encoders. Routes by coarse prefix.
/// <para><b>Numerics validation-pending</b> — the per-block key remap from the original HunyuanVideo naming
/// (<c>double_blocks.*.img_attn_qkv</c>, <c>single_blocks.*.linear1</c>, …) to the diffusers names the reused
/// <c>HunyuanImageBlock</c>/<c>HunyuanImageSingleBlock</c> expect is <b>validation-gated</b> and finalized during
/// the diff pass; this router classifies by prefix so the structural pipeline can load a converted checkpoint.</para></summary>
public static class HunyuanGameCraftCheckpointConverter
{
    public sealed class ConvertedWeights
    {
        public required Dictionary<string, Tensor> Dit { get; init; }
        public required Dictionary<string, Tensor> CameraNet { get; init; }
        public required Dictionary<string, Tensor> Vae { get; init; }
        public required Dictionary<string, Tensor> Llava { get; init; }
        public required Dictionary<string, Tensor> Clip { get; init; }
    }

    private static readonly string[] VaePrefixes = ["vae.", "first_stage_model.", "884-16c-hy0801."];
    private static readonly string[] LlavaPrefixes = ["text_encoder.", "llava.", "language_model.", "llm."];
    private static readonly string[] ClipPrefixes = ["text_encoder_2.", "clip.", "clip_l."];

    public static ConvertedWeights Convert(IReadOnlyDictionary<string, Tensor> all)
    {
        ArgumentNullException.ThrowIfNull(all);
        Dictionary<string, Tensor> dit = [], cam = [], vae = [], llava = [], clip = [];
        foreach ((string key, Tensor t) in all)
        {
            if (key.StartsWith("camera_in.", StringComparison.Ordinal)) { cam[key] = t; continue; } // CameraNet keeps its prefix
            if (TryRoute(key, t, VaePrefixes, vae)) continue;
            if (TryRoute(key, t, LlavaPrefixes, llava)) continue;
            if (TryRoute(key, t, ClipPrefixes, clip)) continue;
            dit[key] = t; // DiT keeps its native key (block remap is applied during the diff pass)
        }
        return new ConvertedWeights { Dit = dit, CameraNet = cam, Vae = vae, Llava = llava, Clip = clip };
    }

    private static bool TryRoute(string key, Tensor t, string[] prefixes, Dictionary<string, Tensor> dst)
    {
        foreach (string p in prefixes)
            if (key.StartsWith(p, StringComparison.Ordinal)) { dst[key[p.Length..]] = t; return true; }
        return false;
    }
}
