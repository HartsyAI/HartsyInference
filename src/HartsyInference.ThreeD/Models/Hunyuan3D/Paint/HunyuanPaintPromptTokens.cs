using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>The learned text tokens baked into the UNet: one block per generated material and one for the reference stream, each <c>[1, tokens, dim]</c>.</summary>
public sealed class HunyuanPaintPromptTokens
{
    /// <summary>Creates the token set.</summary>
    public HunyuanPaintPromptTokens(Tensor[] perMaterial, Tensor reference)
    {
        PerMaterial = perMaterial ?? throw new ArgumentNullException(nameof(perMaterial));
        Reference = reference ?? throw new ArgumentNullException(nameof(reference));
    }

    /// <summary>Tokens per material, in the configuration's order.</summary>
    public Tensor[] PerMaterial { get; }

    /// <summary>Tokens of the reference stream.</summary>
    public Tensor Reference { get; }

    /// <summary>Reads the tokens from the UNet's tensors: <c>learned_text_clip_{material}</c> for 2.1, <c>learned_text_clip_gen</c> for 2.0, and <c>learned_text_clip_ref</c> either way.</summary>
    public static HunyuanPaintPromptTokens FromWeights(IReadOnlyDictionary<string, Tensor> weights, HunyuanPaintConfig config, string prefix = "")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        Tensor[] perMaterial = new Tensor[config.MaterialCount];
        for (int m = 0; m < perMaterial.Length; m++)
            perMaterial[m] = Shape(Hunyuan3DShapeVae.F32(weights[config.MaterialCount == 1 && weights.ContainsKey($"{p}learned_text_clip_gen")
                ? $"{p}learned_text_clip_gen" : $"{p}learned_text_clip_{config.Materials[m]}"]), config);
        return new HunyuanPaintPromptTokens(perMaterial, Shape(Hunyuan3DShapeVae.F32(weights[$"{p}learned_text_clip_ref"]), config));
    }

    private static Tensor Shape(Tensor t, HunyuanPaintConfig config) => t.Reshape(new TensorShape(1, config.TextTokens, config.CrossAttentionDim));
}
