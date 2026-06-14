using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for the HiDream-I1 image transformer (HiDreamImageTransformer2DModel). HiDream pairs 16 double-stream blocks (joint image/text MM-attention) with 32 single-stream blocks (parallel image+text-prefix attention), MoE SwiGLU FFN with shared + 4 routed experts, and a 3-axis RoPE keyed on [layer-id, row, col]. Both Full and Dev share the same architecture; the difference is the distillation training (Dev = guidance-distilled).</summary>
public sealed record HiDreamConfig
{
    /// <summary>Patch size for the latent → token embedding. 2 for HiDream.</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Number of latent channels (16 for HiDream's Flux-style 16-channel VAE).</summary>
    public int InChannels { get; init; } = 16;

    /// <summary>Predicted output channels (= InChannels for velocity prediction).</summary>
    public int OutChannels { get; init; } = 16;

    /// <summary>Number of double-stream (joint MM-attention) transformer blocks.</summary>
    public int NumLayers { get; init; } = 16;

    /// <summary>Number of single-stream (image + text prefix concat) transformer blocks.</summary>
    public int NumSingleLayers { get; init; } = 32;

    /// <summary>Number of attention heads. Always 20 for HiDream-I1.</summary>
    public int NumAttentionHeads { get; init; } = 20;

    /// <summary>Per-head dimension. Always 128 for HiDream-I1. Hidden = NumAttentionHeads * AttentionHeadDim = 2560.</summary>
    public int AttentionHeadDim { get; init; } = 128;

    /// <summary>Inner model dimension (= NumAttentionHeads * AttentionHeadDim).</summary>
    public int InnerDim => NumAttentionHeads * AttentionHeadDim;

    /// <summary>Pooled text embedding dim (concat of CLIP-L 768 + CLIP-G 1280 = 2048).</summary>
    public int TextEmbDim { get; init; } = 2048;

    /// <summary>Per-axis RoPE dimensions. HiDream uses 3 axes (layer-id, row, col) with [64, 32, 32]; sum equals AttentionHeadDim (128).</summary>
    public int[] AxesDimsRope { get; init; } = [64, 32, 32];

    /// <summary>Maximum patchified resolution as (h_patches, w_patches). Drives `max_seq` precomputation. Default [128, 128] = 16384 max image tokens, fitting up to 2048×2048 images at patch_size=2.</summary>
    public int[] MaxResolution { get; init; } = [128, 128];

    /// <summary>Caption channel dims, one per caption projection. HiDream uses [4096, 4096] = T5-XXL feature dim + Llama hidden. Each Llama layer + the T5 representation are projected through these (T5 uses the last entry; Llama layers use the first entry).</summary>
    public int[] CaptionChannels { get; init; } = [4096, 4096];

    /// <summary>Llama-3.1 hidden-state layer indices to harvest. The pipeline collects `output_hidden_states` from Llama and feeds the entries indexed here into successive transformer blocks. The default mirrors the diffusers config: layers 0..31 (one per encoder layer) followed by 16 copies of layer 31 — total 48 entries, matching NumLayers + NumSingleLayers.</summary>
    public int[] LlamaLayers { get; init; } = BuildDefaultLlamaLayers();

    /// <summary>Number of routed experts in the MoE FFN of the image stream's double-block FFN. Default 4 per the HF config.</summary>
    public int NumRoutedExperts { get; init; } = 4;

    /// <summary>Number of activated experts (top-k) during inference. Default 2 of 4.</summary>
    public int NumActivatedExperts { get; init; } = 2;

    /// <summary>RoPE base frequency (theta). 10000 in the diffusers reference.</summary>
    public float RopeTheta { get; init; } = 10000f;

    /// <summary>Default flow-match scheduler shift used by the HiDream pipeline. The HF pipeline computes a dynamic shift from the patchified sequence length; this is the static fallback for low-res t2i and matches the Full-quality reference.</summary>
    public float SchedulerShift { get; init; } = 3.0f;

    /// <summary>RMSNorm epsilon used by the per-head Q/K norms inside HiDreamAttention.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>HiDream-I1-Full preset (the public reference). 16 double + 32 single blocks, 20 heads × 128 dim = 2560 hidden, 4 experts (2 active).</summary>
    public static HiDreamConfig Full => new();

    /// <summary>HiDream-I1-Dev preset. Same architecture as Full; the difference is guidance distillation in training. Pipelines may use a lower CFG scale at inference. Architecture-level settings are identical to <see cref="Full"/>.</summary>
    public static HiDreamConfig Dev => new();

    /// <summary>Builds the default Llama-layer index list used by both Full and Dev configs (0..31 then 16 copies of 31). Total length = 48 = NumLayers (16) + NumSingleLayers (32).</summary>
    private static int[] BuildDefaultLlamaLayers()
    {
        int[] layers = new int[48];
        for (int i = 0; i < 32; i++) layers[i] = i;
        for (int i = 32; i < 48; i++) layers[i] = 31;
        return layers;
    }

    /// <summary>Auto-detects the depth (NumLayers + NumSingleLayers) from a converted transformer weight dictionary by scanning <c>double_stream_blocks.{i}.*</c> and <c>single_stream_blocks.{i}.*</c> indices. Returns the public Full preset with detected counts substituted in.</summary>
    public static HiDreamConfig AutoDetect(IReadOnlyDictionary<string, Tensor> transformerWeights)
    {
        int maxDouble = -1;
        int maxSingle = -1;

        foreach (string key in transformerWeights.Keys)
        {
            if (key.StartsWith("double_stream_blocks."))
            {
                string afterPrefix = key["double_stream_blocks.".Length..];
                int dot = afterPrefix.IndexOf('.');
                if (dot < 0) continue;
                if (int.TryParse(afterPrefix[..dot], out int idx) && idx > maxDouble) maxDouble = idx;
            }
            else if (key.StartsWith("single_stream_blocks."))
            {
                string afterPrefix = key["single_stream_blocks.".Length..];
                int dot = afterPrefix.IndexOf('.');
                if (dot < 0) continue;
                if (int.TryParse(afterPrefix[..dot], out int idx) && idx > maxSingle) maxSingle = idx;
            }
        }

        int detectedNumLayers = maxDouble + 1;
        int detectedNumSingle = maxSingle + 1;
        if (detectedNumLayers <= 0) detectedNumLayers = 16;
        if (detectedNumSingle <= 0) detectedNumSingle = 32;

        return new HiDreamConfig
        {
            NumLayers = detectedNumLayers,
            NumSingleLayers = detectedNumSingle,
        };
    }
}
