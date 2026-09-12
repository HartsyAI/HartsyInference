using HartsyInference.Audio.Models.LanguageModels.Qwen3;

namespace HartsyInference.Audio.Models.Music;

/// <summary>Configuration for YuE2, M·A·P's AR–NAR Mixture-of-Transformers music model. Despite the name it shares
/// no architecture with <see cref="YueConfig">YuE v1</see>: instead of two LLaMA stages over xcodec RVQ tokens, an
/// autoregressive Qwen3 LM plans an ABC score and emits semantic codec tokens, and a second, separately-weighted
/// Qwen3-geometry transformer flow-matches continuous 64-channel audio latents while attending to the AR model's
/// per-layer KV cache. An Oobleck VAE decodes those latents to 48 kHz stereo. See
/// <c>docs/Research/YUE2_ARCHITECTURE.md</c>.</summary>
/// <remarks><para>Both stacks share one geometry — hidden 2048 / 28L / 16 heads / 8 KV / head_dim 128 / SwiGLU 6144
/// / RoPE θ=1e6 — which is exactly <see cref="Qwen3Config.Talker1_7B"/> widened to the 184,704-entry YuE2 vocab.
/// That identity is load-bearing: the AR cache is handed to the matching NAR layer as an attention prefix, so the
/// two must agree on layer count, KV head count and head_dim.</para>
///
/// <para>The weights we consume are the Comfy-Org repack (<c>yue2_3b_bf16.safetensors</c>), a single file carrying
/// <c>text_encoders.</c> (AR LM + the embedded tokenizer JSON), <c>model.diffusion_model.</c> (NAR) and <c>vae.</c>.
/// Model weights are CC BY-NC 4.0.</para></remarks>
public sealed record Yue2Config
{
    /// <summary>The autoregressive planner/semantic LM. Owns <c>embed_tokens</c> and <c>lm_head</c>.</summary>
    public required Qwen3Config Ar { get; init; }

    /// <summary>The non-autoregressive acoustic transformer. Same geometry, no embedding or output head — its input
    /// is <c>vae2llm</c> of a padded latent chunk and its output is <c>llm2vae</c> of the final hidden state.</summary>
    public required Qwen3Config Nar { get; init; }

    /// <summary>Channels in one audio latent frame.</summary>
    public int LatentDim { get; init; } = 64;

    /// <summary>Width of the sinusoidal timestep embedding feeding <c>time_embedder.mlp.0</c>.</summary>
    public int TimestepEmbedWidth { get; init; } = 256;

    /// <summary>Rows in the stored <c>latent_pos_embed.pe</c> buffer; also the AR context limit.</summary>
    public int MaxLatentFrames { get; init; } = 24_576;

    /// <summary>Flow-matching timestep warp. The release ships 1.0, which makes the warp the identity — the DiT then
    /// sees <c>t ∈ (0,1]</c> directly. Kept explicit because the checkpoint declares it.</summary>
    public float TimestepShift { get; init; } = 1.0f;

    /// <summary>Latent frames per second of audio.</summary>
    public int FrameRateHz { get; init; } = 25;

    /// <summary>Decoded audio sample rate.</summary>
    public int SampleRate { get; init; } = 48_000;

    /// <summary>Decoded audio channel count.</summary>
    public int Channels { get; init; } = 2;

    /// <summary>Midpoint-solver steps for the acoustic ODE; 32 is the release protocol.</summary>
    public int OdeSteps { get; init; } = 32;

    /// <summary>The released 3B checkpoint, verified against the safetensors header of
    /// <c>Comfy-Org/YuE2/checkpoints/yue2_3b_bf16.safetensors</c> and <c>m-a-p/YuE2-3B/config.json</c>.</summary>
    public static Yue2Config V1
    {
        get
        {
            Qwen3Config body = new()
            {
                HiddenSize = 2_048, NumHiddenLayers = 28, NumAttentionHeads = 16, NumKeyValueHeads = 8,
                HeadDim = 128, IntermediateSize = 6_144,
                MaxPositionEmbeddings = Yue2Protocol.Context, RopeTheta = 1_000_000f, RmsNormEps = 1e-6f,
            };
            return new Yue2Config { Ar = body, Nar = body };
        }
    }
}
