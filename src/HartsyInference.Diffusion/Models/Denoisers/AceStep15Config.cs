namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Configuration for ACE-Step v1.5 turbo (2B) — the Qwen3-block flow-matching music DiT over Oobleck-VAE
/// latents (25 Hz, 64-ch, 48 kHz stereo). Defaults verbatim from <c>configuration_acestep_v15.py</c>; the fixed
/// 8-step turbo timestep tables come from the shipped <c>SHIFT_TIMESTEPS</c>. See
/// <c>docs/Research/ACE_STEP_15_INFERENCE.md</c>.</summary>
public sealed record AceStep15Config
{
    /// <summary>Model width (= heads · head dim).</summary>
    public int HiddenSize { get; init; } = 2048;

    /// <summary>DiT layers.</summary>
    public int NumLayers { get; init; } = 24;

    /// <summary>Query heads.</summary>
    public int NumHeads { get; init; } = 16;

    /// <summary>Key/value heads (GQA 16:8).</summary>
    public int NumKvHeads { get; init; } = 8;

    /// <summary>Per-head dim.</summary>
    public int HeadDim { get; init; } = 128;

    /// <summary>SwiGLU MLP inner dim.</summary>
    public int IntermediateSize { get; init; } = 6144;

    /// <summary>RMSNorm epsilon.</summary>
    public float RmsNormEps { get; init; } = 1e-6f;

    /// <summary>RoPE base.</summary>
    public double RopeTheta { get; init; } = 1_000_000.0;

    /// <summary>Bidirectional sliding-attention window (token distance) on alternating layers.</summary>
    public int SlidingWindow { get; init; } = 128;

    /// <summary>DiT input channels: latent · 3 (src ‖ chunk mask ‖ noisy latent — reference concat order).</summary>
    public int InChannels { get; init; } = 192;

    /// <summary>Conv1d patchify kernel/stride (25 Hz latents → 12.5 Hz tokens).</summary>
    public int PatchSize { get; init; } = 2;

    /// <summary>Oobleck latent channels (DiT output channels).</summary>
    public int LatentChannels { get; init; } = 64;

    /// <summary>Qwen3-Embedding-0.6B hidden width (text AND lyric features).</summary>
    public int TextHiddenDim { get; init; } = 1024;

    /// <summary>Reference-audio acoustic latent width (Oobleck latents).</summary>
    public int TimbreHiddenDim { get; init; } = 64;

    /// <summary>Lyric encoder depth.</summary>
    public int LyricEncoderLayers { get; init; } = 8;

    /// <summary>Timbre encoder depth.</summary>
    public int TimbreEncoderLayers { get; init; } = 4;

    /// <summary>Timestep sinusoid width (cos-first, t scaled ×1000).</summary>
    public int FreqDim { get; init; } = 256;

    /// <summary>Latent frames per second.</summary>
    public int LatentRate { get; init; } = 25;

    /// <summary>Output sample rate.</summary>
    public int SampleRate { get; init; } = 48_000;

    /// <summary>PCM samples per latent frame (Oobleck 2·4·4·6·10).</summary>
    public int SamplesPerLatent { get; init; } = 1920;

    /// <summary>Turbo step count (fixed table, no CFG).</summary>
    public int NumInferenceSteps { get; init; } = 8;

    /// <summary>Default timestep-table shift.</summary>
    public float FlowShift { get; init; } = 3.0f;

    /// <summary>True for distilled turbo checkpoints (8-step, guidance baked in — CFG must stay off);
    /// false for base/sft (continuous schedule + CFG). From the checkpoint's <c>is_turbo</c>.</summary>
    public bool IsTurbo { get; init; } = true;

    /// <summary>Condition-encoder width (<c>encoder_hidden_size</c>). 0 = same as the decoder (all 2B
    /// checkpoints); XL keeps a 2048-wide encoder under a 2560-wide decoder.</summary>
    public int EncoderHiddenSize { get; init; }

    /// <summary>Condition-encoder SwiGLU inner dim (<c>encoder_intermediate_size</c>); 0 = decoder's.</summary>
    public int EncoderIntermediateSize { get; init; }

    /// <summary>Condition-encoder query heads (<c>encoder_num_attention_heads</c>); 0 = decoder's.</summary>
    public int EncoderNumHeads { get; init; }

    /// <summary>Condition-encoder key/value heads (<c>encoder_num_key_value_heads</c>); 0 = decoder's.</summary>
    public int EncoderNumKvHeads { get; init; }

    /// <summary>True for layers using the 128-token sliding window: even indices (the reference computes
    /// <c>"sliding_attention" if (i + 1) % 2 else "full_attention"</c>, so layer 0 slides, layer 1 is full, …).
    /// Shared by the DiT and both condition encoders.</summary>
    public bool IsSlidingLayer(int layerIndex) => (layerIndex + 1) % 2 == 1;

    /// <summary>Latent frames for a duration at 25 Hz.</summary>
    public int LatentFrames(double durationSeconds) => (int)Math.Round(durationSeconds * LatentRate);

    /// <summary>The exact frame count the pipeline generates for a duration (patch-aligned, floor 4·patch) —
    /// lmHints must match this length.</summary>
    public int FrameCount(double durationSeconds)
        => Math.Max(4 * PatchSize, (LatentFrames(durationSeconds) + PatchSize - 1) / PatchSize * PatchSize);

    /// <summary>The fixed turbo timestep table (8 descending values; the final step integrates to 0). The shift is
    /// snapped to the nearest of the reference's <c>VALID_SHIFTS</c> [1, 2, 3]. These tables are exactly
    /// <see cref="GetTimesteps(int,float,bool)"/> at 8 steps.</summary>
    public static float[] GetTimesteps(float shift) => GetTimesteps(8, shift, snapShift: true);

    /// <summary>The shared flow schedule: <c>t = linspace(1,0,steps+1)</c> then, for shift ≠ 1,
    /// <c>t = shift·t / (1 + (shift−1)·t)</c>. Returns the first <paramref name="steps"/> values (the final
    /// step integrates to the implicit trailing 0). Turbo snaps shift to the trained {1,2,3}
    /// (<paramref name="snapShift"/>); base/sft/continuous accept arbitrary shift.</summary>
    public static float[] GetTimesteps(int steps, float shift, bool snapShift = false)
    {
        if (steps < 1) throw new ArgumentOutOfRangeException(nameof(steps));
        if (snapShift) shift = shift < 1.5f ? 1f : shift < 2.5f ? 2f : 3f;
        float[] ts = new float[steps];
        for (int k = 0; k < steps; k++)
        {
            double t = 1.0 - k / (double)steps;
            if (shift != 1f) t = shift * t / (1.0 + (shift - 1.0) * t);
            ts[k] = (float)t;
        }
        return ts;
    }

    /// <summary>The published v1.5 turbo 2B model.</summary>
    public static AceStep15Config Turbo => new();

    /// <summary>The condition-side config (upstream's <c>encoder_config = deepcopy(config)</c> with the
    /// <c>encoder_*</c> dims): the condition encoder, tokenizer/detokenizer, and null_condition_emb all run at
    /// encoder width. Identity for 2B checkpoints; XL keeps a 2048-wide encoder under a 2560-wide decoder.</summary>
    public AceStep15Config EncoderVariant() => this with
    {
        HiddenSize = EncoderHiddenSize > 0 ? EncoderHiddenSize : HiddenSize,
        IntermediateSize = EncoderIntermediateSize > 0 ? EncoderIntermediateSize : IntermediateSize,
        NumHeads = EncoderNumHeads > 0 ? EncoderNumHeads : NumHeads,
        NumKvHeads = EncoderNumKvHeads > 0 ? EncoderNumKvHeads : NumKvHeads,
        EncoderHiddenSize = 0, EncoderIntermediateSize = 0, EncoderNumHeads = 0, EncoderNumKvHeads = 0,
    };

    /// <summary>Builds a config from a checkpoint's <c>config.json</c> (upstream
    /// <c>configuration_acestep_v15.py</c> keys). Missing keys keep the 2B turbo defaults; XL's
    /// <c>encoder_*</c> keys override the encoder dims independently of the decoder.</summary>
    public static AceStep15Config FromJson(string configJson)
    {
        using System.Text.Json.JsonDocument doc = System.Text.Json.JsonDocument.Parse(configJson);
        System.Text.Json.JsonElement r = doc.RootElement;
        int I(string key, int dflt) => r.TryGetProperty(key, out System.Text.Json.JsonElement e) && e.ValueKind == System.Text.Json.JsonValueKind.Number ? e.GetInt32() : dflt;
        float F(string key, float dflt) => r.TryGetProperty(key, out System.Text.Json.JsonElement e) && e.ValueKind == System.Text.Json.JsonValueKind.Number ? (float)e.GetDouble() : dflt;
        bool B(string key, bool dflt) => r.TryGetProperty(key, out System.Text.Json.JsonElement e) && (e.ValueKind == System.Text.Json.JsonValueKind.True || e.ValueKind == System.Text.Json.JsonValueKind.False) ? e.GetBoolean() : dflt;

        int hidden = I("hidden_size", 2048);
        int inter = I("intermediate_size", 6144);
        int heads = I("num_attention_heads", 16);
        int kvHeads = I("num_key_value_heads", 8);
        return new AceStep15Config
        {
            HiddenSize = hidden,
            NumLayers = I("num_hidden_layers", 24),
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = I("head_dim", 128),
            IntermediateSize = inter,
            RmsNormEps = F("rms_norm_eps", 1e-6f),
            RopeTheta = r.TryGetProperty("rope_theta", out System.Text.Json.JsonElement rt) && rt.ValueKind == System.Text.Json.JsonValueKind.Number ? rt.GetDouble() : 1_000_000.0,
            SlidingWindow = I("sliding_window", 128),
            InChannels = I("in_channels", 192),
            PatchSize = I("patch_size", 2),
            LatentChannels = I("audio_acoustic_hidden_dim", 64),
            TextHiddenDim = I("text_hidden_dim", 1024),
            TimbreHiddenDim = I("timbre_hidden_dim", 64),
            LyricEncoderLayers = I("num_lyric_encoder_hidden_layers", 8),
            TimbreEncoderLayers = I("num_timbre_encoder_hidden_layers", 4),
            IsTurbo = B("is_turbo", true),
            // 2B checkpoints have no encoder_* keys (encoder == decoder dims); XL sets them explicitly.
            EncoderHiddenSize = I("encoder_hidden_size", hidden),
            EncoderIntermediateSize = I("encoder_intermediate_size", inter),
            EncoderNumHeads = I("encoder_num_attention_heads", heads),
            EncoderNumKvHeads = I("encoder_num_key_value_heads", kvHeads),
        };
    }
}
