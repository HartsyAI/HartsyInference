using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Quality;

/// <summary>Per-component dtype overrides applied at weight-load time. Captures the dtype each major model component (backbone, text encoder, VAE) is cast to. VAE is locked to FP16 minimum — FP8 VAE produces visible artifacts.</summary>
public sealed record QualityProfile
{
    /// <summary>DiT/UNet backbone dtype. Typical values: F16 / BF16 / F8E4M3 / Q8_0 / Q4_K.</summary>
    public required DType BackboneDType { get; init; }

    /// <summary>Text encoder dtype (T5, CLIP, Llama-style). Typical values: F16 / BF16 / F8E4M3 / Q8_0.</summary>
    public required DType TextEncoderDType { get; init; }

    /// <summary>VAE encoder/decoder dtype. Must be F16/BF16/F32 — FP8/Q8/Q4 VAE rejected.</summary>
    public required DType VaeDType { get; init; }

    /// <summary>Whether to keep weights resident in CPU RAM as well after GPU upload (for fast re-use). False = stream from disk on every load.</summary>
    public bool KeepCpuMirror { get; init; } = false;

    /// <summary>Maps a <see cref="QualityPreset"/> to a concrete profile. Throws on <see cref="QualityPreset.Custom"/>.</summary>
    public static QualityProfile From(QualityPreset preset) => preset switch
    {
        QualityPreset.Maximum => new QualityProfile { BackboneDType = DType.F16, TextEncoderDType = DType.F16, VaeDType = DType.F16 },
        QualityPreset.High => new QualityProfile { BackboneDType = DType.F8E4M3, TextEncoderDType = DType.F16, VaeDType = DType.F16 },
        QualityPreset.Medium => new QualityProfile { BackboneDType = DType.Q8_0, TextEncoderDType = DType.F8E4M3, VaeDType = DType.F16 },
        QualityPreset.Low => new QualityProfile { BackboneDType = DType.Q4_K, TextEncoderDType = DType.Q4_K, VaeDType = DType.F16 },
        QualityPreset.Custom => throw new ArgumentException($"{nameof(QualityPreset.Custom)} requires an explicit {nameof(QualityProfile)} — pass one to the pipeline directly.", nameof(preset)),
        _ => throw new ArgumentOutOfRangeException(nameof(preset), preset, null),
    };

    /// <summary>The default profile when no quality preset is specified. Equivalent to <see cref="QualityPreset.Maximum"/>.</summary>
    public static QualityProfile Default { get; } = From(QualityPreset.Maximum);

    /// <summary>Validates the profile. Rejects FP8/quantized VAE (quality-critical path).</summary>
    public void Validate()
    {
        if (VaeDType.IsFp8 || VaeDType.IsQuantized)
            throw new HartsyInferenceException($"VAE dtype {VaeDType} is not supported. VAE must run at F16/BF16/F32 — FP8/Q8/Q4 VAE produces visible artifacts on smooth gradients.");
        if (!BackboneDType.IsFloatingPoint && !BackboneDType.IsQuantized)
            throw new HartsyInferenceException($"Backbone dtype must be a floating-point or quantized format; got {BackboneDType}.");
        if (!TextEncoderDType.IsFloatingPoint && !TextEncoderDType.IsQuantized)
            throw new HartsyInferenceException($"Text encoder dtype must be a floating-point or quantized format; got {TextEncoderDType}.");
    }
}
