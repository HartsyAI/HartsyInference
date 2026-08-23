namespace HartsyInference.Core.Rope;

/// <summary>RoPE inverse-frequency scaling strategy for long-context extrapolation, shared by the LLM decoder and diffusion text encoders.</summary>
/// <remarks>Shared by the LLM decoder (<c>GenericTransformer</c>) and the diffusion text encoders (GPT-OSS YaRN,
/// Cosmos/Anima NTK). Only the inverse-frequency transform differs per type; the cos/sin <i>layout</i> stays each
/// model's own.
///
/// <para><see cref="None"/> = standard RoPE (Qwen/Mistral/Gemma). <see cref="Linear"/> = position-interpolation
/// (divide all frequencies by factor). <see cref="Llama3"/> = Llama-3.1+ piecewise wavelength rescale.
/// <see cref="Yarn"/> = NTK-by-parts blend + mscale (GPT-OSS, Llama-3.1-extended, DeepSeek). <see cref="DynamicNtk"/>
/// = base rescaled by the live sequence length. <see cref="LongRope"/> = per-dimension factor table (Phi-3).</para></remarks>
public enum RopeScalingType
{
    /// <summary>Standard RoPE: inv_freq[k] = 1 / theta^(2k/D).</summary>
    None,

    /// <summary>Linear position interpolation: inv_freq /= factor.</summary>
    Linear,

    /// <summary>Llama-3 piecewise wavelength rescale (low/high freq factors over the original context).</summary>
    Llama3,

    /// <summary>YaRN: blends extrapolation (high freq, kept) and interpolation (low freq, /factor) over a dimension-index ramp, plus an attention mscale baked into cos/sin.</summary>
    Yarn,

    /// <summary>Dynamic NTK: the RoPE base is rescaled by the current sequence length past the original context.</summary>
    DynamicNtk,

    /// <summary>LongRope (Phi-3): a per-dimension frequency-factor table (short below the original context, long above).</summary>
    LongRope,
}
