namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>How a Drama 3 checkpoint stores its weights (<c>quantization_config.quant_method</c>).</summary>
public enum FishAudioDrama3Quantization
{
    /// <summary>No quantization config: every weight in its float dtype (the unquantized original).</summary>
    None,

    /// <summary><c>bsgemm-moe</c>: routed experts in F8 E4M3 with 128×128 F32 block scales, everything else BF16 (<c>fishaudio/drama-3</c>).</summary>
    Fp8Block,

    /// <summary><c>drama3-int4</c>: routed experts and the listed attention projections in symmetric 4-bit GPTQ, group 128
    /// (<c>fishaudio/drama-3-int4</c>).</summary>
    Int4Gptq,
}
