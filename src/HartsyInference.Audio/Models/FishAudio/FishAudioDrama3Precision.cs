namespace HartsyInference.Audio.Models.FishAudio;

/// <summary>The precision a Drama 3 checkpoint is run at. Only the backbone's matrices change: the embeddings, norms, router, the
/// linear attention's small gate projections, the output head and the whole codebook tower stay as the checkpoint stores them.</summary>
public enum FishAudioDrama3Precision
{
    /// <summary>The checkpoint's own: BF16 as stored, block-FP8 experts as Q8_0, GPTQ int4 weights as Q4_0 (bit-exact).</summary>
    Native,

    /// <summary>Every backbone matrix in BF16 (FP8 and int4 weights dequantized).</summary>
    Bf16,

    /// <summary>Experts and dense projections in Q8_0.</summary>
    Q8_0,

    /// <summary>Experts in Q6_K.</summary>
    Q6_K,

    /// <summary>Experts in Q5_K; attention projections in Q6_K.</summary>
    Q5_K,

    /// <summary>Experts in Q4_K; attention projections in Q6_K.</summary>
    Q4_K,
}
