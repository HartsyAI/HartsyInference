namespace HartsyInference.LLM.DeepSeekV41;

/// <summary>How the host reference model holds the weights it reads on every token.</summary>
public enum DeepSeekV41Residency
{
    /// <summary>As the checkpoint stores them (FP8 with E8M0 block scales, MXFP4, BF16), decoded by row window while used. About a quarter of the memory of <see cref="WidenedF32"/> with the same results.</summary>
    Stored,

    /// <summary>Every weight widened to F32 at load: the original reference path, roughly 31 GiB for dense, embedding and head weights of the official checkpoint plus 140 MB per cached expert.</summary>
    WidenedF32,
}
