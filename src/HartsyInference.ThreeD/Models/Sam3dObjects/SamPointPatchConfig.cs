namespace HartsyInference.ThreeD.Models.Sam3dObjects;

/// <summary>
/// Shape of the pointmap embedder (upstream <c>PointPatchEmbed</c>, <c>embedder/pointmap.py</c>).
/// </summary>
/// <param name="InputSize">The pointmap is resized (nearest) to <c>InputSize x InputSize</c>.</param>
/// <param name="PatchSize">Side of one window; each window becomes one token.</param>
/// <param name="EmbedDim">Token width.</param>
/// <param name="Heads">Heads of the in-window attention block (fixed at 16 upstream).</param>
/// <param name="MlpRatio">MLP ratio of that block (fixed at 2.0 upstream).</param>
/// <param name="Remap">The coordinate remap (<c>remap_output</c>).</param>
public sealed record SamPointPatchConfig(int InputSize, int PatchSize, int EmbedDim, int Heads, double MlpRatio, SamPointRemapType Remap)
{
    /// <summary>The released embedder (<c>ss_generator.yaml</c>: <c>embed_dim</c> 512, <c>input_size</c> 256, <c>patch_size</c> 8, <c>remap_output</c> linear).</summary>
    public static SamPointPatchConfig Released { get; } = new(256, 8, 512, 16, 2.0, SamPointRemapType.Linear);

    /// <summary>Windows per side.</summary>
    public int WindowsPerSide => InputSize / PatchSize;

    /// <summary>Windows in the image, one output token each.</summary>
    public int Windows => WindowsPerSide * WindowsPerSide;

    /// <summary>Tokens in one window's attention: the pooled token plus the pixels.</summary>
    public int WindowTokens => 1 + PatchSize * PatchSize;

    /// <summary>Hidden width of the block's MLP (<c>int(dim · ratio)</c>, as timm's <c>Mlp</c>).</summary>
    public int MlpHidden => (int)(EmbedDim * MlpRatio);

    /// <exception cref="ArgumentException">The configuration is inconsistent.</exception>
    public void Validate()
    {
        if (InputSize <= 0 || PatchSize <= 0 || EmbedDim <= 0 || Heads <= 0 || !(MlpRatio > 0.0))
            throw new ArgumentException("The embedder needs positive sizes.");
        if (InputSize % PatchSize != 0)
            throw new ArgumentException("The input size must be a multiple of the patch size.");
        if (EmbedDim % Heads != 0)
            throw new ArgumentException("The embedding width must divide by the head count.");
    }
}
