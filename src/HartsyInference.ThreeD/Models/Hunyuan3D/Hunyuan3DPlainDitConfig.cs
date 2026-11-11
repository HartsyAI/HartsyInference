namespace HartsyInference.ThreeD.Models.Hunyuan3D;

/// <summary>Dimensions that only the Hunyuan3D-2.1 <c>HunYuanDiTPlain</c> denoiser has: its depth and mixture-of-experts layout.</summary>
/// <param name="Depth">Number of transformer blocks; blocks after the middle take U-skips.</param>
/// <param name="MoeBlocks">How many of the last blocks use a mixture of experts instead of a dense MLP.</param>
/// <param name="Experts">Routed experts per mixture block.</param>
/// <param name="TopK">Experts each token is routed to.</param>
public sealed record Hunyuan3DPlainDitConfig(int Depth, int MoeBlocks, int Experts, int TopK);
