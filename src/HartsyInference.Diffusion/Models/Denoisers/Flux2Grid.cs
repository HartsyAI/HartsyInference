namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>One image grid of a multi-grid <see cref="Flux2Transformer.ForwardGrids"/> call.</summary>
/// <param name="Sigma">The grid's noise level in [0, 1]: 0 for a clean (given) grid, 1 for pure noise.</param>
/// <param name="Modality">Row of the modality embedding its tokens take.</param>
/// <param name="TimeCoordinate">The grid's RoPE T coordinate.</param>
public readonly record struct Flux2Grid(float Sigma, int Modality, float TimeCoordinate);
