namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>The two shape flows: the 512 model on a 32 latent grid, and the 1024 model that refines it.</summary>
public enum Trellis2ShapeFlowKind
{
    Res512,
    Res1024,
}
