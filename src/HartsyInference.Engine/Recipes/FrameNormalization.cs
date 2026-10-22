namespace HartsyInference.Engine.Recipes;

/// <summary>What planning does with a requested frame count that is not on a family's <see cref="FrameGrid"/>.</summary>
public enum FrameNormalization
{
    /// <summary>Refuse the request, naming the nearest valid counts.</summary>
    Reject,

    /// <summary>Move to the nearest count on the grid, recorded as a normalization; a tie goes to the even multiple of
    /// the step, as <see cref="Math.Round(double)"/> rounds.</summary>
    SnapNearest,

    /// <summary>Move to the next count on the grid at or above the request, recorded as a normalization.</summary>
    SnapUp,
}
