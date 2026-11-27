namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Which conditioning a Pixal3D flow stage needs: the image encoded at <paramref name="ImageSize"/> pixels, its patch map sampled at each
/// cell of a <paramref name="GridResolution"/> grid (every cell in token order when <paramref name="Coords"/> is null, else the given
/// <c>(batch, x, y, z)</c> voxels), and, when <paramref name="NafSize"/> is positive, also the NAF-upsampled map at that size.</summary>
public readonly record struct Pixal3DConditionSpec(int ImageSize, int GridResolution, int[]? Coords, int NafSize);
