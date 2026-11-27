namespace HartsyInference.ThreeD.Paint;

/// <summary>The mesh vertices of one view: the world-to-camera matrix, homogeneous clip positions (four floats per vertex) and camera-space
/// positions (three floats per vertex; the camera looks down -z).</summary>
public sealed record PaintViewProjection(double[] WorldToCamera, float[] Clip, float[] Camera);
