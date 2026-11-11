namespace HartsyInference.ThreeD.Pipelines.Requests;

/// <summary>One conditioning image of a multiview request: interleaved RGB24, already recentred and composited on white.</summary>
public readonly record struct Hunyuan3DViewImage(Hunyuan3DView View, byte[] Rgb, int Width, int Height);
