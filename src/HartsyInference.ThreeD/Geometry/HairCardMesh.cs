namespace HartsyInference.ThreeD.Geometry;

/// <summary>Hair cards as one mesh with per-vertex UVs and COLOR_0 (linear RGB), plus the linear base colour of its material.</summary>
public sealed record HairCardMesh(Mesh Mesh, float[] BaseColorLinear);
