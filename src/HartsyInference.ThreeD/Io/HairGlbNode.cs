using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>A named scene-root node holding one mesh with its material, as <c>HairCardGlbWriter</c> writes it.</summary>
public sealed record HairGlbNode(string Name, Mesh Mesh, HairGlbMaterial Material);
