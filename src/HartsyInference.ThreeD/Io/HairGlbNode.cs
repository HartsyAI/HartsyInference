using HartsyInference.ThreeD.Geometry;

namespace HartsyInference.ThreeD.Io;

/// <summary>A named scene-root node holding one mesh with its material, as <c>HairCardGlbWriter</c> writes it. A non-null
/// <paramref name="Skin"/> makes the mesh skinned to the rig passed to the writer.</summary>
public sealed record HairGlbNode(string Name, Mesh Mesh, HairGlbMaterial Material, HairGlbSkin? Skin = null);
