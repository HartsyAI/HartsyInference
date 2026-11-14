namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>Settings for <see cref="NarrowBandRemesher"/>, the CPU counterpart of CuMesh's <c>remesh_narrow_band_dc</c> that upstream TRELLIS.2's <c>to_glb(remesh=True)</c> uses.</summary>
public sealed record NarrowBandRemeshOptions
{
    /// <summary>Cells along the longest side of the grid; 32 times a power of two (or a multiple of one: 1536 starts from 48), 32 to 4096. The surface is rebuilt at this resolution.</summary>
    public required int Resolution { get; init; }

    /// <summary>Half the thickness of the shell, in voxels: the zero set is the surface offset by this distance on both sides (upstream 1).</summary>
    public double Band { get; init; } = 1.0;

    /// <summary>How far each vertex moves back towards its closest point on the input, 0 (stay on the offset surface) to 1 (on the input); upstream's function default is 0.9 and its demo app uses 0.</summary>
    public double ProjectBack { get; init; }

    /// <summary>Keeps only the outer sheet of the shell, found by flood-filling the space outside it, which halves the faces and leaves no hidden inner sheet to unwrap. A hole wider than twice the band lets the fill in, and both sheets then stay, as upstream's always are.</summary>
    public bool OuterShellOnly { get; init; }

    /// <summary>Centre of the grid, or null for the centre of the input's bounding box. Upstream passes the centre of the attribute volume.</summary>
    public System.Numerics.Vector3? Center { get; init; }

    /// <summary>Edge length of the grid cube, or null for the input's longest bounding-box side widened by three bands as upstream does. A caller that holds a volume passes the volume's.</summary>
    public double? Scale { get; init; }
}
