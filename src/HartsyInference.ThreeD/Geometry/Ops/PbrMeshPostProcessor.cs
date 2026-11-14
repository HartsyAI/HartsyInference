using HartsyInference.ThreeD.Io;

namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Model-independent mesh post-processing: fill holes, snapshot the surface, decimate in two cleaned passes, unwrap, bake a sparse
/// PBR attribute volume into UV space, inpaint the gutters and package the result. The pure-C# counterpart of upstream <c>to_glb</c>
/// (standard, non-remesh path); see docs/Research/MESH_POSTPROCESS.md for the differences.
/// </summary>
public static class PbrMeshPostProcessor
{
    /// <summary>Post-processes <paramref name="mesh"/> and bakes <paramref name="volume"/> into a PBR material.</summary>
    /// <param name="mesh">Source triangle mesh in the volume's world space (not modified).</param>
    /// <param name="volume">Sparse attributes with values in [0,1].</param>
    /// <param name="layout">Channel positions inside <paramref name="volume"/>.</param>
    /// <param name="options">Settings; defaults follow upstream.</param>
    /// <param name="cancellationToken">Checked between stages and inside decimation and baking.</param>
    public static PbrPostProcessResult Process(Mesh mesh, SparseAttributeVolume volume, PbrAttributeLayout layout,
        PbrPostProcessOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(mesh);
        ArgumentNullException.ThrowIfNull(volume);
        options ??= new PbrPostProcessOptions();
        layout.Validate(volume.Channels);
        if (options.DecimationTarget < 1 || options.TextureSize < 1)
            throw new ArgumentOutOfRangeException(nameof(options), "Decimation target and texture size must be positive.");
        if (!(options.OpeningWidthRatio >= 0))
            throw new ArgumentOutOfRangeException(nameof(options), "The opening width ratio must not be negative.");

        MeshCleanOptions initial = new()
        {
            SmallComponentAreaRatio = 0,
            HoleFillMaxPerimeter = options.HoleFillMaxPerimeter,
            RepairNonManifoldEdges = false,
            UnifyOrientation = false,
        };
        Mesh source = MeshCleaner.Clean(mesh, initial);
        TriangleBvh bvh = new(source);
        cancellationToken.ThrowIfCancellationRequested();

        MeshCleanOptions between = new()
        {
            SmallComponentAreaRatio = options.SmallComponentAreaRatio,
            HoleFillMaxPerimeter = options.HoleFillMaxPerimeter,
            HoleFillMaxWidth = options.OpeningWidthRatio * ShapeMeshFinisher.Diagonal(mesh),
        };
        Mesh reduced = MeshDecimator.DecimateStaged(source, options.DecimationTarget, options.IntermediateFactor, between,
            cancellationToken: cancellationToken);
        MeshOps.ComputeVertexNormals(reduced);
        cancellationToken.ThrowIfCancellationRequested();

        UvUnwrapOptions unwrapOptions = new()
        {
            ConeHalfAngleDegrees = options.ChartConeHalfAngleDegrees,
            TextureSize = options.TextureSize,
            PaddingTexels = options.PaddingTexels,
        };
        UvUnwrapResult unwrapped = UvUnwrapper.Unwrap(reduced, unwrapOptions, cancellationToken);

        int size = options.TextureSize;
        BakedAttributes baked = TextureBaker.Bake(unwrapped.Mesh, options.SnapToSourceSurface ? bvh : null, volume, size, size,
            cancellationToken);
        PbrMaterial material = BuildMaterial(baked, layout, options);
        int covered = 0;
        foreach (bool c in baked.Covered)
            if (c) covered++;
        return new PbrPostProcessResult(unwrapped.Mesh, material, covered);
    }

    /// <summary>Runs <see cref="Process"/> and serializes the result with <see cref="PbrGlbWriter"/>.</summary>
    public static byte[] ProcessToGlb(Mesh mesh, SparseAttributeVolume volume, PbrAttributeLayout layout,
        PbrPostProcessOptions? options = null, PbrGlbOptions? glbOptions = null, CancellationToken cancellationToken = default)
    {
        PbrPostProcessResult result = Process(mesh, volume, layout, options, cancellationToken);
        return PbrGlbWriter.Write(result.Mesh, result.Material, glbOptions);
    }

    private static PbrMaterial BuildMaterial(BakedAttributes baked, PbrAttributeLayout layout, PbrPostProcessOptions options)
    {
        int w = baked.Width, h = baked.Height, c = baked.Channels;
        float[] color = Inpaint(baked, [layout.BaseColor, layout.BaseColor + 1, layout.BaseColor + 2], options.BaseColorInpaintRadius);
        float[] metallic = Inpaint(baked, [layout.Metallic], options.OtherInpaintRadius);
        float[] roughness = Inpaint(baked, [layout.Roughness], options.OtherInpaintRadius);
        float[]? alpha = layout.Alpha >= 0 ? Inpaint(baked, [layout.Alpha], options.OtherInpaintRadius) : null;
        return new PbrMaterial
        {
            Name = options.MaterialName,
            BaseColor = TextureImage.FromFloats(color, w, h, 3, 0, 3),
            Metallic = TextureImage.FromFloats(metallic, w, h, 1, 0, 1),
            Roughness = TextureImage.FromFloats(roughness, w, h, 1, 0, 1),
            Alpha = alpha is null ? null : TextureImage.FromFloats(alpha, w, h, 1, 0, 1),
        };
    }

    private static float[] Inpaint(BakedAttributes baked, int[] channels, int radius)
    {
        int texels = baked.Width * baked.Height;
        float[] packed = new float[texels * channels.Length];
        for (int i = 0; i < texels; i++)
            for (int k = 0; k < channels.Length; k++)
                packed[i * channels.Length + k] = baked.Data[i * baked.Channels + channels[k]];
        // Bounded by the neighbours: on a sparse atlas (a million tiny charts, 6 to 20 % of the texels covered in places) the unbounded Telea gradient correction ran away across the wide gaps and filled
        // them with black and white stripes (22 % black and 22 % white in the top 200 rows of a TRELLIS.2 bust atlas, none with the bound).
        return Inpainter.Inpaint(packed, baked.Width, baked.Height, channels.Length, baked.Covered, radius, boundByNeighbours: true);
    }
}
