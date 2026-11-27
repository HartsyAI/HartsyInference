namespace HartsyInference.ThreeD.Geometry.Ops;

/// <summary>
/// Bakes a sparse attribute volume into UV-space float images: rasterize the UV-mapped mesh, interpolate each texel's 3D position,
/// optionally snap it to the closest point on a (typically higher-resolution) source surface, then sample the volume there. This is the
/// pure-C# counterpart of the upstream nvdiffrast + BVH + grid_sample_3d bake.
/// </summary>
public static class TextureBaker
{
    /// <summary>Bakes <paramref name="volume"/> onto the UV-mapped <paramref name="uvMesh"/> at the given texture size.</summary>
    /// <param name="uvMesh">Mesh whose UVs define the atlas.</param>
    /// <param name="sourceBvh">Surface to snap texel positions onto, or null to sample at the mesh's own surface.</param>
    /// <param name="volume">Attribute volume to sample.</param>
    /// <param name="width">Texture width.</param>
    /// <param name="height">Texture height.</param>
    /// <param name="cancellationToken">Cancels between rows.</param>
    /// <returns>Channel image; texels the rasterizer missed, or whose eight volume neighbors were all empty, are uncovered.</returns>
    public static BakedAttributes Bake(Mesh uvMesh, TriangleBvh? sourceBvh, SparseAttributeVolume volume, int width, int height,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(uvMesh);
        ArgumentNullException.ThrowIfNull(volume);
        if (uvMesh.Uvs is null || uvMesh.Uvs.Length != uvMesh.VertexCount * 2)
            throw new ArgumentException("The mesh needs one UV per vertex.", nameof(uvMesh));

        RasterizedUvMap raster = Rasterizer.Rasterize(uvMesh.Uvs, uvMesh.Indices, width, height, cancellationToken);
        BakedAttributes result = new(width, height, volume.Channels);
        float[] verts = uvMesh.Vertices;
        int[] idx = uvMesh.Indices;
        int channels = volume.Channels;
        ParallelOptions options = new() { CancellationToken = cancellationToken };
        Parallel.For(0, height, options, y =>
        {
            for (int x = 0; x < width; x++)
            {
                int texel = y * width + x;
                int tri = raster.TriangleIds[texel];
                if (tri < 0)
                    continue;
                double b1 = raster.Barycentric[texel * 2], b2 = raster.Barycentric[texel * 2 + 1], b0 = 1.0 - b1 - b2;
                int a = idx[tri * 3] * 3, b = idx[tri * 3 + 1] * 3, c = idx[tri * 3 + 2] * 3;
                double px = b0 * verts[a] + b1 * verts[b] + b2 * verts[c];
                double py = b0 * verts[a + 1] + b1 * verts[b + 1] + b2 * verts[c + 1];
                double pz = b0 * verts[a + 2] + b1 * verts[b + 2] + b2 * verts[c + 2];
                if (sourceBvh is not null)
                {
                    BvhHit hit = sourceBvh.ClosestPoint(px, py, pz);
                    px = hit.X; py = hit.Y; pz = hit.Z;
                }
                Span<float> dst = result.Data.AsSpan(texel * channels, channels);
                result.Covered[texel] = SparseAttributeSampler.SampleWorld(volume, px, py, pz, dst);
            }
        });
        return result;
    }
}
