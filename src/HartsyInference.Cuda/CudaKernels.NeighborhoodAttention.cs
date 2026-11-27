using HartsyInference.Core.Backends.Contracts;

namespace HartsyInference.Cuda;

// 3D neighborhood attention (Kernels/attention/neighborhood_attention.cu): the per-query kernel for any head width and the
// query-tiled kernels for head width 64. The module is optional: without it the backend runs the host reference.
public sealed partial class CudaKernels
{
    private CudaModule? _neighborhoodModule;
    private nint _neighborhoodAttentionF32, _neighborhoodTiled8x8F32, _neighborhoodTiled4x8F32;

    private void LoadNeighborhoodAttentionKernels()
    {
        string path = Ptx("neighborhood_attention");
        if (!CanLoad(path)) return;
        _neighborhoodModule = LoadOwnedModule(path);
        _neighborhoodAttentionF32 = _neighborhoodModule.GetFunction("neighborhood_attention_f32");
        _neighborhoodTiled8x8F32 = _neighborhoodModule.GetFunction("neighborhood_attention_tiled8x8_f32");
        _neighborhoodTiled4x8F32 = _neighborhoodModule.GetFunction("neighborhood_attention_tiled4x8_f32");
    }

    /// <summary>Whether the neighborhood attention module loaded; without it the backend runs the host reference.</summary>
    public bool HasNeighborhoodAttentionKernels => _neighborhoodModule is not null;

    /// <summary>Shared-memory bytes the per-query kernel needs: the q row plus one score per window position.</summary>
    public static int NeighborhoodAttentionSharedBytes(in NeighborhoodGeometry g) => (g.HeadDim + g.WindowDepth * g.WindowHeight * g.WindowWidth) * sizeof(float);

    /// <summary>Query-tile height and width the tiled kernels stage a shared window for, or (0, 0) when the shape is not one they
    /// cover: head width 64, and a tile only pays for itself once it holds real queries.</summary>
    public (int H, int W) NeighborhoodAttentionTile(in NeighborhoodGeometry g)
    {
        if (_neighborhoodTiled8x8F32 == 0 || g.HeadDim != 64) return (0, 0);
        if (g.Height >= 8 && g.Width >= 8) return (8, 8);
        if (g.Height >= 4 && g.Width >= 8) return (4, 8);
        return (0, 0);
    }

    /// <summary>Per-query neighborhood attention: one block per (batch, t, h, w, head). Threads split the window in the score pass and
    /// the head width in the value pass; 256 covers head width 64 and keeps 8 warps on the window.</summary>
    public unsafe void LaunchNeighborhoodAttention(ulong output, ulong q, ulong k, ulong v, in NeighborhoodGeometry g, float scale, nint stream)
    {
        if (_neighborhoodModule is null)
            throw new InvalidOperationException("neighborhood_attention.ptx is not loaded; gate on HasNeighborhoodAttentionKernels first.");
        uint gridDim = (uint)((long)g.Batch * g.Depth * g.Height * g.Width * g.Heads);
        Launch(_neighborhoodAttentionF32, gridDim, (uint)NeighborhoodAttentionSharedBytes(g), output, q, k, v, g, scale, stream);
    }

    /// <summary>Query-tiled neighborhood attention: one block per (batch, t, h-tile, w-tile, head) streaming the tile's union window through
    /// shared memory. The tile must come from <see cref="NeighborhoodAttentionTile"/>.</summary>
    public unsafe void LaunchNeighborhoodAttentionTiled(ulong output, ulong q, ulong k, ulong v, in NeighborhoodGeometry g, float scale, int tileH, int tileW, nint stream)
    {
        nint func = tileH == 8 ? _neighborhoodTiled8x8F32 : _neighborhoodTiled4x8F32;
        if (func == 0)
            throw new InvalidOperationException("The tiled neighborhood attention kernels are not loaded; gate on NeighborhoodAttentionTile first.");
        long hTiles = (g.Height + tileH - 1) / tileH, wTiles = (g.Width + tileW - 1) / tileW;
        uint gridDim = (uint)((long)g.Batch * g.Depth * hTiles * wTiles * g.Heads);
        Launch(func, gridDim, 0, output, q, k, v, g, scale, stream);
    }

    private static unsafe void Launch(nint func, uint gridDim, uint sharedBytes, ulong output, ulong q, ulong k, ulong v, in NeighborhoodGeometry g, float scale, nint stream)
    {
        ulong outArg = output, qArg = q, kArg = k, vArg = v;
        uint bArg = (uint)g.Batch, tArg = (uint)g.Depth, hArg = (uint)g.Height, wArg = (uint)g.Width;
        uint headsArg = (uint)g.Heads, hdArg = (uint)g.HeadDim;
        uint ktArg = (uint)g.WindowDepth, khArg = (uint)g.WindowHeight, kwArg = (uint)g.WindowWidth;
        float scaleArg = scale;
        void** args = stackalloc void*[14];
        args[0] = &outArg; args[1] = &qArg; args[2] = &kArg; args[3] = &vArg;
        args[4] = &bArg; args[5] = &tArg; args[6] = &hArg; args[7] = &wArg;
        args[8] = &headsArg; args[9] = &hdArg;
        args[10] = &ktArg; args[11] = &khArg; args[12] = &kwArg; args[13] = &scaleArg;
        CudaDriverApi.cuLaunchKernel(func, gridDim, 1, 1, 256, 1, 1, sharedBytes, stream, (nint)args, 0).ThrowOnError();
    }
}
