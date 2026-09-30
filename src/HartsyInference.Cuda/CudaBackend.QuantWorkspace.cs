using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Cuda;

// Recipe dequant to BF16 for the expert path.
public sealed partial class CudaBackend
{
    private CudaQuantWorkspace? _quantWorkspace;
    private readonly object _quantWorkspaceGate = new();

    /// <summary>The backend's BF16 dequant ring for recipe weights (MXFP4, FP8 block, ModelOpt NVFP4, MLX affine).</summary>
    public CudaQuantWorkspace QuantWorkspace
    {
        get
        {
            lock (_quantWorkspaceGate) return _quantWorkspace ??= new CudaQuantWorkspace(this);
        }
    }

    /// <summary>Whether the recipe dequant kernels are loaded on this device.</summary>
    public bool SupportsRecipeDequant => _kernels is { HasRecipeDequantKernels: true };

    internal bool TryGetResidentPointer(Tensor tensor, out ulong devicePointer) =>
        _streamingCache.TryGetDevicePointer(tensor, out devicePointer);

    internal ulong AllocateWorkspace(nuint bytes)
    {
        using OpScope _op = EnterOp();
        return CudaMemory.AllocatePersistent(bytes);
    }

    internal void FreeWorkspace(ulong[] buffers)
    {
        if (_stream is not null) _stream.Synchronize();
        foreach (ulong buffer in buffers)
        {
            if (buffer != 0) CudaMemory.Free(buffer);
        }
    }

    internal void DisposeQuantWorkspace()
    {
        CudaQuantWorkspace? workspace;
        lock (_quantWorkspaceGate)
        {
            workspace = _quantWorkspace;
            _quantWorkspace = null;
        }
        workspace?.Dispose();
    }

    internal void LaunchRecipeDequant(QuantRecipe recipe, ulong packed, ulong scale, ulong extra, ulong output)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (_kernels is not { HasRecipeDequantKernels: true })
            throw new NotSupportedException("dequant_recipe_to_bf16.ptx is not loaded (Kernels/dequant/build.sh).");
        if (recipe.Encoding == QuantEncoding.Exl3Trellis)
        {
            if (!_kernels.HasExl3DequantKernel)
                throw new NotSupportedException("The EXL3 dequant kernel is not available on this device (missing PTX entry or no 66,048-byte shared memory opt-in).");
            // packed = trellis, scale = suh, extra = svh (the workspace resolves the resident pointers).
            _kernels.LaunchExl3Dequant(
                output, packed, scale, extra, checked((int)(recipe.LogicalRows / Exl3Format.TileSize)), 0,
                checked((int)(recipe.LogicalRows / Exl3Format.HadamardBlock)), checked((int)recipe.LogicalCols), _stream.Handle);
            return;
        }
        long stride = recipe.Scale!.Shape[1];
        int rows = checked((int)recipe.LogicalRows);
        int cols = checked((int)recipe.LogicalCols);
        int offset = checked((int)recipe.ScaleColOffset);
        int blockRows = recipe.Geometry.BlockRows;
        int blockCols = recipe.Geometry.BlockCols;
        switch (recipe.Encoding)
        {
            case QuantEncoding.Mxfp4E8M0:
                _kernels.LaunchMxfp4E8m0Dequant(output, packed, scale, rows, cols / 2, stride, offset, blockRows, blockCols, _stream.Handle);
                break;
            case QuantEncoding.Fp8E4M3BlockE8M0:
                _kernels.LaunchFp8BlockE8m0Dequant(output, packed, scale, rows, cols, stride, offset, blockRows, blockCols, _stream.Handle);
                break;
            case QuantEncoding.Nvfp4:
                _kernels.LaunchNvfp4ModelOptDequant(
                    output, packed, scale, extra, rows, cols / 2, stride, offset, blockRows, blockCols, _stream.Handle);
                break;
            case QuantEncoding.AffineInt4:
            case QuantEncoding.AffineInt8:
                int bits = recipe.Encoding == QuantEncoding.AffineInt4 ? 4 : 8;
                _kernels.LaunchAffineDequant(output, packed, scale, extra, rows, cols, stride, offset, blockRows, blockCols, bits, _stream.Handle);
                break;
            default:
                throw new NotSupportedException($"No device dequant for {recipe.Encoding}.");
        }
    }
}
