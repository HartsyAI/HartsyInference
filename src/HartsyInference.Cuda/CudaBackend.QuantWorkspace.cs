using HartsyInference.Core.Tensors;
using HartsyInference.Core.Tensors.Quant;

namespace HartsyInference.Cuda;

// Recipe dequant to BF16 for the expert path.
public sealed partial class CudaBackend
{
    private CudaQuantWorkspace? _quantWorkspace;
    private readonly object _quantWorkspaceGate = new();

    /// <summary>The backend's BF16 dequant ring for MXFP4 and FP8-block recipe weights.</summary>
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

    internal void LaunchRecipeDequant(QuantRecipe recipe, ulong packed, ulong scale, ulong output)
    {
        using OpScope _op = EnterOp();
        EnsureKernels();
        if (_kernels is not { HasRecipeDequantKernels: true })
            throw new NotSupportedException("dequant_recipe_to_bf16.ptx is not loaded (Kernels/dequant/build.sh).");
        long stride = recipe.Scale!.Shape[1];
        int rows = checked((int)recipe.LogicalRows);
        int cols = checked((int)recipe.LogicalCols);
        int offset = checked((int)recipe.ScaleColOffset);
        if (recipe.Encoding == QuantEncoding.Mxfp4E8M0)
        {
            _kernels.LaunchMxfp4E8m0Dequant(output, packed, scale, rows, cols / 2, stride, offset,
                recipe.Geometry.BlockRows, recipe.Geometry.BlockCols, _stream.Handle);
        }
        else
        {
            _kernels.LaunchFp8BlockE8m0Dequant(output, packed, scale, rows, cols, stride, offset,
                recipe.Geometry.BlockRows, recipe.Geometry.BlockCols, _stream.Handle);
        }
    }
}
