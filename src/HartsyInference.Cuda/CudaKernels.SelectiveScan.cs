namespace HartsyInference.Cuda;

// The Mamba-1 selective scan (IBackend.SelectiveScan; Kernels/lm/ssm_selective_scan_f32.cu). One warp per (row, channel),
// four warps per block.
public sealed partial class CudaKernels
{
    /// <summary>Widest state count the selective-scan kernel holds: eight states per lane across a 32-lane warp.</summary>
    public const int SelectiveScanMaxStates = 256;

    private const int SelectiveScanWarpsPerBlock = 4;

    private CudaModule? _selectiveScanModule;
    private nint _ssmSelectiveScan;

    /// <summary>True when ssm_selective_scan_f32.ptx loaded.</summary>
    public bool HasSelectiveScanKernel => _ssmSelectiveScan != 0;

    // Optional module: its absence leaves the selective scan on the host reference instead of failing construction.
    private void LoadSelectiveScanKernel()
    {
        string path = Ptx("ssm_selective_scan_f32");
        if (!CanLoad(path))
            return;
        _selectiveScanModule = LoadOwnedModule(path);
        _ssmSelectiveScan = _selectiveScanModule.GetFunction("ssm_selective_scan_f32");
    }

    /// <summary>The selective scan over <paramref name="rows"/> sequences of <paramref name="frames"/> steps. <paramref name="output"/>
    /// must be a buffer distinct from <paramref name="u"/>; <paramref name="z"/> and <paramref name="state"/> are read only when
    /// nonzero (<paramref name="hasZ"/>, <paramref name="hasState"/>), and the state is written back only when <paramref name="hasState"/> is set.</summary>
    public unsafe void LaunchSsmSelectiveScan(ulong output, ulong u, ulong delta, ulong a, ulong b, ulong c, ulong d, ulong z,
        ulong state, bool hasZ, bool hasState, int rows, int frames, int channels, int states, nint stream)
    {
        if (_ssmSelectiveScan == 0) throw new InvalidOperationException("ssm_selective_scan_f32.ptx not present in the Ptx folder.");
        if (states < 1 || states > SelectiveScanMaxStates)
            throw new NotSupportedException($"The CUDA selective scan holds up to {SelectiveScanMaxStates} states per channel; got {states}.");
        ulong oA = output, uA = u, dA = delta, aA = a, bA = b, cA = c, dpA = d, zA = z, stA = state;
        uint hasZA = hasZ ? 1u : 0u, hasStateA = hasState ? 1u : 0u;
        uint rowsA = (uint)rows, framesA = (uint)frames, channelsA = (uint)channels, statesA = (uint)states;
        // Fifteen kernel parameters, in signature order; each points at a local that outlives the launch call.
        void** args = stackalloc void*[15];
        args[0] = &oA; args[1] = &uA; args[2] = &dA; args[3] = &aA; args[4] = &bA; args[5] = &cA; args[6] = &dpA;
        args[7] = &zA; args[8] = &stA; args[9] = &hasZA; args[10] = &hasStateA; args[11] = &rowsA; args[12] = &framesA;
        args[13] = &channelsA; args[14] = &statesA;
        uint gridX = (uint)((channels + SelectiveScanWarpsPerBlock - 1) / SelectiveScanWarpsPerBlock);
        CudaDriverApi.cuLaunchKernel(_ssmSelectiveScan, gridX, (uint)rows, 1, 32 * SelectiveScanWarpsPerBlock, 1, 1, 0, stream,
            (nint)args, 0).ThrowOnError();
    }
}
