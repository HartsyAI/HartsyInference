namespace HartsyInference.Cuda;

// Per-key-channel delta-rule prefill (Kimi KDA; Kernels/lm/ssm_channel_decay_prefill_f32.cu). One warp per (state row,
// value head), the same row layout as the Qwen3.5 scan in CudaKernels.SsmPrefill.cs.
public sealed partial class CudaKernels
{
    private CudaModule? _channelDecayModule;
    private nint _ssmChannelDecayPrefill;

    /// <summary>True when ssm_channel_decay_prefill_f32.ptx loaded.</summary>
    public bool HasChannelDecayPrefillKernel => _ssmChannelDecayPrefill != 0;

    // Optional module: its absence leaves the channel-decay scan on the host reference instead of failing construction.
    private void LoadChannelDecayPrefillKernel()
    {
        string path = Ptx("ssm_channel_decay_prefill_f32");
        if (!CanLoad(path))
            return;
        _channelDecayModule = LoadOwnedModule(path);
        _ssmChannelDecayPrefill = _channelDecayModule.GetFunction("ssm_channel_decay_prefill_f32");
    }

    /// <summary>The per-channel-decay delta-rule scan over <paramref name="seq"/> tokens; the output is the raw readout.</summary>
    public unsafe void LaunchSsmChannelDecayPrefill(ulong output, ulong state, ulong q, ulong k, ulong v, ulong logDecay,
        ulong betaRaw, int seq, int hv, int sv, int sk, int repeat, nint stream)
    {
        if (_ssmChannelDecayPrefill == 0) throw new InvalidOperationException("ssm_channel_decay_prefill_f32.ptx not present in the Ptx folder.");
        if (sk > SsmPrefillMaxKeyDim) throw new NotSupportedException($"The CUDA channel-decay scan holds key heads up to {SsmPrefillMaxKeyDim} wide; got {sk}.");
        ulong oA = output, stA = state, qA = q, kA = k, vA = v, gA = logDecay, beA = betaRaw;
        uint seqA = (uint)seq, hvA = (uint)hv, svA = (uint)sv, skA = (uint)sk, repA = (uint)repeat;
        void** scan = stackalloc void*[12];
        scan[0] = &oA; scan[1] = &stA; scan[2] = &qA; scan[3] = &kA; scan[4] = &vA; scan[5] = &gA; scan[6] = &beA;
        scan[7] = &seqA; scan[8] = &hvA; scan[9] = &svA; scan[10] = &skA; scan[11] = &repA;
        uint gridX = (uint)((sv + SsmScanRowsPerBlock - 1) / SsmScanRowsPerBlock);
        CudaDriverApi.cuLaunchKernel(_ssmChannelDecayPrefill, gridX, (uint)hv, 1, 32 * SsmScanRowsPerBlock, 1, 1, 0, stream,
            (nint)scan, 0).ThrowOnError();
    }
}
