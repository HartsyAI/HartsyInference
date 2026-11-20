using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Exceptions;

namespace HartsyInference.Cuda;

// The partial-state scratch of split and grouped decode: a running maximum, a running sum and an accumulator per
// (query block, key split), written by the split launch and read by the combine launch of the same call.
public sealed partial class CudaBackend
{
    private readonly HighWaterWorkspace _decodePartials = new(static bytes => GpuTransferHelper.AllocateDevice((nuint)bytes),
        GpuTransferHelper.FreeDevice);

    /// <summary>Bytes the eager decode partial workspace holds now (its high-water mark since it was last released).</summary>
    internal long DecodePartialsWorkspaceBytes => _decodePartials.Bytes;

    /// <summary>The three partial buffers of one decode call. <see cref="Transient"/> ones are freed when the call returns.</summary>
    private readonly record struct DecodePartials(ulong M, ulong L, ulong Acc, bool Transient);

    /// <summary>Scratch for <paramref name="blocks"/> query blocks over <paramref name="splits"/> key splits at head dim
    /// <paramref name="d"/>. An eager call takes sections of one buffer kept at its high-water size, so a decode step
    /// allocates nothing once the buffer has grown. A call recorded into a graph allocates three buffers from the capture's
    /// allocator, as before: a replay must not write into a buffer an eager call can replace.</summary>
    private DecodePartials RentDecodePartials(long blocks, int splits, int d, string site)
    {
        long scalarBytes = blocks * splits * sizeof(float);
        long accBytes = scalarBytes * d;
        if (CapturingGraph is null)
        {
            long section = (scalarBytes + 255) & ~255L;
            long bytes = 2 * section + accBytes;
            ulong buffer = _decodePartials.Use(bytes, lessee: null);
            if (buffer == 0) buffer = GrowDecodePartials(bytes);
            if (buffer != 0) return new DecodePartials(buffer, buffer + (ulong)section, buffer + (ulong)(2 * section), Transient: false);
        }
        ulong pM = 0, pL = 0, pAcc = 0;
        try
        {
            pM = GpuTransferHelper.AllocateDevice((nuint)scalarBytes);
            pL = GpuTransferHelper.AllocateDevice((nuint)scalarBytes);
            pAcc = GpuTransferHelper.AllocateDevice((nuint)accBytes);
        }
        catch
        {
            GpuTransferHelper.FreeDevice(pM);
            GpuTransferHelper.FreeDevice(pL);
            GpuTransferHelper.FreeDevice(pAcc);
            throw;
        }
        AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, site, 2 * scalarBytes + accBytes, count: 3);
        return new DecodePartials(pM, pL, pAcc, Transient: true);
    }

    /// <summary>The workspace grown to <paramref name="bytes"/>, or 0 when a capture is open on the stream after all or the
    /// size does not fit; the call then takes transient buffers.</summary>
    private ulong GrowDecodePartials(long bytes)
    {
        if (StreamIsCapturing()) return 0;
        try
        {
            ulong buffer = _decodePartials.Grow(bytes);
            if (buffer != 0) AttentionEvidence.Observed(AttentionEvidenceKind.Workspace, "cuda.flash.partials.workspace.grow", bytes);
            return buffer;
        }
        catch (OutOfVramException)
        {
            return 0;
        }
    }

    private static void ReturnDecodePartials(in DecodePartials partials)
    {
        if (!partials.Transient) return;
        GpuTransferHelper.FreeDevice(partials.M);
        GpuTransferHelper.FreeDevice(partials.L);
        GpuTransferHelper.FreeDevice(partials.Acc);
    }
}
