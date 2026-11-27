using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The guided Euler step: the unnormalized form as one shader with the blend form a spec constant, so a device-resident latent
// stays resident across the denoise loop (the host default would read it back and evict it every step, which also broke step-graph
// capture); the normalized forms run the host reference.
public sealed partial class VulkanBackend
{
    /// <summary>Spec constant id of the guided Euler shader: the blend form.</summary>
    private const uint GuidedEulerFormId = 3;

    /// <inheritdoc />
    public unsafe void GuidedEulerStep(Tensor z, Tensor cond, Tensor uncond, in GuidedEulerSpec spec)
    {
        GuidedEulerGeometry g = GuidedEulerContract.Validate(z, cond, uncond, spec);
        if (spec.Delta == 0f) return;
        if (spec.Norm.Kind != GuidanceNormKind.None)
        {
            GuidedEulerReference.Apply(z, cond, uncond, spec);
            return;
        }
        if (g.Count > uint.MaxValue)
            throw new NotSupportedException($"Vulkan GuidedEulerStep addresses at most {uint.MaxValue} elements; got {z.Shape}.");
        using OpScope _op = EnterOp();
        VulkanBuffer zBuf = GetBuffer(z);
        VulkanBuffer condBuf = GetBuffer(cond);
        VulkanBuffer uncondBuf = ReferenceEquals(cond, uncond) ? condBuf : GetBuffer(uncond);

        SpecConstant[] constants = [SpecConstant.UInt(0, LocalX1D), SpecConstant.UInt(1, 1), SpecConstant.UInt(2, 1), SpecConstant.UInt(GuidedEulerFormId, (uint)spec.Form)];
        VulkanKernel k = GetKernel("guided_euler", 3, constants);
        Span<byte> pc = stackalloc byte[3 * 4];
        BinaryWriteUInt(pc, 0, (uint)g.Count);
        BinaryWriteFloat(pc, 4, spec.Guidance);
        BinaryWriteFloat(pc, 8, spec.Delta);
        Span<ulong> bufs = stackalloc ulong[] { zBuf.Handle, condBuf.Handle, uncondBuf.Handle };
        Dispatch(k, bufs, pc, GroupCount(g.Count, LocalX1D));

        // In place: z keeps this buffer. Clear the stale callbacks before re-caching so the old sync callback cannot free it.
        z._gpuSyncCallback = null;
        z._gpuDisposeCallback = null;
        CacheOutput(z, zBuf);

        if (uncondBuf != condBuf) _xfer.FreeDevice(uncondBuf);
        _xfer.FreeDevice(condBuf);
    }
}
