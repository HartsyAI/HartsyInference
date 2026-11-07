using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Contracts;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// Rotary position embedding: the generic shader for every geometry (F32 and F16 activations, F32 tables, an optional
// device position buffer); BF16 activations take the host reference.
public sealed partial class VulkanBackend
{
    /// <inheritdoc />
    public void Rotary(in RotaryCall call)
    {
        using OpScope _op = EnterOp();
        RotaryGeometry g = RotaryContract.Validate(call);
        if (call.X.DType == DType.BF16)
        {
            RotaryReference.Apply(call);
            return;
        }
        if (g.PairCount > uint.MaxValue || call.Spec.X.MaxElementIndex > uint.MaxValue || call.Spec.Cos.MaxElementIndex > uint.MaxValue || call.Spec.Sin.MaxElementIndex > uint.MaxValue)
            throw new NotSupportedException($"Vulkan Rotary addresses at most {uint.MaxValue} elements; got {call.Spec}.");
        VulkanBuffer? posBuf = null;
        if (g.DevicePosition != 0 && !_scalarBuffers.TryGetValue(g.DevicePosition, out posBuf))
            throw new NotSupportedException("Rotary with a device position needs a buffer from AllocDevicePos.");
        VulkanBuffer xBuf = GetBuffer(call.X);
        VulkanBuffer cosBuf = GetBuffer(call.Cos);
        VulkanBuffer sinBuf = GetBuffer(call.Sin);
        try
        {
            VulkanKernel kernel = GetKernel("rotary" + DtypeSuffix(call.X.DType), 4, _default1DSpec);
            Span<byte> pcBytes = stackalloc byte[(int)VulkanDescriptorManager.PushConstantRangeBytes];
            Authoring.PushConstants pc = new(pcBytes);
            for (int axis = 0; axis < RotaryGeometry.IndexAxes; axis++) pc.U32((uint)g.Dim(axis));
            for (int axis = 0; axis < RotaryGeometry.IndexAxes; axis++) pc.U32((uint)g.XStride(axis));
            for (int axis = 0; axis < RotaryGeometry.IndexAxes; axis++) pc.U32((uint)g.CosStride(axis));
            for (int axis = 0; axis < RotaryGeometry.IndexAxes; axis++) pc.U32((uint)g.SinStride(axis));
            pc.U32((uint)g.XSecond);
            pc.U32((uint)g.CosSecond);
            pc.U32((uint)g.SinSecond);
            pc.U32((uint)g.XOffset);
            pc.U32((uint)g.CosOffset);
            pc.U32((uint)g.SinOffset);
            pc.U32((uint)g.PositionRowStride);
            pc.U32((uint)g.PositionSlot);
            // The position binding is read only with a row stride; without one it binds the cosine buffer as a placeholder.
            Span<ulong> bufs = stackalloc ulong[] { xBuf.Handle, cosBuf.Handle, sinBuf.Handle, (posBuf ?? cosBuf).Handle };
            Dispatch(kernel, bufs, pc.Written, GroupCount(g.PairCount, LocalX1D));
            // In place on x's existing buffer: the rebind is what tells the tensor its device contents are current.
            CacheOutput(call.X, xBuf);
        }
        catch (Exception ex)
        {
            Logs.Error("Vulkan Rotary dispatch failed", ex);
            throw;
        }
    }

    /// <inheritdoc />
    public void RmsNormRotary(Tensor output, Tensor input, Tensor normWeight, float eps, Tensor? cos, Tensor? sin, in RotarySpec rotation)
        => IBackend.ComposeRmsNormRotary(this, output, input, normWeight, eps, cos, sin, rotation);
}
