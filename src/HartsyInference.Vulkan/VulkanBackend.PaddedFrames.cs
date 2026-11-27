using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Vulkan;

// The padded input of a batched CausalConv3d on the device. The default writes it with a host loop that reads the input back, so a causal conv
// stack paid one device-to-host read per conv. This composes it from strided copies, which run on the device.
public sealed partial class VulkanBackend
{
    /// <inheritdoc/>
    /// <remarks>Reflected spatial padding has no strided form and F32 is the only dtype the composition copies, so those calls keep the host path.</remarks>
    public unsafe void BuildPaddedFrames(Tensor padded, Tensor input, Tensor? cache, int zeroPad,
        bool replicateFirst = false, int padH = 0, int padW = 0, bool reflectSpatial = false)
    {
        if (reflectSpatial || padded.DType != DType.F32 || input.DType != DType.F32 || (cache is not null && cache.DType != DType.F32))
        {
            IBackend.BuildPaddedFramesReference(padded, input, cache, zeroPad, replicateFirst, padH, padW, reflectSpatial);
            return;
        }
        PaddedFrames.Compose(this, padded, input, cache, zeroPad, replicateFirst, padH, padW);
    }
}
