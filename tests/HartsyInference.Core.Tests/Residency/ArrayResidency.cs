using HartsyInference.Core.Backends.Gpu;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>A residency cache whose device buffers are arrays: the logic both GPU backends share, without a GPU.</summary>
internal sealed class ArrayResidency : GpuResidencyCache<float[]>
{
    /// <summary>Leaves <paramref name="values"/> as an op's output would: on the device only.</summary>
    public float[] Produce(Tensor output, float[] values)
    {
        CacheActivation(output, values, ByteSize(output));
        return values;
    }

    protected override float[] AllocateDevice(long bytes) => new float[bytes / sizeof(float)];

    protected override void FreeDevice(float[] buffer, long bytes) { }

    protected override void Upload(float[] destination, Tensor source, long bytes) => source.AsReadOnlySpan<float>().CopyTo(destination);

    protected override unsafe void DownloadSynced(nint hostDestination, float[] source, long bytes) =>
        source.AsSpan().CopyTo(new Span<float>((void*)hostDestination, source.Length));

    protected override void MakeCurrent() { }
}
