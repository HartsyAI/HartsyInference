using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Interactive.Memory;

namespace SharpInference.Interactive.Tests;

/// <summary>Rolling-buffer semantics: bounded eviction, deep latent copies, Last(n) ordering.</summary>
public unsafe class FrameHistoryBufferTests
{
    [Fact]
    public void Add_EvictsOldestAtCapacity_AndCopiesLatents()
    {
        using FrameHistoryBuffer buffer = new(capacity: 3);
        float[] pose = new float[16];

        Tensor frame = new Tensor(new TensorShape([1L, 2, 1, 2, 2]), DType.F32);
        for (long i = 0; i < 5; i++)
        {
            *(float*)frame.DataPointer = i;     // mutate the SOURCE between adds
            buffer.Add(frame, pose, frameIndex: i);
        }
        frame.Dispose();

        Assert.Equal(3, buffer.Count);
        Assert.Equal(2, buffer[0].FrameIndex);
        Assert.Equal(4, buffer[2].FrameIndex);
        // Deep copies: each stored latent kept the value at Add time, not the source's final value.
        Assert.Equal(2f, *(float*)buffer[0].Latent.DataPointer);
        Assert.Equal(4f, *(float*)buffer[2].Latent.DataPointer);

        FrameHistoryBuffer.Entry[] last2 = buffer.Last(2);
        Assert.Equal(3, last2[0].FrameIndex);
        Assert.Equal(4, last2[1].FrameIndex);
        Assert.Throws<ArgumentOutOfRangeException>(() => buffer.Last(4));
    }
}
