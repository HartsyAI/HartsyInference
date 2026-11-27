using System.Reflection;
using HartsyInference.Core.Memory.Vram.Streaming;
using Xunit;

namespace HartsyInference.Core.Tests.Residency;

/// <summary>The streaming window's bytes: prefetchAhead + 1 blocks at the widest block's size, the amount a headroom must include.</summary>
public sealed class StreamingWindowTests
{
    [Fact]
    public void TheWindowIsPrefetchAheadPlusOneTimesTheWidestBlock()
    {
        IStreamableDenoiser denoiser = DispatchProxy.Create<IStreamableDenoiser, FakeDenoiser>();
        ((FakeDenoiser)(object)denoiser).Sizes = [100, 400, 250];

        Assert.Equal(3 * 400, StreamingWindow.Bytes(denoiser, prefetchAhead: 2));
        Assert.Equal(1 * 400, StreamingWindow.Bytes(denoiser, prefetchAhead: 0));
    }

    /// <summary>A denoiser with block sizes; the window reads only the count and each block's estimated bytes.</summary>
    public class FakeDenoiser : DispatchProxy
    {
        public long[] Sizes { get; set; } = [];

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            switch (targetMethod?.Name)
            {
                case "get_BlockCount":
                    return Sizes.Length;
                case "GetBlock":
                    IStreamingBlock block = DispatchProxy.Create<IStreamingBlock, FakeBlock>();
                    ((FakeBlock)(object)block).Bytes = Sizes[(int)args![0]!];
                    return block;
                default:
                    return null;
            }
        }
    }

    public class FakeBlock : DispatchProxy
    {
        public long Bytes { get; set; }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) =>
            targetMethod?.Name == "get_EstimatedWeightBytes" ? Bytes : null;
    }
}
