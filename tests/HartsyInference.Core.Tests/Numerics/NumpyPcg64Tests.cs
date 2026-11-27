using Xunit;
using HartsyInference.Core.Numerics;

namespace HartsyInference.Core.Tests.Numerics;

/// <summary>Expected values come from numpy 2.5.4: <c>default_rng(seed).bit_generator.random_raw(3)</c>, <c>.random(2)</c> and <c>.integers(0, high, size, dtype=np.int64)</c>.</summary>
public sealed class NumpyPcg64Tests
{
    [Theory]
    [InlineData(0u, 11749869230777074271UL, 4976686463289251617UL, 755828109848996024UL)]
    [InlineData(20014u, 15162556381065574746UL, 7352191438554230333UL, 4488349457964797913UL)]
    [InlineData(120084u, 3604390295989161767UL, 8285703047990029387UL, 15704527335329931138UL)]
    [InlineData(5u, 14849682912918955432UL, 14903876974979881461UL, 9506078739185184192UL)]
    public void RawStream_MatchesNumpy(uint seed, ulong a, ulong b, ulong c)
    {
        NumpyPcg64 bits = new(seed);
        Assert.Equal([a, b, c], [bits.NextUInt64(), bits.NextUInt64(), bits.NextUInt64()]);
    }

    [Fact]
    public void NextDouble_MatchesNumpy()
    {
        NumpyPcg64 bits = new(3);
        Assert.Equal(0.08564916714362436, bits.NextDouble());
        Assert.Equal(0.2368105065960997, bits.NextDouble());
    }

    [Theory]
    [InlineData(0u, 1099511627776L, new long[] { 700346781657, 296633628802, 45050865998, 18172327443, 894200084524 })]
    [InlineData(20014u, 1099511640121L, new long[] { 903758797127, 438224763935, 267526478073, 843715879960, 637312404786 })]
    [InlineData(5u, 4611686018427387911L, new long[]
    {
        3712420728229738863, 3725969243744970370, 2376519684796296051, 1318026228599734559, 248711466137453627,
    })]
    public void WideIntegerDraws_MatchNumpy(uint seed, long high, long[] expected)
    {
        NumpyPcg64 bits = new(seed);
        Assert.Equal(expected, expected.Select(_ => bits.NextInt64(0, high)).ToArray());
    }

    [Fact]
    public void NarrowRanges_AreRefusedRatherThanDrawnWrong()
    {
        Assert.Throws<NotSupportedException>(() => new NumpyPcg64(0).NextInt64(0, 1000));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NumpyPcg64(0).NextInt64(5, 5));
    }
}
