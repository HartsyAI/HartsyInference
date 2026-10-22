using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Tests.Common.Attention;
using Xunit;

namespace HartsyInference.Core.Tests.Attention;

/// <summary>Negative controls for <see cref="AttentionTraceOps"/>: the spelling that the trace pins compare changes when what a site does with
/// an attention call changes, and does not change between the legacy SDPA entry and the layer's Dense call.</summary>
public sealed class AttentionTraceOpsTests
{
    private const string Plain = "Attn Dense scale=0.5 f16=False out=[1, 2, 3, 4]:F32 q=[1, 2, 3, 4]:F32 k=[1, 2, 3, 4]:F32 v=[1, 2, 3, 4]:F32 mask=none live=4";

    private static Tensor T(params long[] dims) => new(new TensorShape(dims), DType.F32);

    /// <summary>A synthetic site: one legacy SDPA call, with the knobs a pin must be able to see.</summary>
    private static string[] Site(float scale = 0.5f, bool masked = false, bool freeQBefore = false, bool freeQAfter = false, bool keepExtra = false) =>
        AttentionTraceOps.Record(b =>
        {
            Tensor o = T(1, 2, 3, 4), q = T(1, 2, 3, 4), k = T(1, 2, 3, 4), v = T(1, 2, 3, 4), extra = T(2, 2);
            b.Silu(extra, extra);
            if (!keepExtra) extra.Dispose();
            if (freeQBefore) q.Dispose();
            b.ScaledDotProductAttention(o, q, k, v, masked ? T(1, 2, 3, 3) : null, scale);
            if (freeQAfter) q.Dispose();
        });

    private static string WithoutLive(string line) => line[..line.IndexOf(" live=", StringComparison.Ordinal)];

    [Fact]
    public void Spell_ReadsEveryOpThenTheOperandStateAfterTheRun()
    {
        Assert.Equal(["Silu", Plain, "attention #0 after run: out=live q=live k=live v=live mask=none"], Site());
    }

    [Fact]
    public void Spell_SeesAQFreedBeforeTheCallAndOneFreedAfterIt()
    {
        string[] keep = Site(), after = Site(freeQAfter: true), before = Site(freeQBefore: true);
        Assert.Equal("attention #0 after run: out=live q=freed k=live v=live mask=none", after[^1]);
        Assert.NotEqual(keep, after);
        Assert.Equal(after[^1], before[^1]);
        Assert.Equal(Plain, after[1]);
        Assert.Equal(Plain.Replace("live=4", "live=3"), before[1]);
    }

    [Fact]
    public void Spell_SeesADifferentScale()
    {
        Assert.Equal(Plain.Replace("scale=0.5", "scale=0.353553"), Site(scale: 1f / MathF.Sqrt(8))[1]);
        Assert.NotEqual(Site(), Site(scale: 0.25f));
    }

    [Fact]
    public void Spell_SeesAMaskPresentOrAbsent()
    {
        string[] masked = Site(masked: true);
        Assert.Equal(Plain.Replace("mask=none", "mask=[1, 2, 3, 3]:F32"), masked[1]);
        Assert.Equal("attention #0 after run: out=live q=live k=live v=live mask=live", masked[^1]);
        Assert.NotEqual(Site(), masked);
    }

    [Fact]
    public void Spell_SeesADifferentLiveCountAtTheCall()
    {
        string[] gone = Site(), kept = Site(keepExtra: true);
        Assert.Equal(WithoutLive(gone[1]), WithoutLive(kept[1]));
        Assert.Equal(gone[0], kept[0]);
        Assert.Equal("live=4", gone[1][^6..]);
        Assert.Equal("live=5", kept[1][^6..]);
        Assert.NotEqual(gone, kept);
    }

    [Fact]
    public void Spell_ReadsTheLegacyEntryAndTheLayersDenseCallTheSame()
    {
        foreach ((bool masked, bool allowF16) in new[] { (false, false), (true, false), (false, true), (true, true) })
        {
            string[] legacy = AttentionTraceOps.Record(b =>
            {
                Tensor o = T(1, 2, 3, 4), q = T(1, 2, 3, 4), k = T(1, 2, 5, 4), v = T(1, 2, 5, 4);
                b.ScaledDotProductAttention(o, q, k, v, masked ? T(1, 1, 3, 5) : null, 0.5f, allowF16);
            });
            string[] layer = AttentionTraceOps.Record(b =>
            {
                Tensor o = T(1, 2, 3, 4), q = T(1, 2, 3, 4), k = T(1, 2, 5, 4), v = T(1, 2, 5, 4);
                b.Attention(AttentionCall.FromScaledDotProductAttention(o, q, k, v, masked ? T(1, 1, 3, 5) : null, 0.5f, allowF16));
            });
            Assert.Equal(legacy, layer);
            Assert.StartsWith($"Attn Dense scale=0.5 f16={allowF16} ", legacy[0]);
        }
    }

    [Fact]
    public void Record_EndsTheRunAtAnOpTheStubDoesNotServe()
    {
        Assert.Equal(["Silu"], AttentionTraceOps.Record(b =>
        {
            b.Silu(T(2), T(2));
            b.Tanh(T(2), T(2));
            b.Silu(T(2), T(2));
        }));
        Assert.Equal(["Silu"], AttentionTraceOps.Record(b =>
        {
            b.Silu(T(2), T(2));
            typeof(AttentionTraceBackend).GetMethod("Tanh")!.Invoke(b, [T(2), T(2)]);
        }));
    }
}
