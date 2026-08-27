using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Exercises the hand-written coercion on each manually declared knob against the bound its original call site enforced.</summary>
/// <remarks>Only the out-of-range cases prove anything — every happy path passes with the coercion deleted. That
/// matters more now than it did: settings arrive from a file or <c>--set</c> rather than from a parser that used
/// to apply the bound on the way in, so this is the only thing standing between a typo and a kernel.
/// <para>The two shapes deliberately disagree: a rejecting knob sends an out-of-range value to its default, while
/// a clamping knob sends it to the nearest bound. <c>numerics.gemvWpb</c> at 17 is 16, NOT the default 4.</para></remarks>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class ManualKnobGuardTests
{
    /// <summary>Applies <paramref name="value"/> as a supplied setting, or leaves the knob at its default when null.</summary>
    private static void With<T>(Knob<T> knob, T? value, Action body) where T : struct
    {
        try
        {
            if (value is T v)
            {
                KnobStore.Set(knob, v);
            }
            body();
        }
        finally
        {
            KnobStore.Clear(knob);
        }
    }

    /// <summary>Rejecting knob: non-positive falls back to the declared 14 GB.</summary>
    [Theory]
    [InlineData(-5L, 14L)]
    [InlineData(0L, 14L)]
    [InlineData(20L, 20L)]
    [InlineData(null, 14L)]
    public void AudioEvictBelowGb_RejectsNonPositive(long? value, long expected)
        => With(EngineKnobs.AudioEvictBelowGb, value, () => Assert.Equal(expected, EngineKnobs.AudioEvictBelowGb.Value));

    /// <summary>Rejecting knob: non-positive falls back to the measured 8192 knee.</summary>
    [Theory]
    [InlineData(0, 8192)]
    [InlineData(-1, 8192)]
    [InlineData(4096, 4096)]
    [InlineData(null, 8192)]
    public void SageF16MinSkv_RejectsNonPositive(int? value, int expected)
        => With(EngineKnobs.SageF16MinSkv, value, () => Assert.Equal(expected, EngineKnobs.SageF16MinSkv.Value));

    /// <summary>Rejecting knob: negative falls back to 0.</summary>
    [Theory]
    [InlineData(-1L, 0L)]
    [InlineData(512L, 512L)]
    [InlineData(null, 0L)]
    public void Int8RowBudgetMb_RejectsNegative(long? value, long expected)
        => With(EngineKnobs.Int8RowBudgetMb, value, () => Assert.Equal(expected, EngineKnobs.Int8RowBudgetMb.Value));

    /// <summary>Rejecting knob: non-positive falls back to 1024 MB.</summary>
    [Theory]
    [InlineData(0L, 1024L)]
    [InlineData(2048L, 2048L)]
    [InlineData(null, 1024L)]
    public void Im2colBandMb_RejectsNonPositive(long? value, long expected)
        => With(EngineKnobs.Im2colBandMb, value, () => Assert.Equal(expected, EngineKnobs.Im2colBandMb.Value));

    /// <summary>Clamping knob: out of range becomes the nearest bound, NOT the default. 17 is 16, not 4.</summary>
    [Theory]
    [InlineData(17, 16)]
    [InlineData(99, 16)]
    [InlineData(0, 1)]
    [InlineData(-3, 1)]
    [InlineData(8, 8)]
    [InlineData(null, 4)]
    public void GemvWpb_ClampsRatherThanRejecting(int? value, int expected)
        => With(EngineKnobs.GemvWpb, value, () => Assert.Equal(expected, EngineKnobs.GemvWpb.Value));

    /// <summary>Clamping knob: bounded to 8..2048.</summary>
    [Theory]
    [InlineData(4000L, 2048L)]
    [InlineData(2L, 8L)]
    [InlineData(64L, 64L)]
    [InlineData(null, 32L)]
    public void GraphArenaMb_ClampsToItsBounds(long? value, long expected)
        => With(EngineKnobs.GraphArenaMb, value, () => Assert.Equal(expected, EngineKnobs.GraphArenaMb.Value));

    /// <summary>Uncoerced knob: takes any value, including negatives, so the rejections above are specific rather than blanket.</summary>
    [Theory]
    [InlineData(8, 8)]
    [InlineData(-1, -1)]
    [InlineData(null, -1)]
    public void GemvKsplit_AcceptsAnyValue(int? value, int expected)
        => With(EngineKnobs.GemvKsplit, value, () => Assert.Equal(expected, EngineKnobs.GemvKsplit.Value));

    /// <summary>Uncoerced knob with a 1536 MB default.</summary>
    [Theory]
    [InlineData(2048L, 2048L)]
    [InlineData(null, 1536L)]
    public void AutopromoteHeadroomMb_DefaultsTo1536(long? value, long expected)
        => With(EngineKnobs.AutopromoteHeadroomMb, value, () => Assert.Equal(expected, EngineKnobs.AutopromoteHeadroomMb.Value));

    /// <summary>Contextual-default knobs stay null until something takes a position, so the call site keeps its own default.</summary>
    /// <remarks>Pinning a constant here would be a real bug: <c>numerics.fp8Native</c> would enable FP8 on pre-Ada
    /// cards that cannot execute it.</remarks>
    [Fact]
    public void ContextualDefaults_StayNullUntilSet()
    {
        Assert.Null(EngineKnobs.Fp8Native.Value);
        Assert.Null(EngineKnobs.Ltx2Shift.Value);
        Assert.Null(EngineKnobs.H3ChunkRows.Value);

        KnobStore.Set(EngineKnobs.Fp8Native, true);
        try
        {
            Assert.True(EngineKnobs.Fp8Native.Value);
        }
        finally
        {
            KnobStore.Clear(EngineKnobs.Fp8Native);
        }
        Assert.Null(EngineKnobs.Fp8Native.Value);
    }
}
