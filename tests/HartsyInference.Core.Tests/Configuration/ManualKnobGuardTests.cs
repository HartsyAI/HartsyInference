using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tests.MemoryManagement;
using Xunit;

namespace HartsyInference.Core.Tests.Configuration;

/// <summary>Exercises the hand-written coercion on each manually declared knob against the bound its old call site enforced.</summary>
/// <remarks><see cref="KnobGrammarTests"/> pins the mechanisms generically; this pins the individual lambdas, which
/// is where a mistyped bound would ship. Only the out-of-range cases prove anything — every happy path passes with
/// the coercion deleted.
/// <para>Note the two shapes deliberately disagree: a rejecting knob sends an out-of-range value to its default,
/// while a clamping knob sends it to the nearest bound. <c>numerics.gemvWpb</c> at 17 is 16, NOT the default 4.</para></remarks>
[Collection(EnvironmentSensitiveCollection.Name)]
public sealed class ManualKnobGuardTests
{
    private static void With(string env, string? value, Action body)
    {
        string? previous = Environment.GetEnvironmentVariable(env);
        try
        {
            Environment.SetEnvironmentVariable(env, value);
            body();
        }
        finally
        {
            Environment.SetEnvironmentVariable(env, previous);
        }
    }

    /// <summary>Rejecting knobs: a value outside the accepted range falls back to the declared default.</summary>
    [Theory]
    [InlineData("-5", 14L)]
    [InlineData("0", 14L)]
    [InlineData("20", 20L)]
    [InlineData("notanumber", 14L)]
    [InlineData(null, 14L)]
    public void AudioEvictBelowGb_RejectsNonPositive(string? value, long expected)
        => With("HARTSY_AUDIO_EVICT_BELOW_GB", value, () => Assert.Equal(expected, EngineKnobs.AudioEvictBelowGb.Value));

    [Theory]
    [InlineData("0", 8192)]
    [InlineData("-1", 8192)]
    [InlineData("4096", 4096)]
    [InlineData(null, 8192)]
    public void SageF16MinSkv_RejectsNonPositive(string? value, int expected)
        => With("HARTSY_SAGE_F16_MIN_SKV", value, () => Assert.Equal(expected, EngineKnobs.SageF16MinSkv.Value));

    [Theory]
    [InlineData("-1", 0L)]
    [InlineData("512", 512L)]
    [InlineData(null, 0L)]
    public void Int8RowBudgetMb_RejectsNegative(string? value, long expected)
        => With("HARTSY_INT8_ROW_BUDGET_MB", value, () => Assert.Equal(expected, EngineKnobs.Int8RowBudgetMb.Value));

    [Theory]
    [InlineData("0", 1024L)]
    [InlineData("2048", 2048L)]
    [InlineData(null, 1024L)]
    public void Im2colBandMb_RejectsNonPositive(string? value, long expected)
        => With("HARTSY_IM2COL_BAND_MB", value, () => Assert.Equal(expected, EngineKnobs.Im2colBandMb.Value));

    /// <summary>Clamping knobs: an out-of-range value becomes the nearest bound, NOT the default. 17 is 16, not 4.</summary>
    [Theory]
    [InlineData("17", 16)]
    [InlineData("99", 16)]
    [InlineData("0", 1)]
    [InlineData("-3", 1)]
    [InlineData("8", 8)]
    [InlineData(null, 4)]
    [InlineData("notanumber", 4)]
    public void GemvWpb_ClampsRatherThanRejecting(string? value, int expected)
        => With("HARTSY_GEMV_WPB", value, () => Assert.Equal(expected, EngineKnobs.GemvWpb.Value));

    [Theory]
    [InlineData("4000", 2048L)]
    [InlineData("2", 8L)]
    [InlineData("64", 64L)]
    [InlineData(null, 32L)]
    public void GraphArenaMb_ClampsToItsBounds(string? value, long expected)
        => With("HARTSY_GRAPH_ARENA_MB", value, () => Assert.Equal(expected, EngineKnobs.GraphArenaMb.Value));

    /// <summary>Uncoerced numeric knobs take any parsable value, including negatives, exactly as the bare TryParse did.</summary>
    [Theory]
    [InlineData("8", 8)]
    [InlineData("-1", -1)]
    [InlineData(null, -1)]
    public void GemvKsplit_AcceptsAnyParsableValue(string? value, int expected)
        => With("HARTSY_GEMV_KSPLIT", value, () => Assert.Equal(expected, EngineKnobs.GemvKsplit.Value));

    [Theory]
    [InlineData("2048", 2048L)]
    [InlineData(null, 1536L)]
    public void AutopromoteHeadroomMb_DefaultsTo1536(string? value, long expected)
        => With("HARTSY_AUTOPROMOTE_HEADROOM_MB", value, () => Assert.Equal(expected, EngineKnobs.AutopromoteHeadroomMb.Value));

    /// <summary>Contextual-default knobs stay null unless a recognized spelling takes a position, so the call site keeps its own default.</summary>
    [Theory]
    [InlineData("1", true)]
    [InlineData("true", true)]
    [InlineData("0", false)]
    [InlineData("false", false)]
    [InlineData("garbage", null)]
    [InlineData(null, null)]
    [InlineData("", null)]
    public void Fp8Native_IsAnOverrideNotAConstant(string? value, bool? expected)
        => With("HARTSY_FP8_NATIVE", value, () => Assert.Equal(expected, EngineKnobs.Fp8Native.Value));

    [Theory]
    [InlineData("3.5", 3.5f)]
    [InlineData("notanumber", null)]
    [InlineData(null, null)]
    public void Ltx2Shift_IsAnOverrideNotAConstant(string? value, float? expected)
        => With("HARTSY_LTX2_SHIFT", value, () => Assert.Equal(expected, EngineKnobs.Ltx2Shift.Value));

    [Theory]
    [InlineData("512", 512)]
    [InlineData("0", 0)]
    [InlineData(null, null)]
    public void H3ChunkRows_IsAnOverrideNotAConstant(string? value, int? expected)
        => With("HARTSY_H3_CHUNK_ROWS", value, () => Assert.Equal(expected, EngineKnobs.H3ChunkRows.Value));
}
