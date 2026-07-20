using HartsyInference.LLM.Sampling;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Correctness gate for <see cref="SentinelJsonGrammarStep"/>: grammar masking must stay OFF for
/// plain chat text (the whole point of scoping vs. <see cref="JsonGrammarStep"/>'s always-on masking) and
/// activate/deactivate at exactly the right characters around a text sentinel. Uses a char-level stub
/// tokenizer (token id == char code) so <see cref="SamplerChain"/>-style per-token
/// <c>Apply(logits, history)</c> calls can be driven one character at a time, matching how the real sampler
/// loop calls it (history grows by exactly one id per call).</summary>
public sealed class SentinelJsonGrammarStepTests
{
    private const int VocabSize = 128; // covers printable ASCII by id == char code

    private sealed class CharTokenizer : ILlmTokenizer
    {
        public int[] Encode(string text, bool addSpecial) => throw new NotSupportedException();
        public int[] EncodeOrdinary(string text) => throw new NotSupportedException();
        public string Decode(IReadOnlyList<int> ids) => string.Concat(Array.ConvertAll([.. ids], i => (char)i));
        public int? SpecialId(string token) => null;
        public int? BosId => null;
        public int? EosId => null;
        public IReadOnlyList<int> StopIds => [];
        public string? BosToken => null;
        public string? EosToken => null;
    }

    private static bool IsMasked(Span<float> logits, char c) => float.IsNegativeInfinity(logits[c]);

    /// <summary>Drives the step character-by-character exactly like a real decode loop: history grows by one
    /// id per call, and the logits mask computed BEFORE appending each character is asserted against.</summary>
    private sealed class Driver
    {
        private readonly SentinelJsonGrammarStep _step;
        private readonly List<int> _history = new();

        public Driver(string openSentinel) =>
            _step = new SentinelJsonGrammarStep(new CharTokenizer(), VocabSize, openSentinel);

        /// <summary>Computes the mask that would apply to the NEXT token (i.e. before committing
        /// <paramref name="next"/>), asserts <paramref name="next"/> was/wasn't masked per
        /// <paramref name="expectMasked"/>, then commits it to history.</summary>
        public float[] StepMaskThenCommit(char next, bool? expectMasked = null)
        {
            float[] logits = new float[VocabSize];
            _step.Apply(logits, _history);
            if (expectMasked is bool exp)
            {
                Assert.True(exp == IsMasked(logits, next),
                    $"'{next}' masked={IsMasked(logits, next)}, expected {exp}");
            }
            _history.Add(next);
            return logits;
        }

        public float[] CurrentMask()
        {
            float[] logits = new float[VocabSize];
            _step.Apply(logits, _history);
            return logits;
        }
    }

    [Fact]
    public void OutsideSentinel_PlainText_NeverMasked()
    {
        Driver d = new("<tool_call>");
        // 'H' at JSON ValueStart would be rejected by JsonGrammarState — must NOT be masked while inactive.
        foreach (char c in "Hello, how can I help you today? { not json ] } weird chars ok")
        {
            d.StepMaskThenCommit(c, expectMasked: false);
        }
    }

    [Fact]
    public void SentinelActivatesGrammarMasking_ForTheSpanAfterIt()
    {
        Driver d = new("<tool_call>");
        foreach (char c in "some text before <tool_call>") d.StepMaskThenCommit(c);

        // Now active: 'H' (invalid JSON value-start) must be masked, '{' (valid) must not be.
        float[] afterSentinel = d.CurrentMask();
        Assert.True(IsMasked(afterSentinel, 'H'));
        Assert.False(IsMasked(afterSentinel, '{'));
    }

    [Fact]
    public void DeactivatesTheInstantJsonValueCompletes_LettingCloseTagThrough()
    {
        Driver d = new("<tool_call>");
        foreach (char c in "<tool_call>") d.StepMaskThenCommit(c);

        // Feed a complete JSON object; every char here must be UNmasked at the moment it's committed (the
        // grammar must accept its own emitted text).
        const string body = "{\"name\":\"x\",\"arguments\":{}}";
        foreach (char c in body) d.StepMaskThenCommit(c, expectMasked: false);

        // The instant the value completes, masking must lift — '<' (start of the closing tag) must NOT be
        // masked even though '<' is never valid inside the JSON grammar.
        float[] afterComplete = d.CurrentMask();
        Assert.False(IsMasked(afterComplete, '<'));

        // And the rest of the closing tag plus trailing prose streams completely unconstrained.
        foreach (char c in "</tool_call> hope that helps!") d.StepMaskThenCommit(c, expectMasked: false);
    }

    [Fact]
    public void MultipleSpansInOneGeneration_EachActivateAndDeactivateIndependently()
    {
        Driver d = new("<tool_call>");
        foreach (char c in "<tool_call>{\"a\":1}") d.StepMaskThenCommit(c);
        // Deactivated after the first span — plain text in between must be unconstrained.
        foreach (char c in "</tool_call> ok, next <tool_call>") d.StepMaskThenCommit(c, expectMasked: false);

        // Second span: must be active again (fresh state, not polluted by the first span's completed object).
        float[] mask = d.CurrentMask();
        Assert.True(IsMasked(mask, 'H'));
        Assert.False(IsMasked(mask, '['));

        foreach (char c in "[1,2,3]") d.StepMaskThenCommit(c, expectMasked: false);
        float[] afterSecond = d.CurrentMask();
        Assert.False(IsMasked(afterSecond, '<')); // deactivated again
    }

    [Fact]
    public void SentinelSplitAcrossManySingleCharacterCalls_StillActivatesExactlyAtTheBoundary()
    {
        Driver d = new("<tool_call>");
        const string prefix = "prefix text <tool_call"; // one char short of the sentinel (missing trailing '>')
        foreach (char c in prefix) d.StepMaskThenCommit(c, expectMasked: false);

        // Not yet armed — 'H' still unconstrained.
        Assert.False(IsMasked(d.CurrentMask(), 'H'));

        d.StepMaskThenCommit('>', expectMasked: false); // completes "<tool_call>" text, itself still unconstrained (part of the sentinel, not JSON)

        // Now armed.
        Assert.True(IsMasked(d.CurrentMask(), 'H'));
    }

    [Fact]
    public void TruncationMidSpan_NeverThrows()
    {
        Driver d = new("<tool_call>");
        foreach (char c in "<tool_call>{\"a\":") d.StepMaskThenCommit(c);
        // Generation just stops here (max tokens hit). Nothing further is fed — must not have thrown getting
        // here, and querying the mask again (as a caller finalizing the round might) must still be safe.
        Exception? ex = Record.Exception(() => d.CurrentMask());
        Assert.Null(ex);
    }

    [Fact]
    public void FromOptions_RequiresTokenizerAndVocabSize_WhenSentinelSet()
    {
        SamplingOptions opts = SamplingOptions.Default with { JsonModeSentinel = "<tool_call>" };
        Assert.Throws<ArgumentException>(() => SamplerChain.FromOptions(opts));
    }

    [Fact]
    public void FromOptions_JsonModeWins_WhenBothSet()
    {
        // Doesn't throw building the chain with only a tokenizer (JsonMode's own requirement) even though
        // JsonModeSentinel is also set — proves JsonMode takes priority and the sentinel path is skipped.
        SamplingOptions opts = SamplingOptions.Default with
        {
            Greedy = true, JsonMode = true, JsonModeSentinel = "<tool_call>",
        };
        SamplerChain chain = SamplerChain.FromOptions(opts, new CharTokenizer(), VocabSize);
        Assert.NotNull(chain);
    }

    [Fact]
    public void HasJsonConstraint_TrueForEitherMode()
    {
        Assert.False(SamplingOptions.Default.HasJsonConstraint);
        Assert.True((SamplingOptions.Default with { JsonMode = true }).HasJsonConstraint);
        Assert.True((SamplingOptions.Default with { JsonModeSentinel = "<x>" }).HasJsonConstraint);
    }
}
