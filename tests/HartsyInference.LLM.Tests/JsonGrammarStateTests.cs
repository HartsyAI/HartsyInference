using HartsyInference.LLM.Sampling;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Correctness gate for the hand-written incremental JSON validator
/// (<see cref="JsonGrammarState"/>) that <see cref="JsonGrammarStep"/> relies on to mask invalid tokens
/// during constrained decoding. A bug here would silently let the model emit malformed JSON (a false
/// negative — accepting something invalid) or make correct JSON impossible to produce (a false positive —
/// rejecting something valid, which would manifest as the sampler masking every token, forcing degenerate
/// output). Both classes of bug are checked explicitly.</summary>
public sealed class JsonGrammarStateTests
{
    private static (bool accepted, bool complete) Feed(string json)
    {
        JsonGrammarState state = new();
        bool ok = state.TryFeed(json);
        return (ok, ok && state.IsComplete);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("[]")]
    [InlineData("true")]
    [InlineData("false")]
    [InlineData("null")]
    [InlineData("0")]
    [InlineData("-0")]
    [InlineData("123")]
    [InlineData("-123")]
    [InlineData("1.5")]
    [InlineData("-1.5")]
    [InlineData("1e10")]
    [InlineData("1E10")]
    [InlineData("1e+10")]
    [InlineData("1e-10")]
    [InlineData("1.5e-10")]
    [InlineData("\"\"")]
    [InlineData("\"hello\"")]
    [InlineData("\"a\\\"b\\\\c\\/d\\be\\fg\\nh\\ri\\tj\"")] // all valid escape sequences
    [InlineData("\"\\u00e9\"")]  // unicode escape
    [InlineData("{\"a\":1}")]
    [InlineData("{\"a\":1,\"b\":2}")]
    [InlineData("[1,2,3]")]
    [InlineData("[1,\"two\",3.0,true,false,null,{},[]]")]
    [InlineData("{\"a\":[1,2,3]}")]
    [InlineData("{\"a\":{\"b\":{\"c\":1}}}")] // nested objects
    [InlineData("[[1,2],[3,4]]")] // nested arrays
    [InlineData("  {  \"a\"  :  1  }  ")] // whitespace tolerance
    [InlineData("{\"a\":\"b\",\"c\":{\"d\":true,\"e\":null,\"f\":false}}")]
    public void Accepts_And_Completes_ValidJson(string json)
    {
        (bool accepted, bool complete) = Feed(json);
        Assert.True(accepted, $"should accept: {json}");
        Assert.True(complete, $"should be complete: {json}");
    }

    [Theory]
    [InlineData("{")]        // unclosed object
    [InlineData("[")]        // unclosed array
    [InlineData("{\"a\":1")] // unclosed object with content
    [InlineData("\"unterminated")]
    [InlineData("-")]        // sign with no digit
    [InlineData("1.")]       // no frac digit
    [InlineData("1e")]       // no exp digit
    public void Accepts_But_NotYetComplete_PartialJson(string prefix)
    {
        (bool accepted, bool complete) = Feed(prefix);
        Assert.True(accepted, $"should be a valid (incomplete) prefix: {prefix}");
        Assert.False(complete, $"should NOT be complete: {prefix}");
    }

    [Theory]
    [InlineData("{,}")]              // comma with no key
    [InlineData("[1,]")]             // trailing comma
    [InlineData("{\"a\":}")]         // missing value
    [InlineData("{a:1}")]            // unquoted key
    [InlineData("'single'")]         // single-quoted string
    [InlineData("01")]               // leading zero
    [InlineData(".5")]               // no leading digit
    [InlineData("NaN")]              // not valid JSON
  [InlineData("Infinity")]
    [InlineData("{}}")]              // extra closing brace after complete root value
    [InlineData("[1 2]")]            // missing comma between array elements
    [InlineData("{\"a\" \"b\"}")]    // missing colon
    [InlineData("truee")]            // literal with trailing garbage — the 5th char 'e' after "true" completes at Done, then rejects
    public void Rejects_InvalidJson(string json)
    {
        (bool accepted, _) = Feed(json);
        Assert.False(accepted, $"should reject: {json}");
    }

    [Fact]
    public void Clone_IsolatesTrialFeedFromRealState()
    {
        JsonGrammarState real = new();
        Assert.True(real.TryFeed("{\"a\":"));
        Assert.False(real.IsComplete);

        JsonGrammarState trial = real.Clone();
        Assert.True(trial.TryFeed("1}"));
        Assert.True(trial.IsComplete);

        // The real state must be untouched by feeding the clone.
        Assert.False(real.IsComplete);
        Assert.True(real.TryFeed("2}")); // real state can still independently complete differently
        Assert.True(real.IsComplete);
    }

    [Fact]
    public void IncrementalFeed_MatchesWholeStringFeed()
    {
        const string json = "{\"a\":[1,2,{\"b\":true}],\"c\":null}";
        JsonGrammarState whole = new();
        Assert.True(whole.TryFeed(json));
        Assert.True(whole.IsComplete);

        JsonGrammarState incremental = new();
        foreach (char c in json)
            Assert.True(incremental.TryFeed(c.ToString()), $"failed feeding '{c}'");
        Assert.True(incremental.IsComplete);
    }
}
