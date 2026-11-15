using System.Text.RegularExpressions;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The rewrite of Oniguruma (Ruby syntax) quantifiers that .NET rejects or reads differently in <c>tokenizer.json</c> Split patterns.</summary>
public sealed class OnigurumaPatternTests
{
    [Theory]
    // an interval then + repeats the interval (ERNIE-5.5's right-to-left digit grouping)
    [InlineData(@"\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)", @"\A\p{Nd}{1,2}(?=(?:\p{Nd}{3})+\z)")]
    [InlineData(@"(ab){2}+c", @"(?:(ab){2})+c")]
    [InlineData(@"[xy]{2,}+", @"(?:[xy]{2,})+")]
    // other quantifiers then + are possessive
    [InlineData(@"a++b", @"(?>a+)b")]
    [InlineData(@"\s*+x", @"(?>\s*)x")]
    // {,n} is {0,n}
    [InlineData(@"a{,3}", @"a{0,3}")]
    // a (?#) comment is dropped: its text is not pattern, and an escaped ')' (which .NET would stop at) does not end it
    [InlineData(@"a(?#[ not a class (x)b+", @"ab+")]
    [InlineData(@"a(?#x\)y)b", @"ab")]
    public void Rewrites_the_forms_dotnet_reads_differently(string onig, string dotNet)
    {
        Assert.Equal(dotNet, OnigurumaPattern.ToDotNet(onig));
        _ = new Regex(dotNet); // the rewrite must be a pattern .NET compiles
    }

    [Theory]
    [InlineData(@"'s|'t| ?\p{L}+| ?\p{N}+|\s+(?!\S)|\s+")]
    [InlineData(@"[!""#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+")]
    [InlineData(@"\p{Nd}{1,510}(?=(?>\p{Nd}{510})*(?:\P{Nd}|$))|\G\p{Nd}{510}")]
    [InlineData(@"(?<name>a+?)(?i:b)(?<=c)x{2}?\{literal}")]
    // under (?x) a '(' in a '#' line comment is not a group, so the pattern is not refused as unbalanced
    [InlineData("(?x)a # (\nb")]
    public void Leaves_patterns_without_those_forms_unchanged(string pattern) => Assert.Equal(pattern, OnigurumaPattern.ToDotNet(pattern));

    [Fact]
    public void Rewritten_digit_grouping_splits_like_hugging_face()
    {
        // tokenizers splits "1234567" into 1 | 234 | 567 with ERNIE-5.5's two grouping stages.
        Regex lead = new(OnigurumaPattern.ToDotNet(@"\A\p{Nd}{1,2}(?=\p{Nd}{3}+\z)"));
        Assert.Equal("1", lead.Match("1234567").Value);
        Assert.DoesNotMatch(lead, "123456");
    }

    [Theory]
    [InlineData("(++)")]
    // Oniguruma nests these repeats; the rewrite refuses them rather than guess
    [InlineData("a{2}{3}+")]
    [InlineData(@"\d+{2}+")]
    // a comment that is never closed, including one whose only ')' is escaped
    [InlineData("a(?#oops")]
    [InlineData(@"a(?#x\)")]
    public void A_quantifier_with_nothing_to_repeat_is_refused(string pattern) => Assert.Throws<NotSupportedException>(() => OnigurumaPattern.ToDotNet(pattern));
}
