using System.Text;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The shared ComfyUI attention-hint parser, used by the Fibo 2.0 loader and the MiniMax-H3 router.</summary>
public sealed class ComfyAttentionHintTests
{
    [Fact]
    public void RealFiboBlob_ParsesToItsBackend()
    {
        byte[] bytes = Encoding.UTF8.GetBytes("{\"attention\": \"comfy_kitchen_int8\"}");

        Assert.Equal(35, bytes.Length);
        Assert.Equal("comfy_kitchen_int8", ComfyAttentionHint.TryParseBackend(bytes));
    }

    [Fact]
    public void TrailingNulPadding_IsIgnored()
    {
        byte[] bytes = Encoding.UTF8.GetBytes("{\"attention\": \"comfy_kitchen_int8\"}\0\0\0");

        Assert.Equal("comfy_kitchen_int8", ComfyAttentionHint.TryParseBackend(bytes));
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json")]
    [InlineData("[1]")]
    [InlineData("{\"backend\": \"x\"}")]
    [InlineData("{\"attention\": 4}")]
    [InlineData("{\"attention\": \"\"}")]
    public void MalformedBlob_ReturnsNull(string text)
    {
        Assert.Null(ComfyAttentionHint.TryParseBackend(Encoding.UTF8.GetBytes(text)));
    }

    [Fact]
    public void IsHint_MatchesTheSuffixOnly()
    {
        Assert.True(ComfyAttentionHint.IsHint("layers.0.attention.comfy_attention.config"));
        Assert.True(ComfyAttentionHint.IsHint("blocks.3.attn.comfy_attention.config"));
        Assert.False(ComfyAttentionHint.IsHint("layers.0.attention.comfy_quant"));
    }
}
