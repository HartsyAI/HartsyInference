using System.Text;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the shapes of <c>{layer}.comfy_quant</c> actually observed in LTX 2.5 / MiniMax-H3 checkpoints,
/// plus the reject paths. A descriptor that silently parses to the wrong group size dequantizes every weight in that
/// layer with the wrong rotation and produces plausible-looking noise, so the parse is worth pinning.</summary>
public sealed class ComfyQuantDescriptorTests
{
    private static ComfyQuantDescriptor? Parse(string json) => ComfyQuantDescriptor.TryParse(Encoding.UTF8.GetBytes(json));

    [Theory]
    [InlineData("""{"format":123}""")]
    [InlineData("""{"format":{"name":"int8_tensorwise"}}""")]
    public void RejectsANonStringFormatInsteadOfThrowing(string json)
    {
        // JsonElement.GetString throws InvalidOperationException — not a JsonException — on a non-string, so
        // without an explicit ValueKind check a malformed blob crashes the checkpoint load instead of being skipped.
        Assert.Null(Parse(json));
    }

    [Fact]
    public void ParsesTheRealInt8ConvRotBlob()
    {
        ComfyQuantDescriptor? descriptor = Parse("""{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256, "per_row": true}""");
        Assert.NotNull(descriptor);
        Assert.Equal("int8_tensorwise", descriptor.Format);
        Assert.Equal(256, descriptor.ConvRotGroupSize);
        Assert.False(descriptor.FullPrecisionMatMul);
    }

    [Theory]
    [InlineData("not json at all")]
    [InlineData("[]")]
    [InlineData("""{"format":null}""")]
    public void RejectsWhatIsNotADescriptor(string json) => Assert.Null(Parse(json));

    [Fact]
    public void RejectsAnEmptyBlob() => Assert.Null(ComfyQuantDescriptor.TryParse(ReadOnlySpan<byte>.Empty));

    [Fact]
    public void RejectsABlobTooLargeToBeADescriptor()
    {
        // The 4096-byte ceiling is what stops a real weight tensor that happens to sit under a `.comfy_quant`-suffixed
        // key from being UTF-8 decoded and JSON-parsed.
        string padding = new string('x', 5000);
        Assert.Null(Parse($$"""{"format":"int8_tensorwise","note":"{{padding}}"}"""));
    }
}
