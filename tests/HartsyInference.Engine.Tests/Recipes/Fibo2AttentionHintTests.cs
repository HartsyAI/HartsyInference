using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.ModelAssets.Quant;

namespace HartsyInference.Engine.Tests;

/// <summary>The ComfyUI attention hints in the Fibo 2.0 Comfy-Org int8 builds. The 35-byte blob matches the real
/// <c>layers.0.attention.comfy_attention.config</c> of <c>briaai/RMBG-3.0-Comfy</c> (read from its header and data, 2026-11-24).</summary>
public sealed class Fibo2AttentionHintTests
{
    private const string Hint = "layers.0.attention.comfy_attention.config";

    private static Tensor Blob(string text, int padding = 0)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(text);
        Tensor tensor = new(new TensorShape(bytes.Length + padding), DType.U8);
        Span<byte> span = tensor.AsSpan<byte>();
        bytes.CopyTo(span);
        return tensor;
    }

    [Fact]
    public void RealCheckpointBlob_IsThirtyFiveBytesAndNamesComfyKitchenInt8()
    {
        const string real = "{\"attention\": \"comfy_kitchen_int8\"}";
        Assert.Equal(35, Encoding.UTF8.GetByteCount(real));

        using Tensor blob = Blob(real);
        Assert.Equal("comfy_kitchen_int8", Fibo2AttentionHints.Read(blob, Hint));
    }

    [Fact]
    public void Requested_ReturnsTheDistinctBackendsOfEveryHint()
    {
        using Tensor a = Blob("{\"attention\": \"comfy_kitchen_int8\"}");
        using Tensor b = Blob("{\"attention\": \"comfy_kitchen_int8\"}");
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal)
        {
            [Hint] = a,
            ["layers.1.attention.comfy_attention.config"] = b,
        };

        Assert.Equal(new[] { "comfy_kitchen_int8" }, Fibo2AttentionHints.Requested(weights));
    }

    [Fact]
    public void Requested_IsEmptyWhenTheCheckpointHasNoHints()
    {
        using Tensor weight = new(new TensorShape(2, 2), DType.F16);
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal) { ["layers.0.attention.qkv.weight"] = weight };

        Assert.Empty(Fibo2AttentionHints.Requested(weights));
    }

    [Fact]
    public void Read_Tolerates_TrailingNulPadding()
    {
        using Tensor blob = Blob("{\"attention\": \"comfy_kitchen_int8\"}", padding: 5);

        Assert.Equal("comfy_kitchen_int8", Fibo2AttentionHints.Read(blob, Hint));
    }

    [Fact]
    public void Requested_AcceptsAnUnknownBackend_AndTheWarningNamesIt()
    {
        using Tensor blob = Blob("{\"attention\": \"sage_fp8\"}");
        Dictionary<string, Tensor> weights = new(StringComparer.Ordinal) { [Hint] = blob };

        IReadOnlyList<string> names = Fibo2AttentionHints.Requested(weights);

        Assert.Equal(new[] { "sage_fp8" }, names);
        Assert.Contains("sage_fp8", Fibo2AttentionHints.Describe("sage_fp8"));
        Assert.Contains("float attention by default", Fibo2AttentionHints.Describe("sage_fp8"));
    }

    [Fact]
    public void Describe_NamesComfyKitchenInt8AsTheInt8AttentionDifference()
    {
        string text = Fibo2AttentionHints.Describe("comfy_kitchen_int8");

        Assert.Contains("comfy_kitchen_int8", text);
        Assert.Contains("float attention by default", text);
        Assert.Contains("opt-in", text);
        Assert.Contains("numerically unverified", text);
    }

    [Theory]
    [InlineData("not json")]
    [InlineData("[1, 2]")]
    [InlineData("{\"backend\": \"comfy_kitchen_int8\"}")]
    [InlineData("{\"attention\": 3}")]
    [InlineData("{\"attention\": \"\"}")]
    public void Read_RefusesAnythingThatIsNotTheAttentionForm(string text)
    {
        using Tensor blob = Blob(text);

        Assert.Throws<InvalidDataException>(() => Fibo2AttentionHints.Read(blob, Hint));
    }

    [Fact]
    public void Read_RefusesANonU8Blob()
    {
        using Tensor wrong = new(new TensorShape(35), DType.F16);

        Assert.Throws<InvalidDataException>(() => Fibo2AttentionHints.Read(wrong, Hint));
    }

    [Fact]
    public void IsHint_MatchesOnlyTheAttentionConfigKey()
    {
        Assert.True(ComfyAttentionHint.IsHint(Hint));
        Assert.False(ComfyAttentionHint.IsHint("layers.0.attention.qkv.comfy_quant"));
        Assert.False(ComfyAttentionHint.IsHint("layers.0.attention.qkv.weight"));
    }

    [Fact]
    public void HalfPrecisionWeights_DropsTheHintAndKeepsTheWeights()
    {
        using Tensor weight = new(new TensorShape(2, 2), DType.F16);
        using Tensor hint = Blob("{\"attention\": \"comfy_kitchen_int8\"}");
        Dictionary<string, Tensor> source = new(StringComparer.Ordinal)
        {
            ["layers.0.attention.qkv.weight"] = weight,
            [Hint] = hint,
        };
        List<IDisposable> owned = [];

        Dictionary<string, Tensor> result = Fibo2WeightPreparation.HalfPrecisionWeights(source, owned);

        Assert.Single(result);
        Assert.True(result.ContainsKey("layers.0.attention.qkv.weight"));
        Assert.Empty(owned);
    }
}
