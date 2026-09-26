using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

public sealed unsafe class SafeTensorsMergerTests
{
    [Theory]
    [InlineData(0x3F800000u, (ushort)0x3F80)] // 1.0 exact
    [InlineData(0x3F808000u, (ushort)0x3F80)] // tie, even stays
    [InlineData(0x3F818000u, (ushort)0x3F82)] // tie, odd rounds up to even
    [InlineData(0x3F807FFFu, (ushort)0x3F80)] // just below the tie
    [InlineData(0x3F808001u, (ushort)0x3F81)] // just above the tie
    [InlineData(0x7F7FFFFFu, (ushort)0x7F80)] // max float rounds to +inf, as torch does
    [InlineData(0x7FC00001u, (ushort)0x7FC0)] // NaN → canonical quiet NaN
    public void ToBf16_RoundsToNearestEvenLikeTorch(uint bits, ushort expected)
    {
        Assert.Equal(expected, SafeTensorsMerger.ToBf16(BitConverter.UInt32BitsToSingle(bits)));
    }

    [Fact]
    public void Write_MergesSourcesCastsFloatsAndKeepsExemptions()
    {
        string path = Path.Combine(Path.GetTempPath(), $"merge-{Guid.NewGuid():N}.safetensors");
        using Tensor a = new(new TensorShape(2), DType.F32);
        using Tensor b = new(new TensorShape(2), DType.F32);
        using Tensor ids = new(new TensorShape(2), DType.I64);
        ((float*)a.DataPointer)[0] = 1.5f; ((float*)a.DataPointer)[1] = -2f;
        ((float*)b.DataPointer)[0] = 3f; ((float*)b.DataPointer)[1] = 0.25f;
        ((long*)ids.DataPointer)[0] = 7; ((long*)ids.DataPointer)[1] = 9;
        try
        {
            string payload = SafeTensorsMerger.Write(path, [new("a", a), new("norm.b", b), new("ids", ids)], DType.BF16,
                new Dictionary<string, string> { ["modelspec.hash_sha256"] = "" }, key => key.StartsWith("norm", StringComparison.Ordinal));
            using SafeTensorsLoader loader = new();
            loader.Load(path);
            Assert.Equal(DType.BF16, loader.Descriptors["a"].DType);
            Assert.Equal(DType.F32, loader.Descriptors["norm.b"].DType);
            Assert.Equal(DType.I64, loader.Descriptors["ids"].DType);
            Assert.Equal("0x" + payload, loader.Metadata!["modelspec.hash_sha256"]);
            Tensor cast = loader.GetTensor("a");
            Assert.Equal((ushort)0x3FC0, ((ushort*)cast.DataPointer)[0]);
            Assert.Equal(9L, ((long*)loader.GetTensor("ids").DataPointer)[1]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void Write_RefusesADuplicateKey()
    {
        using Tensor a = new(new TensorShape(1), DType.F32);
        Assert.Throws<InvalidDataException>(() =>
            SafeTensorsMerger.Write(Path.Combine(Path.GetTempPath(), $"dup-{Guid.NewGuid():N}.safetensors"), [new("x", a), new("x", a)]));
    }

    /// <summary>Three one-byte tokens and three merged ones; the merges come back ordered by the merged token's rank, and
    /// each token's candidate splits by the ranks of their halves — transformers' TikTokenConverter order.</summary>
    [Fact]
    public void TiktokenConverter_RecoversMergesInConverterOrder()
    {
        string path = Path.Combine(Path.GetTempPath(), $"tok-{Guid.NewGuid():N}.tiktoken");
        string[] tokens = ["a", "b", "c", "ab", "abc", "bc"];
        File.WriteAllLines(path, tokens.Select((t, i) => $"{Convert.ToBase64String(Encoding.ASCII.GetBytes(t))} {i}"));
        try
        {
            string json = Encoding.UTF8.GetString(TiktokenConverter.ToHuggingFaceJson(path, "x", "NFC"));
            Assert.Contains("\"vocab\":{\"a\":0,\"b\":1,\"c\":2,\"ab\":3,\"abc\":4,\"bc\":5}", json, StringComparison.Ordinal);
            Assert.Contains("\"merges\":[[\"a\",\"b\"],[\"a\",\"bc\"],[\"ab\",\"c\"],[\"b\",\"c\"]]", json, StringComparison.Ordinal);
            Assert.StartsWith("{\"version\":\"1.0\",\"truncation\":null,\"padding\":null,\"added_tokens\":[],\"normalizer\":{\"type\":\"NFC\"}", json, StringComparison.Ordinal);
        }
        finally
        {
            File.Delete(path);
        }
    }
}
