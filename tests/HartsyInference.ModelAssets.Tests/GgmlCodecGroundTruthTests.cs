using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Gguf.Codecs;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Every GGUF codec decodes real blocks to exactly the floats llama.cpp's own dequantizer gives. The blocks and floats in
/// <c>Fixtures/GgmlCodecs/ggml_codec_blocks.safetensors</c> come from ggml's C quantizer and <c>type_traits.to_float</c>
/// (<c>tests/python-reference/dump_ggml_codec_blocks.py</c>), four rows of 512 values per type, one row scaled a thousandfold.</summary>
public sealed class GgmlCodecGroundTruthTests
{
    private static readonly string FixturePath = Path.Combine(HartsyInference.Tests.Common.RepoRoot.Path, "tests",
        "HartsyInference.ModelAssets.Tests", "Fixtures", "GgmlCodecs", "ggml_codec_blocks.safetensors");

    /// <summary>Q8_1 is ggml's activation format: its quantizer writes no file blocks, so the fixture cannot carry it.</summary>
    private static readonly DType[] NotInFixture = [DType.Q8_1];

    public static TheoryData<string> Types()
    {
        TheoryData<string> names = new();
        foreach (DType dtype in GgufCodecRegistry.Registered.Except(NotInFixture).OrderBy(d => d.Name, StringComparer.Ordinal)) names.Add(dtype.Name);
        return names;
    }

    [Theory]
    [MemberData(nameof(Types))]
    public unsafe void Codec_MatchesGgmlBitForBit(string name)
    {
        DType dtype = GgufCodecRegistry.Registered.Single(d => d.Name == name);
        using SafeTensorsLoader fixture = new();
        fixture.Load(FixturePath);
        Assert.True(fixture.Descriptors.ContainsKey($"{name}.blocks"), $"Fixture has no {name} blocks; rerun dump_ggml_codec_blocks.py.");
        byte[] blocks = fixture.GetTensor($"{name}.blocks").AsReadOnlySpan<byte>().ToArray();
        float[] expected = fixture.GetTensor($"{name}.values").AsReadOnlySpan<float>().ToArray();
        Assert.Equal(expected.Length / dtype.BlockElementCount * dtype.BlockByteSize, blocks.Length);

        float[] got = new float[expected.Length];
        fixed (byte* src = blocks)
        fixed (float* dst = got)
        {
            GgufCodecRegistry.Get(dtype).DequantizeToF32(src, dst, got.Length);
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(got[i]) != BitConverter.SingleToInt32Bits(expected[i]))
                Assert.Fail($"{name}[{i}] (block {i / dtype.BlockElementCount}): got {got[i]:R}, ggml {expected[i]:R}");
        }
    }
}
