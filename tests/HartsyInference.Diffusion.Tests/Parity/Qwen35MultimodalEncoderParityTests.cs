using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>A tiny random Qwen3.5 trunk and vision tower run through transformers' <c>Qwen3_5Model</c>
/// (<c>tests/python-reference/qwen35_vl_tiny_reference.py</c>) must match <see cref="Qwen35TextEncoder"/> built from the
/// same checkpoint folder: one 256×256 image through the tower, its 64 tokens spliced over the Picture-1 placeholders, and
/// interleaved multimodal RoPE (sections 5/4/3 over 12 pairs) on the full-attention layers. The text-only prompt must
/// still match too.</summary>
public sealed unsafe class Qwen35MultimodalEncoderParityTests(ITestOutputHelper output)
{
    private const float Tolerance = 2e-4f;
    private const int ImageTokenId = 77;
    private static readonly string Dir = Path.Combine(AppContext.BaseDirectory, "Fixtures", "Qwen35Vl");

    [Fact]
    public void ImageInput_MatchesTransformers()
    {
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "expected.safetensors"));
        using Qwen35TextEncoder encoder = LoadEncoder();
        Assert.True(encoder.HasVision);
        Assert.Equal(64, encoder.CountImageTokens(256, 256));
        Assert.Equal((256, 256), encoder.VisionInputSize(256, 256));
        using Tensor image = ImageFrom(expected.GetTensor("image_u8"));
        using IBackend backend = new CpuBackend();
        using Tensor hidden = encoder.Encode(backend, Ids(expected.GetTensor("input_ids")), [image]);
        AssertClose(expected.GetTensor("hidden"), hidden, "image prompt hidden state");
        // The second prompt of an edit reuses the kept vision tokens instead of running the tower again.
        using Tensor again = encoder.Encode(backend, Ids(expected.GetTensor("input_ids")), [image]);
        Assert.True(new ReadOnlySpan<float>((void*)hidden.DataPointer, (int)hidden.ElementCount)
            .SequenceEqual(new ReadOnlySpan<float>((void*)again.DataPointer, (int)again.ElementCount)));
    }

    [Fact]
    public void TextOnly_StillMatchesTransformers()
    {
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "expected.safetensors"));
        using Qwen35TextEncoder encoder = LoadEncoder();
        using IBackend backend = new CpuBackend();
        using Tensor hidden = encoder.Encode(backend, Ids(expected.GetTensor("text_ids")));
        AssertClose(expected.GetTensor("text_hidden"), hidden, "text prompt hidden state");
    }

    [Fact]
    public void ImageInput_RefusesMismatchedPlaceholders()
    {
        using SafeTensorsLoader expected = new();
        expected.Load(Path.Combine(Dir, "expected.safetensors"));
        using Qwen35TextEncoder encoder = LoadEncoder();
        using Tensor image = ImageFrom(expected.GetTensor("image_u8"));
        using IBackend backend = new CpuBackend();
        int[] ids = Ids(expected.GetTensor("input_ids"));
        int[] shortRun = [.. ids.Where((id, i) => !(id == ImageTokenId && i == Array.IndexOf(ids, ImageTokenId)))];
        Assert.ThrowsAny<ArgumentException>(() => encoder.Encode(backend, shortRun, [image]));
    }

    private static Qwen35TextEncoder LoadEncoder()
    {
        string folder = Path.Combine(Dir, "encoder");
        Qwen35TextEncoder encoder = Qwen35TextEncoder.Load(folder, 512, "test");
        Qwen35VisionTower tower = Qwen35VisionTower.Resolve(encoder.SourceWeights, null, Path.Combine(folder, "config.json"), "test")
            ?? throw new InvalidOperationException("The fixture carries a vision tower.");
        encoder.AttachVision(tower);
        encoder.ImageTokenId = ImageTokenId;
        return encoder;
    }

    private static Tensor ImageFrom(Tensor u8)
    {
        Tensor image = new(new TensorShape(1, 3, u8.Shape[1], u8.Shape[2]), DType.F32);
        byte* src = (byte*)u8.DataPointer;
        float* dst = (float*)image.DataPointer;
        for (long i = 0; i < u8.ElementCount; i++) dst[i] = src[i] / 255f;
        return image;
    }

    private static int[] Ids(Tensor ids) => new ReadOnlySpan<int>((void*)ids.DataPointer, (int)ids.ElementCount).ToArray();

    private void AssertClose(Tensor expected, Tensor actual, string what)
    {
        Assert.Equal(expected.ElementCount, actual.ElementCount);
        ReadOnlySpan<float> e = new((void*)expected.DataPointer, (int)expected.ElementCount);
        ReadOnlySpan<float> a = new((void*)actual.DataPointer, (int)actual.ElementCount);
        float worst = 0f, peak = 0f;
        for (int i = 0; i < e.Length; i++)
        {
            Assert.False(float.IsNaN(a[i]), $"{what}: NaN at {i}");
            worst = MathF.Max(worst, MathF.Abs(e[i] - a[i]));
            peak = MathF.Max(peak, MathF.Abs(e[i]));
        }
        output.WriteLine($"{what}: max |Δ| = {worst:E3} (peak {peak:F3})");
        Assert.True(worst <= Tolerance * MathF.Max(1f, peak), $"{what}: max |Δ| {worst:E3} exceeds {Tolerance:E1}·max(1, {peak:F3}).");
    }
}
