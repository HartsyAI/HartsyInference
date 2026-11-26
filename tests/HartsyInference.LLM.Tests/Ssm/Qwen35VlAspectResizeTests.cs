using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The aspect-preserving resize the Qwen3.5-VL checkpoint's processor applies before its tower (<c>smart_resize</c> with factor 32 and the
/// shipped <c>[65536, 16777216]</c> pixel bounds). The expected sizes are transformers 5.22.2's <c>smart_resize</c> output for the same inputs.</summary>
public sealed class Qwen35VlAspectResizeTests
{
    private const int Factor = 32, Min = 65536, Max = 16777216;

    [Theory]
    [InlineData(768, 1024, 768, 1024)]
    [InlineData(1024, 768, 1024, 768)]
    [InlineData(480, 640, 480, 640)]
    [InlineData(100, 1000, 96, 992)]     // rounds each side to 32 multiples, keeping the aspect ratio
    [InlineData(2000, 1000, 1984, 992)]
    [InlineData(64, 256, 128, 512)]
    [InlineData(20, 30, 224, 320)]       // under the minimum: scaled up to at least 65536 pixels
    [InlineData(4096, 4096, 4096, 4096)] // exactly the maximum: not scaled down
    [InlineData(3000, 4000, 3008, 4000)]
    [InlineData(333, 777, 320, 768)]
    public void SmartResize_MatchesTheCheckpointProcessor(int h, int w, int wantH, int wantW)
    {
        (int gotH, int gotW) = VlmImagePreprocessor.SmartResize(h, w, Factor, Min, Max);
        Assert.Equal((wantH, wantW), (gotH, gotW));
    }

    [Theory]
    [InlineData(1080, 1920, 576, 1024)]   // 576 merged tokens: the engine's default budget
    [InlineData(3000, 4000, 640, 864)]
    [InlineData(4096, 4096, 768, 768)]
    [InlineData(768, 1024, 640, 864)]
    public void SmartResize_AtTheEngineDefaultBudget_MatchesTheProcessorWithThatMaximum(int h, int w, int wantH, int wantW)
    {
        (int gotH, int gotW) = VlmImagePreprocessor.SmartResize(h, w, Factor, Min, Qwen3VlEncoder.DefaultMaxPixels);
        Assert.Equal((wantH, wantW), (gotH, gotW));
        Assert.True((long)gotH * gotW <= Qwen3VlEncoder.DefaultMaxPixels);
    }

    [Fact]
    public void PreprocessAspect_LargePhoto_StaysAtTheDefaultTokenBudget()
    {
        // A 3000×4000 photo: 640×864 px is 20×27 = 540 merged tokens, under the 576 the default budget allows.
        const int w = 4000, h = 3000;
        byte[] rgb = new byte[w * h * 3];
        using Tensor px = VlmImagePreprocessor.PreprocessAspect(rgb, w, h, Factor, Min, Qwen3VlEncoder.DefaultMaxPixels, [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f]);
        Assert.Equal(new TensorShape(1, 3, 640, 864), px.Shape);
        Assert.True(Qwen3VlEncoder.MergedGrid(px, Factor / 2, 2).Tokens <= 576);
    }

    [Fact]
    public void SmartResize_OverTheMaximum_ScalesDownBelowIt()
    {
        (int h, int w) = VlmImagePreprocessor.SmartResize(6000, 6000, Factor, Min, Max);
        Assert.True((long)h * w <= Max);
        Assert.Equal(0, h % Factor);
        Assert.Equal(0, w % Factor);
    }

    [Fact]
    public void SmartResize_RefusesAnAspectRatioAbove200()
    {
        Assert.Throws<ArgumentException>(() => VlmImagePreprocessor.SmartResize(10, 3000, Factor, Min, Max));
    }

    [Fact]
    public void PreprocessAspect_NonSquare_OutputIsTheResizedShapeAndItsGridIsRead()
    {
        // 100×1000 px → 96×992: 3×31 merged tokens, read from the tensor by MergedGrid.
        byte[] rgb = new byte[100 * 1000 * 3];
        using Tensor px = VlmImagePreprocessor.PreprocessAspect(rgb, 1000, 100, Factor, Min, Max, [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f]);
        Assert.Equal(new TensorShape(1, 3, 96, 992), px.Shape);
        Assert.Equal(new MRopeImageGrid(1, 3, 31), Qwen3VlEncoder.MergedGrid(px, Factor / 2, 2));
    }

    [Fact]
    public void PreprocessAspect_ConstantImage_StaysConstantAfterNormalization()
    {
        // Bicubic weights sum to one, so a flat image keeps its value through the resample: (200/255 - 0.5) / 0.5 everywhere.
        const int w = 37, h = 23;
        byte[] rgb = new byte[w * h * 3];
        Array.Fill(rgb, (byte)200);
        using Tensor px = VlmImagePreprocessor.PreprocessAspect(rgb, w, h, Factor, Min, Max, [0.5f, 0.5f, 0.5f], [0.5f, 0.5f, 0.5f]);
        float want = (200f / 255f - 0.5f) / 0.5f;
        unsafe
        {
            float* p = (float*)px.DataPointer;
            for (long i = 0; i < px.ElementCount; i++)
                Assert.True(MathF.Abs(p[i] - want) < 1e-5f, $"element {i}: {p[i]} vs {want}");
        }
    }
}
