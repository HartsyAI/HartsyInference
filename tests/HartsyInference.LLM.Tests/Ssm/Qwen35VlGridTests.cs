using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Multimodal;
using Xunit;

namespace HartsyInference.LLM.Tests.Ssm;

/// <summary>The image grid behind Qwen3.5-VL's M-RoPE: read from the pixels the tower encodes (never from the token count, where a 2×8 grid and a
/// 4×4 grid both hold 16 tokens), and used for the placeholder run's positions.</summary>
public sealed class Qwen35VlGridTests
{
    private const int Patch = 16, Merge = 2;

    [Fact]
    public void MergedGrid_NonSquarePixels_ReadsTheRealGrid()
    {
        // 64×256 px → 4×16 patches → 2×8 merged tokens (16 placeholders), not the 4×4 a square guess would give.
        using Tensor px = new(new TensorShape(1, 3, 64, 256), DType.F32);
        MRopeImageGrid grid = Qwen3VlEncoder.MergedGrid(px, Patch, Merge);
        Assert.Equal(new MRopeImageGrid(1, 2, 8), grid);
        Assert.Equal(16, grid.Tokens);
    }

    [Fact]
    public void MergedGrid_SquarePixels_KeepsTheSquareGrid()
    {
        using Tensor px = new(new TensorShape(1, 3, 64, 64), DType.F32);
        Assert.Equal(new MRopeImageGrid(1, 2, 2), Qwen3VlEncoder.MergedGrid(px, Patch, Merge));
    }

    [Fact]
    public void MergedGrid_RefusesASizeThatIsNotWholeMergedPatches()
    {
        using Tensor px = new(new TensorShape(1, 3, 60, 64), DType.F32);
        Assert.Throws<ArgumentException>(() => Qwen3VlEncoder.MergedGrid(px, Patch, Merge));
    }

    [Fact]
    public void BuildPositions_NonSquareGrid_PlacesRowsAndColumnsOnTheRealGrid()
    {
        // [pre 3][image 2×8][post 2]: the image starts at 3, so row r / column c sit at 3 + r and 3 + c; the text after it resumes at 3 + max(1, 2, 8).
        const int pre = 3, post = 2, seqLen = pre + 16 + post;
        int[] positions = Qwen35VlGenerator.BuildPositions(seqLen, pre, 16, new MRopeImageGrid(1, 2, 8), out int next);

        int H(int i) => positions[seqLen + i];
        int W(int i) => positions[2 * seqLen + i];
        int T(int i) => positions[i];

        Assert.Equal(0, T(0)); Assert.Equal(1, H(1)); Assert.Equal(2, W(2));   // leading text
        Assert.Equal(3, T(3)); Assert.Equal(3, H(3)); Assert.Equal(3, W(3));   // image (0,0)
        Assert.Equal(3, H(3 + 7)); Assert.Equal(10, W(3 + 7));                 // row 0, column 7 (a 4×4 guess puts this at row 1, column 3)
        Assert.Equal(4, H(3 + 8 + 1)); Assert.Equal(4, W(3 + 8 + 1));          // row 1, column 1
        Assert.Equal(4, H(3 + 15)); Assert.Equal(10, W(3 + 15));               // row 1, column 7
        Assert.Equal(11, T(19)); Assert.Equal(11, H(19)); Assert.Equal(11, W(19));   // text after the image
        Assert.Equal(12, T(20));
        Assert.Equal(13, next);
    }

    [Fact]
    public void BuildPositions_SquareGrid_KeepsTheSquareLayout()
    {
        // [pre 2][image 4×4][post 1]: image (1,1) sits at 2 + 1 on both axes; text after resumes at 2 + 4.
        const int pre = 2, post = 1, seqLen = pre + 16 + post;
        int[] positions = Qwen35VlGenerator.BuildPositions(seqLen, pre, 16, new MRopeImageGrid(1, 4, 4), out int next);
        Assert.Equal(3, positions[seqLen + pre + 5]);
        Assert.Equal(3, positions[2 * seqLen + pre + 5]);
        Assert.Equal(6, positions[seqLen + seqLen - 1]);
        Assert.Equal(7, next);
    }

    [Fact]
    public void BuildPositions_RefusesAGridThatDoesNotHoldTheEncoderTokens()
    {
        Assert.Throws<ArgumentException>(() => Qwen35VlGenerator.BuildPositions(21, 3, 4, new MRopeImageGrid(1, 2, 8), out _));
    }
}
