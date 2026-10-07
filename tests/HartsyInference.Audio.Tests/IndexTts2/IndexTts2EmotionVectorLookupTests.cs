using HartsyInference.Audio.Models.IndexTts2;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using Xunit;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>Tiny random <c>feat1</c>/<c>feat2</c>-shaped exemplar banks through
/// <see cref="IndexTts2EmotionVectorLookup"/>: category sizes, cosine-nearest selection, normalization, and
/// determinism only. Says nothing about parity with the real bundled <c>feat1.pt</c>/<c>feat2.pt</c>.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class IndexTts2EmotionVectorLookupTests
{
    private const int StyleDim = 6;
    private const int Hidden = 10;

    private static (Tensor spk, Tensor emo) BuildBanks(Random rng)
    {
        int total = IndexTts2EmotionVectorLookup.EmoNum.Sum();
        Tensor spk = new(new TensorShape(total, StyleDim), DType.F32);
        foreach (ref float v in spk.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);
        Tensor emo = new(new TensorShape(total, Hidden), DType.F32);
        foreach (ref float v in emo.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);
        return (spk, emo);
    }

    [Fact]
    public void NormalizeEmoVec_AppliesBiasAndCapsSumAt0_8()
    {
        float[] raw = [1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f];
        float[] normalized = IndexTts2EmotionVectorLookup.NormalizeEmoVec(raw, applyBias: true);
        Assert.Equal(8, normalized.Length);
        float sum = normalized.Sum();
        Assert.True(sum <= 0.8f + 1e-5f, $"sum {sum} exceeds the 0.8 cap.");

        // Relative weighting between categories must still reflect the fixed bias array after rescaling.
        float[] expectedBias = IndexTts2EmotionVectorLookup.Bias;
        float ratio = normalized[0] / expectedBias[0];
        for (int i = 1; i < normalized.Length; i++)
            Assert.Equal(ratio, normalized[i] / expectedBias[i], 4);
    }

    [Fact]
    public void NormalizeEmoVec_LeavesSmallVectorsUnscaled()
    {
        float[] raw = [0.1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f];
        float[] normalized = IndexTts2EmotionVectorLookup.NormalizeEmoVec(raw, applyBias: false);
        Assert.Equal(0.1f, normalized[0], 5);
    }

    [Fact]
    public void NormalizeEmoVec_RejectsWrongLength()
    {
        Assert.Throws<ArgumentException>(() => IndexTts2EmotionVectorLookup.NormalizeEmoVec([1f, 2f, 3f]));
    }

    [Fact]
    public void ComputeEmoVecMat_SelectsTheExactMatchingExemplar_WhenStyleEqualsOneRow()
    {
        Random rng = new(1);
        (Tensor spk, Tensor emo) = BuildBanks(rng);
        using IndexTts2EmotionVectorLookup lookup = new(spk, emo);

        // Point the style vector exactly at category 0's (EmoNum[0] == 3) row index 1, so the cosine-nearest
        // search must pick that row deterministically regardless of the other rows' random values.
        int rowIndex = 1;
        float[] style = new float[StyleDim];
        unsafe
        {
            float* row = (float*)spk.DataPointer + (long)rowIndex * StyleDim;
            for (int c = 0; c < StyleDim; c++) style[c] = row[c];
        }
        using Tensor styleTensor = new(new TensorShape(1, StyleDim), DType.F32);
        style.CopyTo(styleTensor.AsSpan<float>());

        float[] emoVector = [1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f]; // weight only category 0
        uint rngState = 42;
        using Tensor result = lookup.ComputeEmoVecMat(emoVector, styleTensor, useRandom: false, ref rngState);

        float[] expectedRow = new float[Hidden];
        unsafe
        {
            float* row = (float*)emo.DataPointer + (long)rowIndex * Hidden;
            for (int c = 0; c < Hidden; c++) expectedRow[c] = row[c];
        }
        // Weights are applied exactly as given (the reference's library path never normalizes them).
        Span<float> got = result.AsSpan<float>();
        for (int c = 0; c < Hidden; c++)
            Assert.Equal(emoVector[0] * expectedRow[c], got[c], 4);
    }

    [Fact]
    public void ComputeEmoVecMat_IsDeterministic_ForTheSameSeed_WithRandomSelection()
    {
        Random rng = new(2);
        (Tensor spk, Tensor emo) = BuildBanks(rng);
        using IndexTts2EmotionVectorLookup lookup = new(spk, emo);
        using Tensor style = new(new TensorShape(1, StyleDim), DType.F32);
        foreach (ref float v in style.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

        float[] emoVector = [0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f];
        uint seedA = 7, seedB = 7;
        using Tensor a = lookup.ComputeEmoVecMat(emoVector, style, useRandom: true, ref seedA);
        using Tensor b = lookup.ComputeEmoVecMat(emoVector, style, useRandom: true, ref seedB);
        Assert.Equal(a.AsSpan<float>().ToArray(), b.AsSpan<float>().ToArray());
    }

    [Fact]
    public void Constructor_RejectsWrongRowCount()
    {
        Tensor spk = new(new TensorShape(5, StyleDim), DType.F32);
        Tensor emo = new(new TensorShape(5, Hidden), DType.F32);
        Assert.Throws<ArgumentException>(() => new IndexTts2EmotionVectorLookup(spk, emo));
        spk.Dispose();
        emo.Dispose();
    }
}

/// <summary>Loads the real bundled <c>feat1.pt</c>/<c>feat2.pt</c> (shipped in the main IndexTTS-2.5 repo; point
/// <c>INDEXTTS2_FEAT1_PT_PATH</c>/<c>INDEXTTS2_FEAT2_PT_PATH</c> at local copies). Confirms the real checkpoint's
/// row count matches <see cref="IndexTts2EmotionVectorLookup.EmoNum"/>'s sum and that a lookup against them
/// produces a finite <c>[1, 1280]</c> result.</summary>
[Trait("Category", "Integration")]
public sealed class IndexTts2EmotionVectorLookupRealWeightTests
{
    [Fact]
    public void ComputeEmoVecMat_SucceedsAgainstRealFeatFiles()
    {
        string? feat1Path = Environment.GetEnvironmentVariable("INDEXTTS2_FEAT1_PT_PATH");
        string? feat2Path = Environment.GetEnvironmentVariable("INDEXTTS2_FEAT2_PT_PATH");
        if (string.IsNullOrEmpty(feat1Path) || !File.Exists(feat1Path)) return;
        if (string.IsNullOrEmpty(feat2Path) || !File.Exists(feat2Path)) return;

        using PytorchPickleLoader feat1Loader = new();
        feat1Loader.Load(feat1Path, recursiveFlatten: true);
        Tensor feat1 = feat1Loader.GetAllTensors()["data"];

        using PytorchPickleLoader feat2Loader = new();
        feat2Loader.Load(feat2Path, recursiveFlatten: true);
        Tensor feat2 = feat2Loader.GetAllTensors()["data"];

        using IndexTts2EmotionVectorLookup lookup = new(feat1, feat2);

        using Tensor style = new(new TensorShape(1, (int)feat1.Shape[1]), DType.F32);
        Random rng = new(3);
        foreach (ref float v in style.AsSpan<float>()) v = (float)(rng.NextDouble() * 2 - 1);

        float[] emoVector = [0.8f, 0f, 0f, 0f, 0f, 0f, 0f, 0f]; // pure "happy"
        uint rngState = 11;
        using Tensor result = lookup.ComputeEmoVecMat(emoVector, style, useRandom: false, ref rngState);

        Assert.Equal(1280, (int)result.Shape[1]);
        foreach (float v in result.AsSpan<float>()) Assert.True(float.IsFinite(v));
    }
}
