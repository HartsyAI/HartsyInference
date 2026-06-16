using System;
using System.Collections.Generic;
using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic tests for the prompt-engineering primitives: ComfyUI emphasis parsing, bracket scheduling/alternation, region-mask rasterization, and regional step gating (no GPU / checkpoint needed).</summary>
public class PromptEngineeringTests
{
    [Fact]
    public void Weighting_PlainText_SingleSpanWeightOne()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("a photo of a cat");
        Assert.Single(spans);
        Assert.Equal("a photo of a cat", spans[0].Text);
        Assert.Equal(1.0f, spans[0].Weight, 5);
    }

    [Fact]
    public void Weighting_BareParens_MultipliesByOnePointOne()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("a (cat)");
        Assert.Equal(2, spans.Count);
        Assert.Equal("a ", spans[0].Text);
        Assert.Equal(1.0f, spans[0].Weight, 5);
        Assert.Equal("cat", spans[1].Text);
        Assert.Equal(1.1f, spans[1].Weight, 5);
    }

    [Fact]
    public void Weighting_ExplicitWeight_SetsWeight()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("(cat:1.3)");
        Assert.Single(spans);
        Assert.Equal("cat", spans[0].Text);
        Assert.Equal(1.3f, spans[0].Weight, 5);
    }

    [Fact]
    public void Weighting_Nested_CompoundsWeights()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("((cat))");
        Assert.Single(spans);
        Assert.Equal("cat", spans[0].Text);
        Assert.Equal(1.1f * 1.1f, spans[0].Weight, 5);
    }

    [Fact]
    public void Weighting_NestedWithExplicit_Compounds()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("(a (cat:1.5))");
        Assert.Equal(2, spans.Count);
        Assert.Equal("a ", spans[0].Text);
        Assert.Equal(1.1f, spans[0].Weight, 5);
        Assert.Equal("cat", spans[1].Text);
        Assert.Equal(1.1f * 1.5f, spans[1].Weight, 5);
    }

    [Fact]
    public void Weighting_EscapedParens_AreLiteral()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("a \\(cat\\)");
        Assert.Single(spans);
        Assert.Equal("a (cat)", spans[0].Text);
        Assert.Equal(1.0f, spans[0].Weight, 5);
    }

    [Fact]
    public void Weighting_SquareBrackets_LeftUntouched()
    {
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse("a [cat:dog:0.5]");
        Assert.Single(spans);
        Assert.Equal("a [cat:dog:0.5]", spans[0].Text);
    }

    [Fact]
    public void Weighting_Chunks_SplitOnBreak()
    {
        IReadOnlyList<IReadOnlyList<WeightedSpan>> chunks = PromptWeighting.ParseChunks("a cat <break> a (dog:1.2)");
        Assert.Equal(2, chunks.Count);
        Assert.Equal("a cat ", chunks[0][0].Text);
        Assert.Equal("dog", chunks[1][^1].Text);
        Assert.Equal(1.2f, chunks[1][^1].Weight, 5);
    }

    [Fact]
    public void Scheduling_NoBrackets_NotDetected()
    {
        Assert.False(PromptScheduling.HasScheduling("a photo of a cat"));
    }

    [Fact]
    public void Scheduling_Switch_DetectedAndResolves()
    {
        Assert.True(PromptScheduling.HasScheduling("a [cat:dog:0.5]"));
        Assert.Equal("a cat", PromptScheduling.ResolveAt("a [cat:dog:0.5]", 2, 10));
        Assert.Equal("a dog", PromptScheduling.ResolveAt("a [cat:dog:0.5]", 7, 10));
    }

    [Fact]
    public void Scheduling_TwoPart_EmptyFrom()
    {
        Assert.Equal("a ", PromptScheduling.ResolveAt("a [cat:0.5]", 2, 10));
        Assert.Equal("a cat", PromptScheduling.ResolveAt("a [cat:0.5]", 7, 10));
    }

    [Fact]
    public void Scheduling_AbsoluteStep_Threshold()
    {
        Assert.Equal("cat", PromptScheduling.ResolveAt("[cat:dog:3]", 2, 10));
        Assert.Equal("dog", PromptScheduling.ResolveAt("[cat:dog:3]", 3, 10));
    }

    [Fact]
    public void Scheduling_Alternation_CyclesPerStep()
    {
        Assert.Equal("cat", PromptScheduling.ResolveAt("[cat|dog]", 0, 10));
        Assert.Equal("dog", PromptScheduling.ResolveAt("[cat|dog]", 1, 10));
        Assert.Equal("cat", PromptScheduling.ResolveAt("[cat|dog]", 2, 10));
    }

    [Fact]
    public void Scheduling_Resolve_DedupesVariants()
    {
        PromptSchedule schedule = PromptScheduling.Resolve("a [cat:dog:0.5]", 10);
        Assert.Equal(2, schedule.Variants.Count);
        Assert.Equal(0, schedule.StepToVariant[0]);
        Assert.Equal(1, schedule.StepToVariant[9]);
    }

    [Fact]
    public void RegionMask_RectToLatent_CoversExpectedCells()
    {
        // 16x16 reference rect covering the left half; downsample to 4x4 latent (exact 4x multiple).
        RegionMask mask = RegionMask.FromRect(new RectMask(0, 0, 8, 16), 16, 16);
        using Tensor latent = mask.ToLatentMask(4, 4);
        Assert.Equal(new TensorShape(1, 1, 4, 4), latent.Shape);
        ReadOnlySpan<float> v = latent.AsReadOnlySpan<float>();
        // Left two columns fully covered (1.0), right two empty (0.0).
        for (int y = 0; y < 4; y++)
        {
            Assert.Equal(1f, v[y * 4 + 0], 4);
            Assert.Equal(1f, v[y * 4 + 1], 4);
            Assert.Equal(0f, v[y * 4 + 2], 4);
            Assert.Equal(0f, v[y * 4 + 3], 4);
        }
    }

    [Fact]
    public void RegionMask_NonAlignedResample_ProducesPartialCoverage()
    {
        // 10x10 rect over left 5 columns, resampled to 3x3 (non-multiple) -> middle column partial.
        RegionMask mask = RegionMask.FromRect(new RectMask(0, 0, 5, 10), 10, 10);
        using Tensor latent = mask.ToLatentMask(3, 3);
        ReadOnlySpan<float> v = latent.AsReadOnlySpan<float>();
        Assert.Equal(1f, v[0], 3);
        Assert.True(v[1] > 0f && v[1] < 1f);
        Assert.Equal(0f, v[2], 3);
    }

    [Fact]
    public void RegionMask_Bitmap_LengthValidated()
    {
        Assert.Throws<HartsyInferenceException>(() => RegionMask.FromBitmap(new float[3], 2, 2));
    }

    [Fact]
    public void RegionalPlan_ResolveStep_GatesByWindow()
    {
        using Tensor baseCond = new Tensor(new TensorShape(1, 1, 4), DType.F32);
        using Tensor regionCond = new Tensor(new TensorShape(1, 1, 4), DType.F32);
        RegionMask mask = RegionMask.FromRect(new RectMask(0, 0, 4, 4), 8, 8);
        RegionalPlan plan = new RegionalPlan
        {
            BaseCond = baseCond,
            Regions = [new RegionConditioning(regionCond, mask, 0.8f, 2, 6)],
        };
        Span<float> weights = stackalloc float[1];
        plan.ResolveStep(0, weights);
        Assert.Equal(0f, weights[0], 5);
        plan.ResolveStep(3, weights);
        Assert.Equal(0.8f, weights[0], 5);
        plan.ResolveStep(6, weights);
        Assert.Equal(0f, weights[0], 5);
    }

    [Fact]
    public void RegionalPlan_ResolveStep_WrongLengthThrows()
    {
        using Tensor baseCond = new Tensor(new TensorShape(1, 1, 4), DType.F32);
        RegionalPlan plan = new RegionalPlan { BaseCond = baseCond };
        Assert.Throws<HartsyInferenceException>(() =>
        {
            float[] weights = new float[2];
            plan.ResolveStep(0, weights);
        });
    }

    [Fact]
    public void ConditioningSchedule_FromPromptSchedule_SelectsVariant()
    {
        PromptSchedule schedule = PromptScheduling.Resolve("a [cat:dog:0.5]", 10);
        using Tensor variant0 = new Tensor(new TensorShape(1, 1, 4), DType.F32);
        using Tensor variant1 = new Tensor(new TensorShape(1, 1, 4), DType.F32);
        ConditioningSchedule cond = ConditioningSchedule.FromPromptSchedule(schedule, [variant0, variant1]);
        Assert.Equal(0, cond.Resolve(0, 10));
        Assert.Equal(1, cond.Resolve(9, 10));
    }

    [Fact]
    public void Emphasis_ApplyComfy_InterpolatesFromEmptyBaseline()
    {
        float[] hidden = [10f, 10f, 20f, 20f];
        float[] empty = [1f, 1f, 1f, 1f];
        float[] weights = [1f, 1.5f];
        EmphasisMath.ApplyComfy(hidden, empty, weights, 2, 2);
        // Token 0 weight 1.0 -> unchanged; token 1 -> (20-1)*1.5 + 1 = 29.5.
        Assert.Equal(10f, hidden[0], 4);
        Assert.Equal(10f, hidden[1], 4);
        Assert.Equal(29.5f, hidden[2], 4);
        Assert.Equal(29.5f, hidden[3], 4);
    }

    [Fact]
    public void WeightedTokenizer_Break_ProducesTwoWrappedChunks()
    {
        using ClipTokenizer tokenizer = new ClipTokenizer();
        (IReadOnlyList<int[]> ids, IReadOnlyList<float[]> weights) = WeightedPromptTokenizer.Tokenize(tokenizer, "a cat <break> a dog");
        Assert.Equal(2, ids.Count);
        foreach (int[] chunk in ids)
        {
            Assert.Equal(ClipTokenizer.MaxLength, chunk.Length);
            Assert.Equal(ClipTokenizer.StartOfTextId, chunk[0]);
            Assert.Equal(ClipTokenizer.EndOfTextId, chunk[^1]);
        }
        Assert.Equal(1f, weights[0][0], 4);
    }

    [Fact]
    public void WeightedTokenizer_Weight_AppliedToContentTokens()
    {
        using ClipTokenizer tokenizer = new ClipTokenizer();
        (IReadOnlyList<int[]> _, IReadOnlyList<float[]> weights) = WeightedPromptTokenizer.Tokenize(tokenizer, "(cat:1.5)");
        Assert.Single(weights);
        Assert.Equal(1f, weights[0][0], 4);
        Assert.Equal(1.5f, weights[0][1], 4);
        Assert.Equal(1f, weights[0][^1], 4);
    }

    [Fact]
    public void WeightedTokenizer_Overflow_SplitsIntoMultipleChunks()
    {
        using ClipTokenizer tokenizer = new ClipTokenizer();
        string longPrompt = string.Join(" ", System.Linq.Enumerable.Repeat("word", 100));
        (IReadOnlyList<int[]> ids, IReadOnlyList<float[]> _) = WeightedPromptTokenizer.Tokenize(tokenizer, longPrompt);
        Assert.True(ids.Count >= 2);
        Assert.All(ids, chunk => Assert.Equal(ClipTokenizer.MaxLength, chunk.Length));
    }

    [Fact]
    public void WeightedTokenizer_Empty_ProducesOneBareChunk()
    {
        using ClipTokenizer tokenizer = new ClipTokenizer();
        (IReadOnlyList<int[]> ids, IReadOnlyList<float[]> _) = WeightedPromptTokenizer.Tokenize(tokenizer, "");
        Assert.Single(ids);
        Assert.Equal(ClipTokenizer.StartOfTextId, ids[0][0]);
        Assert.Equal(ClipTokenizer.EndOfTextId, ids[0][1]);
    }

    [Fact]
    public void RegionalBias_InsideRegion_ZeroOutsideSuppressed()
    {
        // seq [base0 | regionCols 1,2 | image rows 3,4], one region, mask covers image token 0 only.
        int seqLen = 5;
        using Tensor bias = RegionalAttentionBias.Build(
            seqLen,
            imageStart: 3,
            numImg: 2,
            regionColumnRanges: [(1, 3)],
            regionGridMasks: [[1f, 0f]],
            regionWeights: stackalloc float[] { 1f });
        ReadOnlySpan<float> b = bias.AsReadOnlySpan<float>();
        // Image token 0 (row 3) is inside the region (mask 1, weight 1 -> bias log(1)=0).
        Assert.Equal(0f, b[3 * seqLen + 1], 4);
        // Image token 1 (row 4) is outside (mask 0 -> floored log(MinWeight)).
        Assert.Equal(MathF.Log(RegionalAttentionBias.MinWeight), b[4 * seqLen + 1], 3);
        // Base-text column and image self columns untouched.
        Assert.Equal(0f, b[3 * seqLen + 0], 4);
        Assert.Equal(0f, b[3 * seqLen + 3], 4);
    }

    [Fact]
    public void TextualInversion_Load_NoMatchingDimThrows()
    {
        string path = Path.Combine(Path.GetTempPath(), $"ti_{Guid.NewGuid():N}.safetensors");
        try
        {
            using Tensor emb = new Tensor(new TensorShape(2, 4), DType.F32);
            SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["emb_params"] = emb });
            Assert.Throws<HartsyInferenceException>(() => TextualInversion.Load(path, 8));
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    [Fact]
    public void RegionalBias_HighWeight_BoostsRegionColumns()
    {
        int seqLen = 4;
        using Tensor bias = RegionalAttentionBias.Build(
            seqLen,
            imageStart: 2,
            numImg: 2,
            regionColumnRanges: [(0, 1)],
            regionGridMasks: [[1f, 1f]],
            regionWeights: stackalloc float[] { 2f });
        ReadOnlySpan<float> b = bias.AsReadOnlySpan<float>();
        Assert.Equal(MathF.Log(2f), b[2 * seqLen + 0], 4);
        Assert.Equal(MathF.Log(2f), b[3 * seqLen + 0], 4);
    }
}
