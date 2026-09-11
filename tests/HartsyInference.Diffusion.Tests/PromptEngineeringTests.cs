using System;
using System.Collections.Generic;
using Xunit;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Prompting;
using HartsyInference.Engine.Features;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Pure-logic tests for the prompt-engineering primitives: ComfyUI emphasis parsing, SwarmUI prompt-tag flattening and per-step <c>&lt;alternate:&gt;</c>/<c>&lt;fromto[N]:&gt;</c> scheduling, region-mask rasterization, and regional step gating (no GPU / checkpoint needed).</summary>
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
    public void Flattening_PlainText_Unchanged()
    {
        Assert.Equal("a photo of a cat", PromptTagFlattening.Flatten("a photo of a cat"));
    }

    [Fact]
    public void Flattening_Null_ReturnsEmpty()
    {
        Assert.Equal("", PromptTagFlattening.Flatten(null));
    }

    [Fact]
    public void Flattening_WeightTag_ConvertsToParens()
    {
        Assert.Equal("an (orange:1.5) cat", PromptTagFlattening.Flatten("an <weight[1.5]:orange> cat"));
    }

    [Fact]
    public void Flattening_WeightTag_ThenParsesToSameWeight()
    {
        IReadOnlyList<WeightedSpan> viaTag = PromptWeighting.Parse(PromptTagFlattening.Flatten("<weight[1.5]:orange>"));
        IReadOnlyList<WeightedSpan> viaParens = PromptWeighting.Parse("(orange:1.5)");
        Assert.Equal(viaParens[0].Text, viaTag[0].Text);
        Assert.Equal(viaParens[0].Weight, viaTag[0].Weight, 5);
    }

    [Fact]
    public void Flattening_WeightTag_InvalidWeight_LeftLiteral()
    {
        Assert.Equal("<weight[oops]:cat>", PromptTagFlattening.Flatten("<weight[oops]:cat>"));
    }

    [Fact]
    public void Flattening_NestedWeightTags_CompoundThroughPromptWeighting()
    {
        string flattened = PromptTagFlattening.Flatten("<weight[1.5]:<weight[1.2]:cat>>");
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(flattened);
        Assert.Single(spans);
        Assert.Equal("cat", spans[0].Text);
        Assert.Equal(1.5f * 1.2f, spans[0].Weight, 5);
    }

    [Fact]
    public void Flattening_WeightTag_EscapesLiteralParensInsideSpan()
    {
        // The (loud) the user typed is literal prose, not a nested weight group — it must be escaped so
        // PromptWeighting.Parse treats it as literal text at the outer weight, not an extra 1.1x on "cat".
        string flattened = PromptTagFlattening.Flatten("<weight[1.5]:a (loud) cat>");
        IReadOnlyList<WeightedSpan> spans = PromptWeighting.Parse(flattened);
        Assert.Single(spans);
        Assert.Equal("a (loud) cat", spans[0].Text);
        Assert.Equal(1.5f, spans[0].Weight, 5);
    }

    [Fact]
    public void Flattening_AlternateTag_FlattensToFirstEntry()
    {
        Assert.Equal("a cat", PromptTagFlattening.Flatten("a <alternate:cat, dog>"));
    }

    [Fact]
    public void Flattening_AltShorthand_FlattensToFirstEntry()
    {
        Assert.Equal("a cat", PromptTagFlattening.Flatten("a <alt:cat, dog>"));
    }

    [Fact]
    public void Flattening_AlternatePipeSeparated_FlattensToFirstEntry()
    {
        Assert.Equal("a cat", PromptTagFlattening.Flatten("a <alternate:cat | dog>"));
    }

    [Fact]
    public void Flattening_FromToTag_FlattensToFromValue()
    {
        Assert.Equal("a cat", PromptTagFlattening.Flatten("a <fromto[0.5]:cat, dog>"));
    }

    [Fact]
    public void Flattening_SchedulingDisabled_PreservesAlternateAndFromTo()
    {
        Assert.Equal("a <alternate:cat, dog>", PromptTagFlattening.Flatten("a <alternate:cat, dog>", flattenScheduling: false));
        Assert.Equal("a <fromto[0.5]:cat, dog>", PromptTagFlattening.Flatten("a <fromto[0.5]:cat, dog>", flattenScheduling: false));
    }

    [Fact]
    public void Flattening_SchedulingDisabled_StillConvertsWeightTags()
    {
        Assert.Equal("an (orange:1.5) cat", PromptTagFlattening.Flatten("an <weight[1.5]:orange> cat", flattenScheduling: false));
    }

    [Fact]
    public void Flattening_UnrelatedTags_PassThroughVerbatim()
    {
        Assert.Equal("<region:0,0,1,1> a cat", PromptTagFlattening.Flatten("<region:0,0,1,1> a cat"));
        Assert.Equal("a cat <break> a dog", PromptTagFlattening.Flatten("a cat <break> a dog"));
        Assert.Equal("<embed:myembed> a cat", PromptTagFlattening.Flatten("<embed:myembed> a cat"));
        Assert.Equal("<refcrop:0,face,0.5> a cat", PromptTagFlattening.Flatten("<refcrop:0,face,0.5> a cat"));
        Assert.Equal("<lora:myLora:0.8>", PromptTagFlattening.Flatten("<lora:myLora:0.8>"));
    }

    [Fact]
    public void Flattening_WeightTagInsideUnrelatedTag_StillConverted()
    {
        Assert.Equal(
            "<region:0,0,1,1> an (orange:1.5) cat",
            PromptTagFlattening.Flatten("<region:0,0,1,1> an <weight[1.5]:orange> cat"));
    }

    [Fact]
    public void Flattening_SwarmNestedWeightAroundAlternate_Resolves()
    {
        // SwarmUI's own documented nesting: "(layers of [a|b] features:1.5)" is converted by its
        // LegacyPromptParser to "<weight[1.5]:layers of <alternate:a,b> features>".
        string tag = "<weight[1.5]:layers of <alternate:a,b> features>";
        Assert.Equal("(layers of a features:1.5)", PromptTagFlattening.Flatten(tag));
        Assert.Equal("(layers of <alternate:a,b> features:1.5)", PromptTagFlattening.Flatten(tag, flattenScheduling: false));
    }

    [Fact]
    public void Flattening_WeightNestedInsideHeldBackScheduling_StillConverted()
    {
        // With flattenScheduling:false the alternate tag is deliberately preserved for PromptTagScheduling,
        // but a weight tag INSIDE it must still become parens — otherwise PromptTagScheduling picks that entry
        // and a raw <weight[...]> tag reaches the tokenizer as literal garbage.
        Assert.Equal(
            "<alternate:(a:1.5), b>",
            PromptTagFlattening.Flatten("<alternate:<weight[1.5]:a>, b>", flattenScheduling: false));
    }

    [Fact]
    public void Flattening_WeightInsideHeldBackScheduling_SurvivesStepResolution()
    {
        // End-to-end of the above through the real SDXL/SD1.5 order: flatten (scheduling held back) then
        // resolve per step — every step must yield parseable parens, never a raw tag.
        string flattened = PromptTagFlattening.Flatten("<alternate:<weight[1.5]:a>, b>", flattenScheduling: false);
        Assert.Equal("(a:1.5)", PromptTagScheduling.ResolveAt(flattened, 0, 10));
        Assert.Equal("b", PromptTagScheduling.ResolveAt(flattened, 1, 10));
        Assert.DoesNotContain("<weight", PromptTagScheduling.ResolveAt(flattened, 0, 10));
    }

    [Fact]
    public void Flattening_UnrecognizedTagWithBrackets_PassesThroughByteForByte()
    {
        // Quarry-style dataset tags carry square brackets that are NOT legacy prompt syntax. SwarmUI core
        // stopped stripping them (commit 46263630); we must not disturb them either, at any nesting depth.
        Assert.Equal(
            "<q:tags/deepghs.danbooru2024[rating!=g]>",
            PromptTagFlattening.Flatten("<q:tags/deepghs.danbooru2024[rating!=g]>"));
        Assert.Equal(
            "<q:tags/deepghs.danbooru2024[rating!=g]>",
            PromptTagFlattening.Flatten("<q:tags/deepghs.danbooru2024[rating!=g]>", flattenScheduling: false));
    }

    [Fact]
    public void Flattening_UnrecognizedTagPredata_Preserved()
    {
        // Recursing into an unrecognized tag's data must not lose its [predata] bracket.
        Assert.Equal("<param[cfgscale]:5>", PromptTagFlattening.Flatten("<param[cfgscale]:5>"));
        Assert.Equal("<param[cfgscale]:(5:1.2)>", PromptTagFlattening.Flatten("<param[cfgscale]:<weight[1.2]:5>>"));
    }

    [Fact]
    public void Scheduling_NoTags_NotDetected()
    {
        Assert.False(PromptTagScheduling.HasScheduling("a photo of a cat"));
    }

    [Fact]
    public void Scheduling_FromTo_DetectedAndResolves()
    {
        Assert.True(PromptTagScheduling.HasScheduling("a <fromto[0.5]:cat, dog>"));
        Assert.Equal("a cat", PromptTagScheduling.ResolveAt("a <fromto[0.5]:cat, dog>", 2, 10));
        Assert.Equal("a dog", PromptTagScheduling.ResolveAt("a <fromto[0.5]:cat, dog>", 7, 10));
    }

    [Fact]
    public void Scheduling_FromTo_AbsoluteStep_Threshold()
    {
        Assert.Equal("cat", PromptTagScheduling.ResolveAt("<fromto[3]:cat, dog>", 2, 10));
        Assert.Equal("dog", PromptTagScheduling.ResolveAt("<fromto[3]:cat, dog>", 3, 10));
    }

    [Fact]
    public void Scheduling_Alternate_CyclesPerStep()
    {
        Assert.Equal("cat", PromptTagScheduling.ResolveAt("<alternate:cat, dog>", 0, 10));
        Assert.Equal("dog", PromptTagScheduling.ResolveAt("<alternate:cat, dog>", 1, 10));
        Assert.Equal("cat", PromptTagScheduling.ResolveAt("<alternate:cat, dog>", 2, 10));
    }

    [Fact]
    public void Scheduling_AltShorthand_CyclesPerStep()
    {
        Assert.Equal("cat", PromptTagScheduling.ResolveAt("<alt:cat|dog>", 0, 10));
        Assert.Equal("dog", PromptTagScheduling.ResolveAt("<alt:cat|dog>", 1, 10));
    }

    [Fact]
    public void Scheduling_UnrelatedTags_PassThroughAtEveryStep()
    {
        Assert.Equal(
            "<region:0,0,1,1> (red:1.5) cat",
            PromptTagScheduling.ResolveAt("<region:0,0,1,1> (red:1.5) cat", 0, 10));
        Assert.Equal(
            "<region:0,0,1,1> (red:1.5) cat",
            PromptTagScheduling.ResolveAt("<region:0,0,1,1> (red:1.5) cat", 9, 10));
    }

    [Fact]
    public void Scheduling_FromTo_FractionNotRounded_MatchesReferenceBoundary()
    {
        // SwarmText.py: `if when < 1: when = when * steps` then `step < when` in floating point. 0.5 of 5 steps
        // is 2.5, so steps 0-2 take "from". Rounding the threshold to an int first loses step 2.
        Assert.Equal("cat", PromptTagScheduling.ResolveAt("<fromto[0.5]:cat, dog>", 2, 5));
        Assert.Equal("dog", PromptTagScheduling.ResolveAt("<fromto[0.5]:cat, dog>", 3, 5));
    }

    [Fact]
    public void Scheduling_FromTo_AboveOneIsAbsoluteStep_NotAFraction()
    {
        // `when >= 1` is an absolute step index even when it has a decimal point — 1.5 means "switch between
        // step 1 and step 2", NOT "1.5x the step count".
        Assert.Equal("cat", PromptTagScheduling.ResolveAt("<fromto[1.5]:cat, dog>", 1, 20));
        Assert.Equal("dog", PromptTagScheduling.ResolveAt("<fromto[1.5]:cat, dog>", 2, 20));
    }

    [Fact]
    public void Scheduling_FromTo_NonNumericWhen_IsNotScheduling()
    {
        // Reference returns the tag as literal text when the predata will not parse as a float.
        Assert.False(PromptTagScheduling.HasScheduling("<fromto[oops]:cat, dog>"));
        Assert.Equal("<fromto[oops]:cat, dog>", PromptTagScheduling.ResolveAt("<fromto[oops]:cat, dog>", 0, 10));
        Assert.Equal("<fromto[oops]:cat, dog>", PromptTagFlattening.Flatten("<fromto[oops]:cat, dog>"));
    }

    [Fact]
    public void Scheduling_FromTo_WrongEntryCount_IsLeftLiteral()
    {
        Assert.Equal("<fromto[0.5]:a, b, c>", PromptTagScheduling.ResolveAt("<fromto[0.5]:a, b, c>", 0, 10));
    }

    [Fact]
    public void Scheduling_HasScheduling_DetectsAltAndNestedTags()
    {
        Assert.True(PromptTagScheduling.HasScheduling("a <alt:cat, dog>"));
        Assert.True(PromptTagScheduling.HasScheduling("a <alternate:cat, dog>"));
        Assert.False(PromptTagScheduling.HasScheduling("a (red:1.5) cat <break> <region:0,0,1,1>"));
    }

    [Fact]
    public void WeightingSyntax_BracketsAlone_DoNotForceTheWeightedPath()
    {
        // Brackets carry no grammar any more, so bracket-bearing prose must keep the plain-encode path — on
        // SD1.5 a conditioning schedule forfeits the fused Euler loop and makes a non-default sampler throw.
        Assert.False(WeightedConditioning.HasWeightingSyntax("a [vintage] dress"));
        Assert.False(WeightedConditioning.HasWeightingSyntax("<q:tags/deepghs.danbooru2024[rating!=g]>"));
        Assert.True(WeightedConditioning.HasWeightingSyntax("a (vintage:1.2) dress"));
        Assert.True(WeightedConditioning.HasWeightingSyntax("a dress <break> a hat"));
    }

    [Fact]
    public void Flattening_IsIdempotentOnAlreadyFlattenedText()
    {
        // SdxlRecipePipeline/Sd15RecipePipeline re-flatten what ImagesService already ran with
        // flattenScheduling:false, so a second pass must not disturb the parens it produced.
        string once = PromptTagFlattening.Flatten("an <weight[1.5]:orange> <alternate:cat, dog>", flattenScheduling: false);
        string twice = PromptTagFlattening.Flatten(once);
        Assert.Equal("an (orange:1.5) cat", twice);
        Assert.Equal(twice, PromptTagFlattening.Flatten(twice));
    }

    [Fact]
    public void Scheduling_Resolve_DedupesVariants()
    {
        PromptSchedule schedule = PromptTagScheduling.Resolve("a <fromto[0.5]:cat, dog>", 10);
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
        PromptSchedule schedule = PromptTagScheduling.Resolve("a <fromto[0.5]:cat, dog>", 10);
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

    [Fact]
    public void TextualInversion_DualTensor_SdxlEmbed_LoadsBothDims()
    {
        // SDXL dual embed (clip_l 768 + clip_g 1280). Exercises the SUCCESS path of TextualInversion.Load — which
        // copies out of a loader-mmap'd tensor. Regression for the use-after-free where SelectEmbedding ran after
        // the loader's `using` had already unmapped the file (AccessViolation on the copy).
        string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"sdxlti_{Guid.NewGuid():N}.safetensors");
        try
        {
            using (Tensor l = new Tensor(new TensorShape(4, 768), DType.F32))
            using (Tensor g = new Tensor(new TensorShape(4, 1280), DType.F32))
            {
                l.AsSpan<float>().Fill(0.11f);
                g.AsSpan<float>().Fill(0.22f);
                SafeTensorsWriter.Save(path, new Dictionary<string, Tensor> { ["clip_l"] = l, ["clip_g"] = g });
            }

            using Tensor eL = TextualInversion.Load(path, 768);
            using Tensor eG = TextualInversion.Load(path, 1280);
            Assert.Equal(4, (int)eL.Shape[0]);
            Assert.Equal(768, (int)eL.Shape[1]);
            Assert.Equal(4, (int)eG.Shape[0]);
            Assert.Equal(1280, (int)eG.Shape[1]);
            // Read every element — a dangling mmap pointer would AccessViolation here.
            Assert.Equal(0.11f, eL.AsReadOnlySpan<float>()[eL.AsReadOnlySpan<float>().Length - 1], 4);
            Assert.Equal(0.22f, eG.AsReadOnlySpan<float>()[eG.AsReadOnlySpan<float>().Length - 1], 4);
        }
        finally
        {
            if (System.IO.File.Exists(path)) System.IO.File.Delete(path);
        }
    }
}
