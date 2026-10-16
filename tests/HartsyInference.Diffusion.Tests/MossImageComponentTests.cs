using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MOSS-Image's pieces outside the transformer forward: the training-grid schedule (SGLang's golden values), the
/// guidance recipe (branch topology and the SLG combine, checked against SGLang's
/// <c>test_guidance_formula_window_branch_count_and_request_isolation</c>), the RoPE coordinates and the converter.</summary>
public sealed class MossImageComponentTests
{
    /// <summary>SGLang's <c>test_v2_schedule_and_shift_override</c>: golden nearest-training-grid values from the standalone
    /// reference at 5 steps and shift 5.</summary>
    [Fact]
    public void Schedule_MatchesTheReferenceGoldens()
    {
        float[] sigmas = TrainingGridSchedule.Sigmas(5, 5f);
        Assert.Equal([1.0f, 0.938f, 0.835f, 0.630f, 0.024f], sigmas);
        FlowMatchEulerDiscreteScheduler scheduler = new(shift: 1f);
        scheduler.SetSigmas(sigmas);
        Assert.Equal([1000f, 938f, 835f, 630f, 24f], scheduler.Timesteps.ToArray());
    }

    [Theory]
    [InlineData(50, 5f)]
    [InlineData(30, 3f)]
    [InlineData(28, 1f)]
    public void Schedule_SnapsEveryStepToTheTrainingGrid(int steps, float shift)
    {
        float[] sigmas = TrainingGridSchedule.Sigmas(steps, shift);
        Assert.Equal(1f, sigmas[0]);
        Assert.Equal(steps, sigmas.Length);
        for (int i = 0; i < sigmas.Length; i++)
        {
            float onGrid = MathF.Round(sigmas[i] * 1000f) / 1000f;
            Assert.True(MathF.Abs(sigmas[i] - onGrid) < 1e-6f, $"sigma {i} = {sigmas[i]} is off the 0.001 grid");
            if (i > 0) Assert.True(sigmas[i] < sigmas[i - 1], $"sigma {i} does not descend");
        }
    }

    [Fact]
    public void Schedule_DropsStepsThatSnapToTheSamePoint()
    {
        float[] sigmas = TrainingGridSchedule.Sigmas(1500, 1f);
        Assert.True(sigmas.Length <= 1000);
        Assert.Equal(sigmas.Distinct().Count(), sigmas.Length);
    }

    [Fact]
    public void ReferenceGuidance_SkipsLayer22OnTheUnconditionalBranchOnly()
    {
        using StubConditionings c = new();
        List<MossImageBranch> branches = MossImageGuidance.Reference.BranchesFor(c.Cond, c.Uncond, guided: true, stepIndex: 0, steps: 50);
        Assert.Equal(2, branches.Count);
        Assert.Same(c.Cond, branches[0].Conditioning);
        Assert.Empty(branches[0].SkipLayers);
        Assert.Same(c.Uncond, branches[1].Conditioning);
        Assert.Equal([22], branches[1].SkipLayers);
    }

    /// <summary>SGLang's guidance test topology: a window of [0.25, 0.75) over 4 steps puts steps 1 and 2 inside; the
    /// unconditional branch skips only there and the SLG branch exists only there.</summary>
    [Theory]
    [InlineData(0, false)]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(3, false)]
    public void Window_GatesBothTheUncondSkipAndTheSlgBranch(int stepIndex, bool inside)
    {
        using StubConditionings c = new();
        MossImageGuidance guidance = new() { UncondSkipLayers = [22], SlgLayers = [22], SlgScale = 3f, Start = 0.25f, Stop = 0.75f };
        List<MossImageBranch> branches = guidance.BranchesFor(c.Cond, c.Uncond, guided: true, stepIndex, steps: 4);
        Assert.Equal(inside ? 3 : 2, branches.Count);
        Assert.Equal(inside ? [22] : [], branches[1].SkipLayers);
        if (inside)
        {
            Assert.Same(c.Cond, branches[2].Conditioning);
            Assert.Equal([22], branches[2].SkipLayers);
        }
    }

    [Fact]
    public void Unguided_RunsTheConditionalBranchAlone()
    {
        using StubConditionings c = new();
        Assert.Single(MossImageGuidance.Reference.BranchesFor(c.Cond, null, guided: false, stepIndex: 0, steps: 50));
    }

    /// <summary>The SLG combine against SGLang's formula: <c>uncond + g·(cond − uncond) + s·(cond − skipped)</c> with CFG,
    /// <c>skipped + s·(cond − skipped)</c> without.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SlgCombine_MatchesTheReferenceFormula(bool guided)
    {
        using CpuBackend backend = new();
        float[] condV = [1f, -2f, 0.5f], uncondV = [0.25f, 1f, -1f], skippedV = [-0.5f, 3f, 2f];
        using Tensor cond = Make(condV), uncond = Make(uncondV), skipped = Make(skippedV);
        MossImageGuidance guidance = new() { SlgLayers = [7], SlgScale = 3f };
        const float cfg = 4f;
        using Tensor combined = guidance.CombineSlg(backend, cond, guided ? uncond : null, skipped, cfg);
        ReadOnlySpan<float> got = combined.AsReadOnlySpan<float>();
        for (int i = 0; i < 3; i++)
        {
            float baseV = guided ? uncondV[i] + cfg * (condV[i] - uncondV[i]) : skippedV[i];
            Assert.Equal(baseV + 3f * (condV[i] - skippedV[i]), got[i], 5);
        }
    }

    [Fact]
    public void Guidance_RefusesALayerTheModelDoesNotHave()
    {
        MossImageGuidance guidance = new() { UncondSkipLayers = [42] };
        Assert.Throws<ArgumentOutOfRangeException>(() => guidance.Validate(42));
        MossImageGuidance.Reference.Validate(42);
    }

    /// <summary>The reference's coordinate rule on a 2×3 grid with 4 text tokens: image tokens sit at
    /// <c>(0, 4, row − 0.5, col − 1)</c>, text token <c>i</c> at <c>(0, i, 0, 0)</c>; channel <c>i</c> pairs with
    /// <c>i + 64</c> and reads axis <c>i / 16</c>.</summary>
    [Fact]
    public void Rope_PlacesImageAndTextTokensOnTheReferenceCoordinates()
    {
        const int headDim = 128;
        float[] bands = MossImageRope.Bands(headDim, 10000f);
        Assert.Equal(16, bands.Length);
        Assert.Equal(1f, bands[0]);
        (Tensor cos, Tensor sin) = MossImageRope.Joint(2, 3, 4, headDim, 10000f);
        using (cos)
        using (sin)
        {
            ReadOnlySpan<float> s = sin.AsReadOnlySpan<float>();
            // Token 5 is image (row 1, col 2): t = 4, h = 0.5, w = 1. Channel 16 (axis t, band 0), 32 (axis h), 48 (axis w).
            int token = 5;
            Assert.Equal(MathF.Sin(4f), s[token * headDim + 16], 6);
            Assert.Equal(MathF.Sin(0.5f), s[token * headDim + 32], 6);
            Assert.Equal(MathF.Sin(1f), s[token * headDim + 48], 6);
            Assert.Equal(0f, s[token * headDim + 0]);
            Assert.Equal(s[token * headDim + 16], s[token * headDim + 16 + 64]);
            // Token 8 is text token 2: only axis t moves.
            Assert.Equal(MathF.Sin(2f), s[8 * headDim + 16], 6);
            Assert.Equal(0f, s[8 * headDim + 32]);
        }
    }

    [Fact]
    public void Converter_RecognisesTheReferenceKeySet_AndRefusesStrangers()
    {
        string[] keys = [.. MossImageCheckpointConverter.SignatureKeys];
        Assert.True(MossImageCheckpointConverter.MatchesByKeys(keys.Select(k => "model.diffusion_model." + k)));
        Assert.False(MossImageCheckpointConverter.MatchesByKeys(["shared_adaLN_modulation.1.weight", "layers.0.mlp.gate_proj.weight"]));
        Dictionary<string, Tensor> weights = keys.ToDictionary(k => k, _ => new Tensor(new TensorShape(1), DType.F32));
        weights["rope.bands"] = new Tensor(new TensorShape(16), DType.F32);
        Dictionary<string, Tensor> converted = MossImageCheckpointConverter.ConvertTransformer(weights);
        Assert.False(converted.ContainsKey("rope.bands"));
        weights["something_else.weight"] = new Tensor(new TensorShape(1), DType.F32);
        Assert.Throws<NotSupportedException>(() => MossImageCheckpointConverter.ConvertTransformer(weights));
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    private static Tensor Make(float[] values)
    {
        Tensor tensor = new(new TensorShape(1, values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>Two distinct conditionings from the tiny fixture, for the branch-topology tests.</summary>
    private sealed class StubConditionings : IDisposable
    {
        private readonly MossImageFixture _fixture = new();
        private readonly CpuBackend _backend = new();

        public StubConditionings()
        {
            Tensor context = _fixture.Tensors["inputs.context"];
            Cond = _fixture.Transformer.PrepareConditioning(_backend, context, MossImageFixture.Height, MossImageFixture.Width);
            Uncond = _fixture.Transformer.PrepareConditioning(_backend, context, MossImageFixture.Height, MossImageFixture.Width);
        }

        public MossImageConditioning Cond { get; }

        public MossImageConditioning Uncond { get; }

        public void Dispose()
        {
            Cond.Dispose();
            Uncond.Dispose();
            _backend.Dispose();
            _fixture.Dispose();
        }
    }
}
