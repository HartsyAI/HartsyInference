using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The two-pass CFG predictor: which branches it runs, in what order, what it pairs, and what it frees when the second pass throws.</summary>
public sealed class TwoPassCfgPredictorTests
{
    private static readonly object Cond = new();
    private static readonly object Uncond = new();

    [Fact]
    public void WithoutAnUnconditionalBranch_OnePassRuns_AndIsPairedWithItself()
    {
        List<object> branches = [];
        using Tensor x = new(new TensorShape(2), DType.F32);
        TwoPassCfgPredictor<object> predictor = new(PredictionType.FlowVelocity, (_, _, _, branch) =>
        {
            branches.Add(branch);
            return new Tensor(new TensorShape(2), DType.F32);
        }, Cond, null, 4f);

        using DenoisePrediction prediction = predictor.Predict(x, 0.5f, 0);

        Assert.Equal(new[] { Cond }, branches);
        Assert.Same(prediction.Cond, prediction.Uncond);
        Assert.Equal(1f, prediction.Guidance);
    }

    [Fact]
    public void WithAnUnconditionalBranch_TheConditionalPassRunsFirst_AndThePairCarriesTheScale()
    {
        List<(object Branch, float Sigma, int Step)> calls = [];
        Dictionary<object, Tensor> outputs = [];
        using Tensor x = new(new TensorShape(2), DType.F32);
        TwoPassCfgPredictor<object> predictor = new(PredictionType.FlowVelocity, (input, sigma, step, branch) =>
        {
            Assert.Same(x, input);
            calls.Add((branch, sigma, step));
            Tensor output = new(new TensorShape(2), DType.F32);
            outputs[branch] = output;
            return output;
        }, Cond, Uncond, 4f);

        using DenoisePrediction prediction = predictor.Predict(x, 0.25f, 3);

        Assert.Equal(new (object Branch, float Sigma, int Step)[] { (Cond, 0.25f, 3), (Uncond, 0.25f, 3) }, calls);
        Assert.Same(outputs[Cond], prediction.Cond);
        Assert.Same(outputs[Uncond], prediction.Uncond);
        Assert.Equal(4f, prediction.Guidance);
        Assert.Equal(PredictionType.FlowVelocity, predictor.Prediction);
    }

    [Fact]
    public void AnUnconditionalPassThatThrows_FreesTheConditionalOutput()
    {
        Tensor? conditional = null;
        using Tensor x = new(new TensorShape(2), DType.F32);
        TwoPassCfgPredictor<object> predictor = new(PredictionType.FlowVelocity, (_, _, _, branch) =>
        {
            if (ReferenceEquals(branch, Uncond))
                throw new InvalidOperationException("second pass");
            conditional = new Tensor(new TensorShape(2), DType.F32);
            return conditional;
        }, Cond, Uncond, 4f);

        InvalidOperationException thrown = Assert.Throws<InvalidOperationException>(() => predictor.Predict(x, 0.5f, 0));

        Assert.Equal("second pass", thrown.Message);
        Assert.NotNull(conditional);
        Assert.Throws<ObjectDisposedException>(() => conditional!.IntoShape(conditional.Shape));
    }
}
