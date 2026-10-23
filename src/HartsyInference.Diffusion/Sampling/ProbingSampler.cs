using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>A sampler whose model evaluations pass through the <see cref="DenoiserProbe"/>; <see cref="SamplerRegistry.Create"/>
/// wraps every sampler it builds in one while the probe is set.</summary>
internal sealed class ProbingSampler(ISampler inner, DenoiserProbe probe) : ISampler
{
    private readonly ISampler _inner = inner;
    private readonly DenoiserProbe _probe = probe;

    public string Name => _inner.Name;

    public int StepCount => _inner.StepCount;

    public IReadOnlyList<float> Sigmas => _inner.Sigmas;

    public void Reset(TensorShape latentShape) => _inner.Reset(latentShape);

    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex) =>
        _inner.Step(backend, z, new Predictor(predictor, _probe), stepIndex);

    /// <summary>Probes the first evaluation of each probed step; the pair the sampler integrates is always the arm's own.</summary>
    private sealed class Predictor(IDenoisePredictor inner, DenoiserProbe probe) : IDenoisePredictor
    {
        private static readonly string[] Names = ["cond", "uncond"];

        public PredictionType Prediction => inner.Prediction;

        public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
        {
            bool probed = probe.Takes(stepIndex);
            if (probed && probe.Replays)
                probe.Replay(stepIndex, sigma, [x], inputs => Owned(inner.Predict(inputs[0], sigma, stepIndex)));
            else if (probed)
                probe.RecordInputs(stepIndex, sigma, [x]);
            DenoisePrediction prediction = inner.Predict(x, sigma, stepIndex);
            if (probed && !probe.Replays)
            {
                bool paired = !ReferenceEquals(prediction.Uncond, prediction.Cond);
                probe.RecordOutputs(stepIndex, paired ? prediction.Guidance : 0f, Names, [prediction.Cond, paired ? prediction.Uncond : null]);
            }
            return prediction;
        }

        /// <summary>The pair as tensors the probe may dispose: copies when the prediction only borrows its buffers.</summary>
        private static Tensor?[] Owned(DenoisePrediction prediction)
        {
            Tensor? uncond = ReferenceEquals(prediction.Uncond, prediction.Cond) ? null : prediction.Uncond;
            if (prediction.OwnsTensors)
                return [prediction.Cond, uncond];
            return [DenoiserProbe.Copy(prediction.Cond), uncond is null ? null : DenoiserProbe.Copy(uncond)];
        }
    }
}
