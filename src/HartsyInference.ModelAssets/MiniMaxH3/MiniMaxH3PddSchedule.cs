using HartsyInference.Core.Exceptions;

namespace HartsyInference.ModelAssets.MiniMaxH3;

/// <summary>Exact 32-interval schedule envelope declared by the published MiniMax-H3 PDD adapters.</summary>
public sealed class MiniMaxH3PddSchedule
{
    /// <summary>Number of trained fine intervals in the published head banks.</summary>
    public const int PublishedFineSteps = 32;

    /// <summary>Smallest trained grouping of adjacent fine intervals.</summary>
    public const int PublishedBlockSize = 4;

    /// <summary>Required video flow shift.</summary>
    public const double PublishedVideoShift = 12.0;

    /// <summary>Required audio flow shift.</summary>
    public const double PublishedAudioShift = 3.0;

    /// <summary>Absolute tolerance used only to absorb float serialization at trained knots.</summary>
    public const double KnotTolerance = 1e-6;

    private readonly MiniMaxH3PddStep[] _steps;
    private readonly double[] _sigmas;

    private MiniMaxH3PddSchedule(MiniMaxH3PddStep[] steps, double[] sigmas)
    {
        _steps = steps;
        _sigmas = sigmas;
    }

    /// <summary>Number of transformer evaluations in the coarse schedule.</summary>
    public int Nfe => _steps.Length;

    /// <summary>Descending trained video-sigma boundaries, including terminal zero.</summary>
    public IReadOnlyList<double> Sigmas => _sigmas;

    /// <summary>Coarse intervals and their separate video/audio fusion coefficients.</summary>
    public IReadOnlyList<MiniMaxH3PddStep> Steps => _steps;

    /// <summary>Validates the locked recipe and builds its exact trained-knot schedule.</summary>
    public static MiniMaxH3PddSchedule Create(MiniMaxH3PddExecutionSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        if (!string.Equals(settings.Sampler, "euler", StringComparison.OrdinalIgnoreCase))
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 PDD requires the plain Euler sampler; got '{settings.Sampler}'.");
        }
        if (Math.Abs(settings.CfgScale - 1.0f) > 1e-6f)
            throw new HartsyInferenceException($"MiniMax-H3 PDD locks CFG to 1; got {settings.CfgScale}.");
        if (Math.Abs(settings.VideoFlowShift - PublishedVideoShift) > 1e-9
            || Math.Abs(settings.AudioFlowShift - PublishedAudioShift) > 1e-9)
        {
            throw new HartsyInferenceException(
                $"MiniMax-H3 PDD requires video/audio shifts {PublishedVideoShift}/{PublishedAudioShift}; "
                + $"got {settings.VideoFlowShift}/{settings.AudioFlowShift}.");
        }
        if (Math.Abs(settings.Strength - 1.0f) > 1e-6f)
            throw new HartsyInferenceException($"MiniMax-H3 PDD requires adapter strength 1; got {settings.Strength}.");
        if (settings.HasOtherDistillation || settings.HasVsa || settings.HasStepCache || settings.HasHybrid)
        {
            throw new HartsyInferenceException(
                "MiniMax-H3 PDD cannot stack with Turbo/another distillation adapter, VSA, step caching, or Hybrid packing.");
        }

        int[] partition = settings.Nfe switch
        {
            4 => [8, 8, 8, 8],
            6 => [8, 8, 4, 4, 4, 4],
            8 => [4, 4, 4, 4, 4, 4, 4, 4],
            _ => throw new HartsyInferenceException(
                $"MiniMax-H3 PDD supports only NFE 4, 6, or 8; got {settings.Nfe}."),
        };

        double[] videoFine = FineSigmas(PublishedVideoShift);
        double[] audioFine = FineSigmas(PublishedAudioShift);
        MiniMaxH3PddStep[] steps = new MiniMaxH3PddStep[partition.Length];
        double[] sigmas = new double[partition.Length + 1];
        int start = 0;
        for (int i = 0; i < partition.Length; i++)
        {
            int count = partition[i];
            float[] videoWeights = IntervalWeights(videoFine, start, count);
            float[] audioWeights = IntervalWeights(audioFine, start, count);
            sigmas[i] = videoFine[start];
            steps[i] = new MiniMaxH3PddStep(i, start, count, videoFine[start], videoFine[start + count],
                videoWeights, audioWeights);
            start += count;
        }
        sigmas[^1] = 0.0;
        return new MiniMaxH3PddSchedule(steps, sigmas);
    }

    /// <summary>Resolves an actual Euler interval and rejects off-grid or skipped trained boundaries.</summary>
    public MiniMaxH3PddStep Resolve(double sigma, double sigmaNext)
    {
        for (int i = 0; i < _steps.Length; i++)
        {
            MiniMaxH3PddStep step = _steps[i];
            if (Math.Abs(sigma - step.Sigma) <= KnotTolerance
                && Math.Abs(sigmaNext - step.SigmaNext) <= KnotTolerance)
            {
                return step;
            }
        }
        throw new HartsyInferenceException(
            $"MiniMax-H3 PDD was evaluated off its trained grid at sigma {sigma:G9} -> {sigmaNext:G9}. "
            + $"Use the exact NFE-{Nfe} PDD schedule with plain Euler.");
    }

    /// <summary>Returns the 33 descending shifted-sigma knots used to train one modality's head bank.</summary>
    public static double[] FineSigmas(double shift)
    {
        if (!(shift > 0.0) || !double.IsFinite(shift))
            throw new ArgumentOutOfRangeException(nameof(shift), "Flow shift must be finite and positive.");
        double[] result = new double[PublishedFineSteps + 1];
        for (int i = 0; i <= PublishedFineSteps; i++)
        {
            double baseSigma = (PublishedFineSteps - i) / (double)PublishedFineSteps;
            result[i] = shift * baseSigma / (1.0 + (shift - 1.0) * baseSigma);
        }
        result[0] = 1.0;
        result[^1] = 0.0;
        return result;
    }

    private static float[] IntervalWeights(double[] fine, int start, int count)
    {
        double span = fine[start] - fine[start + count];
        float[] result = new float[count];
        double sum = 0.0;
        for (int i = 0; i < count; i++)
        {
            double delta = fine[start + i] - fine[start + i + 1];
            result[i] = (float)(delta / span);
            sum += result[i];
        }
        result[^1] += (float)(1.0 - sum);
        return result;
    }
}
