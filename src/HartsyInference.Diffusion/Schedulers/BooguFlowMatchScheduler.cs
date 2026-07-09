using HartsyInference.Core.Schedulers;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Flow-match Euler scheduler for Boogu-Image (<c>FlowMatchEulerDiscreteScheduler</c> with
/// <c>time_shift_version="v1"</c> from <c>boogu/schedulers/scheduling_flow_match_euler_discrete_time_shifting.py</c>).
///
/// <para>Boogu integrates the rectified-flow ODE in the <b>ascending</b> direction: the initial latent is pure noise
/// at <c>t=0</c> and the model is integrated forward to clean data at <c>t=1</c> via
/// <c>x_{next} = x + v · (t_next − t)</c>. This is the opposite parameterization to the SD3/Flux
/// <see cref="FlowMatchEulerDiscreteScheduler"/> (which descends from <c>t=1</c> with a negative step), so it gets its
/// own class rather than overloading the shared one.</para>
///
/// <para>Timesteps: <c>t = linspace(0, 1, N+1)[:-1]</c> (ascending), then the static <b>v1</b> logistic time-shift
/// with <c>mu = lin(seq_len)</c> over the line <c>(256 → base_shift)…(4096 → max_shift)</c> (defaults 0.5 / 1.15):</para>
/// <code>
/// t1 = clip(1 − t, eps, 1−eps)
/// y  = exp(mu) / (exp(mu) + (1/t1 − 1)^sigma)      // sigma = 1
/// t' = 1 − y
/// </code>
/// The shifted <c>t'</c> values feed the model directly (the transformer scales by <c>timestep_scale=1000</c> inside
/// its sinusoidal embedding). A terminal <c>1.0</c> is appended so the final step integrates fully to data.
///
/// <para>When <see cref="DoShift"/> is false the raw linspace is used (Turbo paths may opt out). Set
/// <see cref="SeqLen"/> to the packed image token count <c>(H/8/2)·(W/8/2)</c>; the released config pins it at 4096.</para></summary>
public sealed class BooguFlowMatchScheduler : IScheduler
{
    private readonly bool _doShift;
    private readonly int _seqLen;
    private readonly float _baseShift;
    private readonly float _maxShift;

    private float[] _timesteps;
    private float[] _timestepsFull;
    private int _numInferenceSteps;

    /// <summary>Name of this scheduler.</summary>
    public string Name => "boogu_flow_match_v1";

    /// <summary>Number of inference steps configured.</summary>
    public int NumInferenceSteps => _numInferenceSteps;

    /// <summary>The shifted per-step timestep values <c>t'</c> in <c>[0, 1]</c> (ascending). Pass these to the transformer.</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>Initial latent is pure noise; no scaling.</summary>
    public float InitialNoiseSigma => 1.0f;

    /// <summary>Flow-match schedulers do not scale model input.</summary>
    public float ScaleModelInput(int stepIndex) => 1.0f;

    /// <summary>Whether the v1 logistic time-shift is applied (true for the released Base/Edit configs).</summary>
    public bool DoShift => _doShift;

    /// <summary>Token count used to derive the static shift <c>mu = lin(seq_len)</c> (4096 in the released config).</summary>
    public int SeqLen => _seqLen;

    /// <summary>Creates the Boogu v1 flow-match scheduler.</summary>
    /// <param name="seqLen">Packed image token count for the static shift (4096 for ~1024² in the released config).</param>
    /// <param name="doShift">Apply the v1 logistic shift. False uses the raw linspace.</param>
    /// <param name="baseShift">Shift at 256 tokens (0.5).</param>
    /// <param name="maxShift">Shift at 4096 tokens (1.15).</param>
    public BooguFlowMatchScheduler(int seqLen = 4096, bool doShift = true, float baseShift = 0.5f, float maxShift = 1.15f)
    {
        _seqLen = seqLen;
        _doShift = doShift;
        _baseShift = baseShift;
        _maxShift = maxShift;
        _timesteps = Array.Empty<float>();
        _timestepsFull = Array.Empty<float>();
    }

    /// <summary>Configures the ascending shifted timestep schedule for the given step count.</summary>
    public void SetTimesteps(int numInferenceSteps)
    {
        _numInferenceSteps = numInferenceSteps;
        _timesteps = new float[numInferenceSteps];
        _timestepsFull = new float[numInferenceSteps + 1];

        // mu from the (256 -> baseShift) .. (4096 -> maxShift) line, evaluated at seq_len.
        double m = (_maxShift - _baseShift) / (4096.0 - 256.0);
        double b = _baseShift - m * 256.0;
        double mu = m * _seqLen + b;
        double expMu = Math.Exp(mu);
        const double eps = 1e-8;

        for (int i = 0; i < numInferenceSteps; i++)
        {
            double t = (double)i / numInferenceSteps; // linspace(0,1,N+1)[:-1]
            double shifted;
            if (_doShift)
            {
                double t1 = Math.Clamp(1.0 - t, eps, 1.0 - eps);
                double y = expMu / (expMu + (1.0 / t1 - 1.0)); // sigma = 1
                shifted = 1.0 - y;
            }
            else
            {
                shifted = t;
            }
            _timesteps[i] = (float)shifted;
            _timestepsFull[i] = (float)shifted;
        }
        _timestepsFull[numInferenceSteps] = 1.0f; // terminal: integrate fully to data
    }

    /// <summary>The Euler step delta <c>t_next − t</c> (positive; t ascends 0 → 1) — the <c>delta</c> for the
    /// device <c>IBackend.CfgEulerStep</c> drain-free path, identical to the dt used by <see cref="Step"/>.</summary>
    public float Dt(int stepIndex) => _timestepsFull[stepIndex + 1] - _timestepsFull[stepIndex];

    /// <summary>One forward Euler step: <c>x_next = x + v · (t_next − t)</c>.</summary>
    public unsafe void Step(Tensor output, Tensor modelOutput, Tensor sample, int stepIndex)
    {
        float t = _timestepsFull[stepIndex];
        float tNext = _timestepsFull[stepIndex + 1];
        float dt = tNext - t; // positive (t ascends 0 -> 1)

        float* modelPtr = (float*)modelOutput.DataPointer;
        float* samplePtr = (float*)sample.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        for (int i = 0; i < count; i++)
            outPtr[i] = samplePtr[i] + modelPtr[i] * dt;
    }

    /// <summary>Mixes a clean latent toward noise at the given step: <c>x_t = (1 − t)·noise + t·sample</c> (t=0 noise,
    /// t=1 data). Used for img2img-style partial starts; standard edit keeps the reference in a separate stream.</summary>
    public unsafe void AddNoise(Tensor output, Tensor sample, Tensor noise, int stepIndex)
    {
        float t = _timestepsFull[stepIndex];
        float oneMinusT = 1.0f - t;

        float* samplePtr = (float*)sample.DataPointer;
        float* noisePtr = (float*)noise.DataPointer;
        float* outPtr = (float*)output.DataPointer;
        int count = (int)sample.ElementCount;

        for (int i = 0; i < count; i++)
            outPtr[i] = oneMinusT * samplePtr[i] + t * noisePtr[i];
    }
}
