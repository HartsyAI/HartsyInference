using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Schedulers;

/// <summary>DPM-Solver++ 2M (midpoint) for flow-matching velocity prediction — port of Wan's
/// <c>FlowDPMSolverMultistepScheduler</c> (<c>wanxiang/utils/fm_solvers.py</c>) driven through
/// <c>get_sampling_sigmas</c>, which is what Wan-Animate-2 samples with. Flow parameterization:
/// <c>x0 = sample − sigma·v</c>, <c>alpha = 1 − sigma</c>, <c>lambda = log(alpha) − log(sigma)</c>.
///
/// <para>Distinct from <see cref="DpmPlusPlus2MScheduler"/>, which is the VP-diffusion solver (alphas_cumprod,
/// epsilon/v-prediction) and cannot consume a flow velocity. Distinct from <see cref="FlowUniPCMultistepScheduler"/>
/// in the <b>sigma grid</b> as well as the solver: UniPC's grid floors at <c>1/num_train_timesteps</c>, this one
/// linspaces to an exact 0 before the shift, so no two grid points coincide between them.</para>
///
/// <para>Fixed to the reference's configuration: <c>solver_order = 2</c>, <c>solver_type = "midpoint"</c>,
/// <c>algorithm_type = "dpmsolver++"</c>, <c>lower_order_final = true</c>, <c>final_sigmas_type = "zero"</c>.
/// The last clause makes the <b>final step first-order at every step count</b>, not only below 15 — so the
/// second-order update never sees <c>sigma = 0</c>.</para></summary>
public sealed unsafe class FlowDpmPlusPlus2MScheduler
{
    /// <summary>Model outputs the reference keeps for the multistep update.</summary>
    public const int SolverOrder = 2;

    private readonly int _numTrainTimesteps;

    private float[] _sigmas = [];
    private float[] _timesteps = [];
    private float[]? _m0, _m1;   // converted x0 predictions: current, previous
    private int _stepIndex;
    private int _lowerOrderNums;

    public FlowDpmPlusPlus2MScheduler(int numTrainTimesteps = 1000)
    {
        if (numTrainTimesteps < 1) throw new ArgumentOutOfRangeException(nameof(numTrainTimesteps));
        _numTrainTimesteps = numTrainTimesteps;
    }

    /// <summary>Number of inference steps configured by the last <see cref="SetTimesteps"/>.</summary>
    public int NumInferenceSteps => _timesteps.Length;

    /// <summary>Model-conditioning timesteps, <b>truncated to integers</b> — the reference casts
    /// <c>sigma · num_train_timesteps</c> to int64 before it reaches the DiT.</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>The sigma grid (length steps + 1, terminal exactly 0).</summary>
    public ReadOnlySpan<float> Sigmas => _sigmas;

    /// <summary>Builds <c>get_sampling_sigmas</c>: <c>linspace(1, 0, steps + 1)[:steps]</c> shifted by
    /// <c>sigma ← shift·sigma / (1 + (shift−1)·sigma)</c>, terminal 0 appended. <c>sigma[0]</c> is 1 for any shift and
    /// the last non-zero sigma is <c>shift / (steps + shift − 1)</c>. Resets the multistep history.</summary>
    public void SetTimesteps(int numInferenceSteps, float shift)
    {
        if (numInferenceSteps < 1) throw new ArgumentOutOfRangeException(nameof(numInferenceSteps));
        if (shift <= 0f) throw new ArgumentOutOfRangeException(nameof(shift), "Flow shift must be positive.");
        _sigmas = new float[numInferenceSteps + 1];
        _timesteps = new float[numInferenceSteps];
        for (int i = 0; i < numInferenceSteps; i++)
        {
            double s = (double)(numInferenceSteps - i) / numInferenceSteps;
            double sigma = shift * s / (1.0 + (shift - 1.0) * s);
            _sigmas[i] = (float)sigma;
            // The reference truncates from the unrounded sigma: `timesteps` is computed before `sigmas` is cast down.
            _timesteps[i] = (float)Math.Truncate(sigma * _numTrainTimesteps);
        }
        _sigmas[numInferenceSteps] = 0f;

        _m0 = _m1 = null;
        _stepIndex = 0;
        _lowerOrderNums = 0;
    }

    /// <summary>Advances <paramref name="sample"/> in place from <c>sigmas[i]</c> to <c>sigmas[i+1]</c> given the
    /// flow-matching velocity <paramref name="modelOutput"/> evaluated at <c>Timesteps[i]</c>. First-order on the
    /// first step (no history) and on the last step (<c>final_sigmas_type = "zero"</c>); second-order midpoint in
    /// between.</summary>
    public void Step(Tensor sample, Tensor modelOutput)
    {
        if (_timesteps.Length == 0) throw new InvalidOperationException("Call SetTimesteps before Step.");
        if (_stepIndex >= _timesteps.Length) throw new InvalidOperationException("Trajectory complete — call SetTimesteps to start a new one.");
        long n = sample.Shape.ElementCount;
        if (modelOutput.Shape.ElementCount != n)
            throw new ArgumentException("modelOutput element count must match sample.", nameof(modelOutput));

        // convert_model_output (dpmsolver++, flow_prediction): x0 = sample − sigma_t · v
        float sigmaS0 = _sigmas[_stepIndex];
        float[] converted = new float[n];
        float* sp = (float*)sample.DataPointer;
        float* mp = (float*)modelOutput.DataPointer;
        for (long i = 0; i < n; i++) converted[i] = sp[i] - sigmaS0 * mp[i];

        _m1 = _m0;
        _m0 = converted;

        float sigmaT = _sigmas[_stepIndex + 1];
        double alphaT = 1.0 - sigmaT;
        double h = Lambda(sigmaT) - Lambda(sigmaS0);
        // exp(-h) is 0 at the terminal sigma (h = +inf), which collapses the update to x_t = x0 exactly.
        double expNegH = Math.Exp(-h);
        double sampleCoeff = (double)sigmaT / sigmaS0;
        double d0Coeff = alphaT * (expNegH - 1.0);

        bool firstOrder = _lowerOrderNums < 1 || _stepIndex == _timesteps.Length - 1;
        if (firstOrder)
        {
            for (long i = 0; i < n; i++) sp[i] = (float)(sampleCoeff * sp[i] - d0Coeff * _m0[i]);
        }
        else
        {
            double r0 = (Lambda(sigmaS0) - Lambda(_sigmas[_stepIndex - 1])) / h;
            double d1Scale = 1.0 / r0;
            float[] m1 = _m1!;
            for (long i = 0; i < n; i++)
            {
                double d1 = d1Scale * (_m0[i] - m1[i]);
                sp[i] = (float)(sampleCoeff * sp[i] - d0Coeff * _m0[i] - 0.5 * d0Coeff * d1);
            }
        }

        if (_lowerOrderNums < SolverOrder) _lowerOrderNums++;
        _stepIndex++;
    }

    /// <summary><c>log(1 − sigma) − log(sigma)</c>, the flow-matching half-log-SNR.</summary>
    private static double Lambda(float sigma) => Math.Log(1.0 - sigma) - Math.Log(sigma);
}
