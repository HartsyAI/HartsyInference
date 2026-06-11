using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Schedulers;

/// <summary>UniPC multistep predictor–corrector adapted to flow matching — port of the <c>FlowUniPCMultistepScheduler</c>
/// shipped with Wan2.x / Matrix-Game (<c>wan/utils/fm_solvers_unipc.py</c>, itself derived from diffusers UniPC).
/// Used by Matrix-Game 3.0 at both 50 steps (base) and 3 steps (DMD-distilled). Flow-matching mapping:
/// <c>alpha_t = 1 − sigma_t</c>, <c>lambda_t = log(alpha_t) − log(sigma_t)</c>, x0-prediction
/// <c>x0 = sample − sigma_t · v</c>. Stateful across a trajectory (multistep history + corrector) — call
/// <see cref="SetTimesteps"/> to start each new trajectory/segment.
///
/// <para>Unlike <see cref="FlowMatchEulerDiscreteScheduler"/>, <see cref="Step"/> updates the sample <b>in place</b>
/// and advances the internal step index; the corrector retro-corrects the previous prediction using the current model
/// output before the predictor extrapolates, matching the reference exactly.</para></summary>
public sealed unsafe class FlowUniPCMultistepScheduler
{
    /// <summary>UniPC B(h) variant: <c>Bh1 = h</c>, <c>Bh2 = expm1(h)</c>. Reference default is Bh2.</summary>
    public enum SolverVariant
    {
        Bh1,
        Bh2,
    }

    private readonly int _solverOrder;
    private readonly SolverVariant _variant;
    private readonly bool _lowerOrderFinal;
    private readonly int _numTrainTimesteps;

    private float[] _sigmas = [];
    private float[] _timesteps = [];
    private readonly List<float[]> _modelOutputs = new();   // converted x0 predictions, oldest-first
    private float[]? _lastSample;
    private int _stepIndex;
    private int _lowerOrderNums;
    private int _thisOrder;

    public FlowUniPCMultistepScheduler(int solverOrder = 2, SolverVariant variant = SolverVariant.Bh2,
        bool lowerOrderFinal = true, int numTrainTimesteps = 1000)
    {
        if (solverOrder is < 1 or > 3) throw new ArgumentOutOfRangeException(nameof(solverOrder), "UniPC solver order must be 1-3.");
        _solverOrder = solverOrder;
        _variant = variant;
        _lowerOrderFinal = lowerOrderFinal;
        _numTrainTimesteps = numTrainTimesteps;
    }

    /// <summary>Number of inference steps configured by the last <see cref="SetTimesteps"/>.</summary>
    public int NumInferenceSteps => _timesteps.Length;

    /// <summary>Model-conditioning timesteps (<c>sigma · num_train_timesteps</c>), one per step.</summary>
    public ReadOnlySpan<float> Timesteps => _timesteps;

    /// <summary>The sigma grid (length steps + 1, terminal 0).</summary>
    public ReadOnlySpan<float> Sigmas => _sigmas;

    /// <summary>Builds the shifted sigma schedule (<c>σ ← shift·σ / (1 + (shift−1)·σ)</c>) and resets all multistep
    /// state. Call once per trajectory (Matrix-Game instantiates a fresh schedule per segment).</summary>
    public void SetTimesteps(int numInferenceSteps, float shift)
    {
        if (numInferenceSteps < 1) throw new ArgumentOutOfRangeException(nameof(numInferenceSteps));
        float sigmaMax = 1.0f;
        float sigmaMin = 1.0f / _numTrainTimesteps;
        _sigmas = new float[numInferenceSteps + 1];
        _timesteps = new float[numInferenceSteps];
        for (int i = 0; i < numInferenceSteps; i++)
        {
            float sigma = sigmaMax + (sigmaMin - sigmaMax) * i / numInferenceSteps;
            sigma = shift * sigma / (1f + (shift - 1f) * sigma);
            _sigmas[i] = sigma;
            _timesteps[i] = sigma * _numTrainTimesteps;
        }
        _sigmas[numInferenceSteps] = 0f;   // final_sigmas_type = "zero"

        _modelOutputs.Clear();
        _lastSample = null;
        _stepIndex = 0;
        _lowerOrderNums = 0;
        _thisOrder = 1;
    }

    /// <summary>Advances <paramref name="sample"/> in place from <c>sigmas[i]</c> to <c>sigmas[i+1]</c> given the
    /// flow-matching velocity <paramref name="modelOutput"/> evaluated at <c>Timesteps[i]</c>. Applies the UniC
    /// corrector to the previous prediction first (when history exists), then the UniP predictor.</summary>
    public void Step(Tensor sample, Tensor modelOutput)
    {
        if (_timesteps.Length == 0) throw new InvalidOperationException("Call SetTimesteps before Step.");
        if (_stepIndex >= _timesteps.Length) throw new InvalidOperationException("Trajectory complete — call SetTimesteps to start a new one.");
        long n = sample.Shape.ElementCount;
        if (modelOutput.Shape.ElementCount != n)
            throw new ArgumentException("modelOutput element count must match sample.", nameof(modelOutput));

        // convert_model_output (predict_x0, flow prediction): x0 = sample − sigma_t · v
        float sigmaT = _sigmas[_stepIndex];
        float[] converted = new float[n];
        float* sp = (float*)sample.DataPointer;
        float* mp = (float*)modelOutput.DataPointer;
        for (long i = 0; i < n; i++) converted[i] = sp[i] - sigmaT * mp[i];

        bool useCorrector = _stepIndex > 0 && _lastSample is not null;
        if (useCorrector)
            UniCUpdate(sample, converted, _thisOrder);

        // Shift model-output history.
        _modelOutputs.Add(converted);
        while (_modelOutputs.Count > _solverOrder) _modelOutputs.RemoveAt(0);

        int thisOrder = _lowerOrderFinal
            ? Math.Min(_solverOrder, _timesteps.Length - _stepIndex)
            : _solverOrder;
        _thisOrder = Math.Min(thisOrder, _lowerOrderNums + 1);

        _lastSample = new float[n];
        fixed (float* lp = _lastSample)
            Buffer.MemoryCopy((float*)sample.DataPointer, lp, n * 4, n * 4);

        UniPUpdate(sample, _thisOrder);

        if (_lowerOrderNums < _solverOrder) _lowerOrderNums++;
        _stepIndex++;
    }

    /// <summary>UniP predictor: extrapolates the sample from <c>sigmas[step]</c> to <c>sigmas[step+1]</c> using the
    /// converted-output history.</summary>
    private void UniPUpdate(Tensor sample, int order)
    {
        float sigmaS0 = _sigmas[_stepIndex];
        float sigmaT = _sigmas[_stepIndex + 1];
        float[] m0 = _modelOutputs[^1];
        long n = m0.Length;

        (double lambdaT, double alphaT) = LambdaAlpha(sigmaT);
        (double lambdaS0, _) = LambdaAlpha(sigmaS0);
        double h = lambdaT - lambdaS0;
        double hh = -h;   // predict_x0

        double[] rks = new double[order - 1];
        float[][] prev = new float[order - 1][];
        for (int i = 1; i < order; i++)
        {
            float sigmaSi = _sigmas[_stepIndex - i];
            (double lambdaSi, _) = LambdaAlpha(sigmaSi);
            rks[i - 1] = (lambdaSi - lambdaS0) / h;
            prev[i - 1] = _modelOutputs[^(i + 1)];
        }

        (double hPhi1, double bH, double[] b, double[][] r) = BuildSystem(hh, order, rks);

        double[] rhos;
        if (order == 1) rhos = [];
        else if (order == 2) rhos = [0.5];
        else rhos = SolveLinear(r, b, order - 1);

        float* sp = (float*)sample.DataPointer;
        for (long e = 0; e < n; e++)
        {
            double predRes = 0;
            for (int i = 0; i < rhos.Length; i++)
                predRes += rhos[i] * ((prev[i][e] - m0[e]) / rks[i]);
            double xT = (sigmaT / sigmaS0) * sp[e] - alphaT * hPhi1 * m0[e] - alphaT * bH * predRes;
            sp[e] = (float)xT;
        }
    }

    /// <summary>UniC corrector: retro-corrects the previous predictor result (now in <paramref name="sample"/>) using
    /// the model output evaluated at the current step (<paramref name="thisConverted"/>). Writes in place.</summary>
    private void UniCUpdate(Tensor sample, float[] thisConverted, int order)
    {
        float sigmaT = _sigmas[_stepIndex];
        float sigmaS0 = _sigmas[_stepIndex - 1];
        float[] m0 = _modelOutputs[^1];
        float[] x = _lastSample!;
        long n = m0.Length;

        (double lambdaT, double alphaT) = LambdaAlpha(sigmaT);
        (double lambdaS0, _) = LambdaAlpha(sigmaS0);
        double h = lambdaT - lambdaS0;
        double hh = -h;

        double[] rks = new double[order - 1];
        float[][] prev = new float[order - 1][];
        for (int i = 1; i < order; i++)
        {
            float sigmaSi = _sigmas[_stepIndex - 1 - i];
            (double lambdaSi, _) = LambdaAlpha(sigmaSi);
            rks[i - 1] = (lambdaSi - lambdaS0) / h;
            prev[i - 1] = _modelOutputs[^(i + 1)];
        }

        (double hPhi1, double bH, double[] b, double[][] r) = BuildSystem(hh, order, rks);

        double[] rhosC;
        if (order == 1) rhosC = [0.5];
        else rhosC = SolveLinear(r, b, order);

        float* sp = (float*)sample.DataPointer;
        for (long e = 0; e < n; e++)
        {
            double corrRes = 0;
            for (int i = 0; i < order - 1; i++)
                corrRes += rhosC[i] * ((prev[i][e] - m0[e]) / rks[i]);
            double d1T = thisConverted[e] - m0[e];
            double xT = (sigmaT / sigmaS0) * x[e] - alphaT * hPhi1 * m0[e]
                        - alphaT * bH * (corrRes + rhosC[^1] * d1T);
            sp[e] = (float)xT;
        }
    }

    /// <summary>Builds the UniPC coefficient system for step size <paramref name="hh"/>: returns
    /// <c>h·φ₁ = expm1(h)</c>, <c>B(h)</c>, the rhs vector <c>b</c>, and the Vandermonde-style matrix <c>R</c>
    /// (rows <c>rks^(i−1)</c> + the trailing 1.0 column for the corrector's D1_t term).</summary>
    private (double HPhi1, double BH, double[] B, double[][] R) BuildSystem(double hh, int order, double[] rks)
    {
        double hPhi1 = Math.Exp(hh) - 1.0;   // expm1
        double bH = _variant == SolverVariant.Bh1 ? hh : hPhi1;

        // rks for the system include the trailing 1.0 (the current point) used by the corrector.
        double[] rkAll = new double[rks.Length + 1];
        Array.Copy(rks, rkAll, rks.Length);
        rkAll[^1] = 1.0;

        double[] b = new double[order];
        double[][] r = new double[order][];
        double hPhiK = hPhi1 / hh - 1.0;
        double factorial = 1.0;
        for (int i = 1; i <= order; i++)
        {
            r[i - 1] = new double[rkAll.Length];
            for (int j = 0; j < rkAll.Length; j++)
                r[i - 1][j] = Math.Pow(rkAll[j], i - 1);
            b[i - 1] = hPhiK * factorial / bH;
            factorial *= i + 1;
            hPhiK = hPhiK / hh - 1.0 / factorial;
        }

        return (hPhi1, bH, b, r);
    }

    /// <summary>Solves the leading <paramref name="size"/>×<paramref name="size"/> block of <paramref name="r"/>·x = <paramref name="b"/> by Gaussian elimination (size ≤ 3).</summary>
    private static double[] SolveLinear(double[][] r, double[] b, int size)
    {
        double[][] a = new double[size][];
        double[] rhs = new double[size];
        for (int i = 0; i < size; i++)
        {
            a[i] = new double[size];
            Array.Copy(r[i], a[i], size);
            rhs[i] = b[i];
        }
        for (int col = 0; col < size; col++)
        {
            int pivot = col;
            for (int row = col + 1; row < size; row++)
                if (Math.Abs(a[row][col]) > Math.Abs(a[pivot][col])) pivot = row;
            (a[col], a[pivot]) = (a[pivot], a[col]);
            (rhs[col], rhs[pivot]) = (rhs[pivot], rhs[col]);
            double d = a[col][col];
            if (Math.Abs(d) < 1e-12) throw new InvalidOperationException("Singular UniPC coefficient system.");
            for (int row = col + 1; row < size; row++)
            {
                double f = a[row][col] / d;
                for (int k = col; k < size; k++) a[row][k] -= f * a[col][k];
                rhs[row] -= f * rhs[col];
            }
        }
        double[] x = new double[size];
        for (int row = size - 1; row >= 0; row--)
        {
            double v = rhs[row];
            for (int k = row + 1; k < size; k++) v -= a[row][k] * x[k];
            x[row] = v / a[row][row];
        }
        return x;
    }

    private static (double Lambda, double Alpha) LambdaAlpha(float sigma)
    {
        // Flow matching: alpha = 1 − sigma. Clamp the terminal sigma=0 to keep lambda finite.
        double s = Math.Max(sigma, 1e-6);
        double alpha = 1.0 - sigma;
        return (Math.Log(alpha) - Math.Log(s), alpha);
    }
}
