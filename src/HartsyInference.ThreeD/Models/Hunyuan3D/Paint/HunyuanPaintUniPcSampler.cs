namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>diffusers 0.30 <c>UniPCMultistepScheduler</c> as Paint 2.1 configures it: second order, <c>bh2</c>, data prediction on a v-prediction model, trailing
/// timesteps, lower order for the last step and the corrector enabled, final sigma zero. Statefully walks one schedule from the first step to the last.</summary>
public sealed class HunyuanPaintUniPcSampler : IHunyuanPaintSampler
{
    private const int SolverOrder = 2;
    private readonly int[] _timesteps;
    private readonly double[] _sigmas;
    private readonly float[]?[] _outputs = new float[]?[SolverOrder];
    private float[]? _lastSample;
    private int _lowerOrderNums, _thisOrder;

    /// <summary>Prepares <paramref name="steps"/> steps on <paramref name="schedule"/>.</summary>
    public HunyuanPaintUniPcSampler(HunyuanPaintSchedule schedule, int steps)
    {
        ArgumentNullException.ThrowIfNull(schedule);
        _timesteps = HunyuanPaintSchedule.TrailingTimesteps(steps);
        _sigmas = schedule.SigmasAt(_timesteps, 0.0);
    }

    /// <inheritdoc/>
    public IReadOnlyList<int> Timesteps => _timesteps;

    /// <inheritdoc/>
    public double InitialNoiseScale => 1.0;

    /// <inheritdoc/>
    public float[] ScaleModelInput(float[] sample, int step) => sample;

    /// <inheritdoc/>
    public float[] Step(float[] modelOutput, float[] sample, int step, float[]? noise)
    {
        if (step < 0 || step >= _timesteps.Length)
            throw new ArgumentOutOfRangeException(nameof(step));
        (double alpha, double sigma) = AlphaSigma(_sigmas[step]);
        float[] x0 = new float[sample.Length];
        for (int i = 0; i < x0.Length; i++)
            x0[i] = (float)(alpha * sample[i] - sigma * modelOutput[i]);

        if (step > 0 && _lastSample is not null)
            sample = Correct(x0, _lastSample, sample, step, _thisOrder);
        _outputs[0] = _outputs[1];
        _outputs[1] = x0;
        int order = Math.Min(SolverOrder, _timesteps.Length - step);
        _thisOrder = Math.Min(order, _lowerOrderNums + 1);
        _lastSample = sample;
        float[] previous = Predict(sample, step, _thisOrder);
        if (_lowerOrderNums < SolverOrder)
            _lowerOrderNums++;
        return previous;
    }

    private float[] Predict(float[] x, int step, int order)
    {
        float[] m0 = _outputs[^1]!;
        (double alphaT, double sigmaT) = AlphaSigma(_sigmas[step + 1]);
        (double alphaS, double sigmaS) = AlphaSigma(_sigmas[step]);
        double lambdaT = Math.Log(alphaT) - Math.Log(sigmaT), lambdaS = Math.Log(alphaS) - Math.Log(sigmaS), h = lambdaT - lambdaS;
        double hh = -h, hPhi1 = double.ExpM1(hh), bH = double.ExpM1(hh);
        float[] result = new float[x.Length];
        double decay = sigmaT / sigmaS;
        if (order == 1)
        {
            for (int i = 0; i < x.Length; i++)
                result[i] = (float)(decay * x[i] - alphaT * hPhi1 * m0[i]);
            return result;
        }
        float[] m1 = _outputs[^2]!;
        (double alphaI, double sigmaI) = AlphaSigma(_sigmas[step - 1]);
        double rk = (Math.Log(alphaI) - Math.Log(sigmaI) - lambdaS) / h;
        for (int i = 0; i < x.Length; i++)
        {
            double d1 = (m1[i] - m0[i]) / rk;
            result[i] = (float)(decay * x[i] - alphaT * hPhi1 * m0[i] - alphaT * bH * (0.5 * d1));
        }
        return result;
    }

    private float[] Correct(float[] current, float[] last, float[] thisSample, int step, int order)
    {
        float[] m0 = _outputs[^1]!;
        (double alphaT, double sigmaT) = AlphaSigma(_sigmas[step]);
        (double alphaS, double sigmaS) = AlphaSigma(_sigmas[step - 1]);
        double lambdaT = Math.Log(alphaT) - Math.Log(sigmaT), lambdaS = Math.Log(alphaS) - Math.Log(sigmaS), h = lambdaT - lambdaS;
        double hh = -h, hPhi1 = double.ExpM1(hh), bH = double.ExpM1(hh);
        double decay = sigmaT / sigmaS;
        float[] result = new float[thisSample.Length];
        if (order == 1)
        {
            // rhos_c = [0.5]; the correction is the single term on the difference of the new and previous model outputs.
            for (int i = 0; i < result.Length; i++)
                result[i] = (float)(decay * last[i] - alphaT * hPhi1 * m0[i] - alphaT * bH * (0.5 * (current[i] - m0[i])));
            return result;
        }
        float[] m1 = _outputs[^2]!;
        (double alphaI, double sigmaI) = AlphaSigma(_sigmas[step - 2]);
        double rk = (Math.Log(alphaI) - Math.Log(sigmaI) - lambdaS) / h;
        // R = [[1, 1], [rk, 1]], b = [b1, b2] with b_i = hPhi_k * i! / B(h); solve R rho = b.
        double hPhi2 = hPhi1 / hh - 1, b1 = hPhi2 / bH, hPhi3 = hPhi2 / hh - 1.0 / 2, b2 = hPhi3 * 2 / bH;
        double det = 1 - rk;
        double rho0 = (b1 - b2) / det, rho1 = (b2 - rk * b1) / det;
        for (int i = 0; i < result.Length; i++)
        {
            double d1 = (m1[i] - m0[i]) / rk;
            result[i] = (float)(decay * last[i] - alphaT * hPhi1 * m0[i] - alphaT * bH * (rho0 * d1 + rho1 * (current[i] - m0[i])));
        }
        return result;
    }

    private static (double Alpha, double Sigma) AlphaSigma(double sigma)
    {
        double alpha = 1.0 / Math.Sqrt(sigma * sigma + 1);
        return (alpha, sigma * alpha);
    }
}
