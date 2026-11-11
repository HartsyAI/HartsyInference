namespace HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

/// <summary>The discrete-time noise schedule Paint was trained with (<c>ByteDance/sd2.1-base-zsnr</c>, diffusers 0.30 <c>DDIMScheduler</c> config): 1000 steps, scaled-linear betas
/// from 0.00085 to 0.012, rescaled to zero terminal SNR (<c>rescale_zero_terminal_snr</c>, last cumulative alpha set to 2^-24), v-prediction.</summary>
public sealed class HunyuanPaintSchedule
{
    /// <summary>Training steps.</summary>
    public const int TrainSteps = 1000;

    /// <summary>Creates the schedule table.</summary>
    public HunyuanPaintSchedule(double betaStart = 0.00085, double betaEnd = 0.012, bool rescaleZeroTerminalSnr = true)
    {
        double[] betas = new double[TrainSteps];
        double s0 = Math.Sqrt(betaStart), s1 = Math.Sqrt(betaEnd);
        for (int i = 0; i < TrainSteps; i++)
        {
            double b = s0 + (s1 - s0) * i / (TrainSteps - 1);
            betas[i] = b * b;
        }
        if (rescaleZeroTerminalSnr)
            betas = RescaleZeroTerminalSnr(betas);
        AlphasCumprod = new double[TrainSteps];
        double product = 1.0;
        for (int i = 0; i < TrainSteps; i++)
        {
            product *= 1.0 - betas[i];
            AlphasCumprod[i] = product;
        }
        if (rescaleZeroTerminalSnr)
            AlphasCumprod[^1] = Math.Pow(2, -24);
        Sigmas = AlphasCumprod.Select(a => Math.Sqrt((1 - a) / a)).ToArray();
    }

    /// <summary>Cumulative product of (1 - beta) per training step.</summary>
    public double[] AlphasCumprod { get; }

    /// <summary><c>sqrt((1 - alphaBar) / alphaBar)</c> per training step.</summary>
    public double[] Sigmas { get; }

    /// <summary>Timesteps of <c>timestep_spacing="trailing"</c>: <c>round(arange(1000, 0, -1000 / steps)) - 1</c>, highest first.</summary>
    public static int[] TrailingTimesteps(int steps)
    {
        if (steps <= 0 || steps > TrainSteps)
            throw new ArgumentOutOfRangeException(nameof(steps));
        double ratio = (double)TrainSteps / steps;
        int count = (int)Math.Ceiling(TrainSteps / ratio);
        int[] timesteps = new int[count];
        for (int i = 0; i < count; i++)
            timesteps[i] = (int)Math.Round(TrainSteps - i * ratio, MidpointRounding.ToEven) - 1;
        return timesteps;
    }

    /// <summary>Sigmas at the given integer timesteps followed by the terminal value (<c>np.interp</c> at integers is the table entry).</summary>
    public double[] SigmasAt(int[] timesteps, double terminal)
    {
        double[] sigmas = new double[timesteps.Length + 1];
        for (int i = 0; i < timesteps.Length; i++)
            sigmas[i] = Sigmas[timesteps[i]];
        sigmas[^1] = terminal;
        return sigmas;
    }

    private static double[] RescaleZeroTerminalSnr(double[] betas)
    {
        int n = betas.Length;
        double[] sqrtBar = new double[n];
        double product = 1.0;
        for (int i = 0; i < n; i++)
        {
            product *= 1.0 - betas[i];
            sqrtBar[i] = Math.Sqrt(product);
        }
        double first = sqrtBar[0], last = sqrtBar[^1];
        for (int i = 0; i < n; i++)
            sqrtBar[i] = (sqrtBar[i] - last) * (first / (first - last));
        double[] bar = sqrtBar.Select(v => v * v).ToArray();
        double[] rescaled = new double[n];
        rescaled[0] = 1.0 - bar[0];
        for (int i = 1; i < n; i++)
            rescaled[i] = 1.0 - bar[i] / bar[i - 1];
        return rescaled;
    }
}
