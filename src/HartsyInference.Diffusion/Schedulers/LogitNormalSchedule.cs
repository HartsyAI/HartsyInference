namespace HartsyInference.Diffusion.Schedulers;

/// <summary>Logit-normal flow-matching schedule used by Ideogram 4 (<c>ideogram-oss/ideogram4</c> <c>scheduler.py</c>). Maps a uniform grid value <c>t ∈ [0,1]</c> to a noise level via <c>1 − sigmoid(mean + std · Φ⁻¹(t))</c>, clamped to a log-SNR window. This is NOT the SD3-style time-shift used by <see cref="FlowMatchEulerDiscreteScheduler"/>; Ideogram drives its own Euler loop in the pipeline (<c>z += v·(s−t)</c>), so this is a plain stateless warp rather than an <c>IScheduler</c>.
///
/// <para>The <c>mean</c> auto-adjusts for resolution: <c>mean = knownMean + 0.5·ln(numPixels / knownPixels)</c> with <c>knownResolution = 512×512</c>. Construct via <see cref="ForResolution"/>.</para></summary>
public sealed class LogitNormalSchedule
{
    private readonly double _mean;
    private readonly double _std;
    private readonly double _tMin;
    private readonly double _tMax;

    /// <summary>Creates a schedule with an explicit mean/std and log-SNR clamp window.</summary>
    /// <param name="mean">Logit-normal mean (already resolution-adjusted if applicable).</param>
    /// <param name="std">Logit-normal standard deviation.</param>
    /// <param name="logsnrMin">Lower log-SNR bound (default −15) → defines <c>tMax</c>.</param>
    /// <param name="logsnrMax">Upper log-SNR bound (default 18) → defines <c>tMin</c>.</param>
    public LogitNormalSchedule(double mean, double std, double logsnrMin = -15.0, double logsnrMax = 18.0)
    {
        _mean = mean;
        _std = std;
        _tMin = 1.0 / (1.0 + Math.Exp(0.5 * logsnrMax));
        _tMax = 1.0 / (1.0 + Math.Exp(0.5 * logsnrMin));
    }

    /// <summary>Resolution-aware schedule (upstream <c>get_schedule_for_resolution</c>): <c>mean = knownMean + 0.5·ln(H·W / (512·512))</c>.</summary>
    public static LogitNormalSchedule ForResolution(int height, int width, double knownMean, double std,
        int knownHeight = 512, int knownWidth = 512)
    {
        double numPixels = (double)height * width;
        double knownPixels = (double)knownHeight * knownWidth;
        double mean = knownMean + 0.5 * Math.Log(numPixels / knownPixels);
        return new LogitNormalSchedule(mean, std);
    }

    /// <summary>Linear step grid <c>linspace(0, 1, numSteps + 1)</c> (upstream <c>make_step_intervals</c>).</summary>
    public static float[] MakeStepIntervals(int numSteps)
    {
        float[] grid = new float[numSteps + 1];
        for (int i = 0; i <= numSteps; i++)
            grid[i] = (float)((double)i / numSteps);
        return grid;
    }

    /// <summary>Maps a uniform grid value to a clamped noise level: <c>clamp(1 − sigmoid(mean + std·Φ⁻¹(t)), tMin, tMax)</c>.</summary>
    public float Map(float gridValue)
    {
        double z = Ndtri(gridValue);
        double y = _mean + _std * z;
        double t = 1.0 - Sigmoid(y);
        if (t < _tMin) t = _tMin;
        if (t > _tMax) t = _tMax;
        return (float)t;
    }

    private static double Sigmoid(double x)
    {
        // Numerically stable expit.
        if (x >= 0.0)
        {
            double e = Math.Exp(-x);
            return 1.0 / (1.0 + e);
        }
        double ex = Math.Exp(x);
        return ex / (1.0 + ex);
    }

    /// <summary>Inverse standard-normal CDF (probit / <c>Φ⁻¹</c>), Peter Acklam's rational approximation (~1.15e-9 relative error). Returns ±∞ at the boundaries so the caller's clamp handles <c>t=0</c>/<c>t=1</c>.</summary>
    private static double Ndtri(double p)
    {
        if (p <= 0.0) return double.NegativeInfinity;
        if (p >= 1.0) return double.PositiveInfinity;

        // Coefficients for the rational approximation.
        const double a1 = -3.969683028665376e+01, a2 = 2.209460984245205e+02;
        const double a3 = -2.759285104469687e+02, a4 = 1.383577518672690e+02;
        const double a5 = -3.066479806614716e+01, a6 = 2.506628277459239e+00;
        const double b1 = -5.447609879822406e+01, b2 = 1.615858368580409e+02;
        const double b3 = -1.556989798598866e+02, b4 = 6.680131188771972e+01;
        const double b5 = -1.328068155288572e+01;
        const double c1 = -7.784894002430293e-03, c2 = -3.223964580411365e-01;
        const double c3 = -2.400758277161838e+00, c4 = -2.549732539343734e+00;
        const double c5 = 4.374664141464968e+00, c6 = 2.938163982698783e+00;
        const double d1 = 7.784695709041462e-03, d2 = 3.224671290700398e-01;
        const double d3 = 2.445134137142996e+00, d4 = 3.754408661907416e+00;
        const double pLow = 0.02425;
        const double pHigh = 1.0 - pLow;

        double q, r, x;
        if (p < pLow)
        {
            q = Math.Sqrt(-2.0 * Math.Log(p));
            x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
        else if (p <= pHigh)
        {
            q = p - 0.5;
            r = q * q;
            x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
                (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        }
        else
        {
            q = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
            x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
                ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }

        // One Halley refinement step against the true CDF (erfc) for full double precision.
        double e = 0.5 * Erfc(-x / Math.Sqrt(2.0)) - p;
        double u = e * Math.Sqrt(2.0 * Math.PI) * Math.Exp(x * x / 2.0);
        x -= u / (1.0 + x * u / 2.0);
        return x;
    }

    /// <summary>Complementary error function (Numerical Recipes <c>erfcc</c>, ~1.2e-7 fractional error) — used only for the Halley refinement of <see cref="Ndtri"/>.</summary>
    private static double Erfc(double x)
    {
        double z = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.5 * z);
        double ans = t * Math.Exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
            t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0.0 ? ans : 2.0 - ans;
    }
}
