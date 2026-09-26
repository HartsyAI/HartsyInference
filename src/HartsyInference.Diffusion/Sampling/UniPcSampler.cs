using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>ComfyUI's <c>uni_pc</c> (B(h) = h) and <c>uni_pc_bh2</c> (B(h) = e^h − 1): the multistep UniPC
/// predictor–corrector in data-prediction form, order up to 3.</summary>
/// <remarks>ComfyUI works in VP-scaled coordinates; in VE coordinates the same update reduces to the coefficients
/// below, and its data prediction is exactly the denoised estimate. The terminal sigma is replaced by 0.001, as
/// ComfyUI does, and flow latents are then divided by <c>1 − 0.001</c> like its <c>inverse_noise_scaling</c>.</remarks>
public sealed class UniPcSampler : SamplerBase
{
    private const int MaxOrder = 3;
    private const float TerminalSigma = 0.001f;
    private readonly bool _bh2;
    private readonly Tensor?[] _models = new Tensor?[MaxOrder];
    private readonly double[] _times = new double[MaxOrder];
    private int _count;
    private int _order;
    private int _steps;

    /// <summary>Creates the sampler; <paramref name="bh2"/> selects <c>uni_pc_bh2</c>.</summary>
    public UniPcSampler(float[] sigmas, bool bh2, SamplerOptions? options = null)
        : base(sigmas, 0, options) => _bh2 = bh2;

    /// <inheritdoc/>
    public override string Name => _bh2 ? "uni_pc_bh2" : "uni_pc";

    /// <inheritdoc/>
    protected override void OnReset()
    {
        for (int k = 0; k < _models.Length; k++)
        {
            Release(ref _models[k]);
        }
        _count = 0;
    }

    /// <inheritdoc/>
    protected override void OnBegin()
    {
        _steps = LocalLength - 1;
        _order = Math.Min(MaxOrder, LocalLength - 2);
    }

    /// <inheritdoc/>
    protected override void StepLocal(IBackend backend, Tensor z, IDenoisePredictor predictor, int i, int stepIndex)
    {
        if (i == 0)
        {
            Push(backend, Denoise(backend, predictor, z, Sigma(0), stepIndex), Sigma(0));
            if (_order <= 0)
            {
                return;
            }
        }
        int target = i + 1;
        float sigmaTarget = target == _steps && Sigma(target) == 0f ? TerminalSigma : Sigma(target);
        int order;
        bool useCorrector;
        bool append;
        if (target < _order)
        {
            order = target;
            useCorrector = true;
            append = true;
        }
        else
        {
            order = Math.Min(_order, _steps + 1 - target);
            useCorrector = target != _steps;
            append = false;
        }
        Tensor? modelT = Update(backend, z, predictor, sigmaTarget, order, useCorrector, stepIndex);
        if (append)
        {
            Push(backend, modelT!, sigmaTarget);
        }
        else
        {
            Shift(backend, modelT, sigmaTarget);
        }
        if (target == _steps && IsFlow && Sigma(target) == 0f)
        {
            SamplerOps.ScaleInPlace(backend, z, 1.0f / (1.0f - TerminalSigma));
        }
    }

    /// <summary>One <c>multistep_uni_pc_bh_update</c>; returns the corrector's model evaluation, if any.</summary>
    private Tensor? Update(IBackend backend, Tensor z, IDenoisePredictor predictor, float sigmaT, int order, bool useCorrector, int stepIndex)
    {
        double sigmaPrev = _times[_count - 1];
        Tensor model0 = _models[_count - 1]!;
        double lambdaPrev = -Math.Log(sigmaPrev);
        double h = -Math.Log(sigmaT) - lambdaPrev;
        Span<double> rks = stackalloc double[MaxOrder];
        for (int k = 1; k < order; k++)
        {
            rks[k - 1] = (-Math.Log(_times[_count - 1 - k]) - lambdaPrev) / h;
        }
        rks[order - 1] = 1.0;

        double hh = -h;
        double hPhi1 = SamplerMath.Expm1(hh);
        double hPhiK = (hPhi1 / hh) - 1.0;
        double factorial = 1.0;
        double bH = _bh2 ? SamplerMath.Expm1(hh) : hh;
        Span<double> rMatrix = stackalloc double[MaxOrder * MaxOrder];
        Span<double> bVec = stackalloc double[MaxOrder];
        for (int p = 1; p <= order; p++)
        {
            for (int j = 0; j < order; j++)
            {
                rMatrix[((p - 1) * order) + j] = Math.Pow(rks[j], p - 1);
            }
            bVec[p - 1] = hPhiK * factorial / bH;
            factorial *= p + 1;
            hPhiK = (hPhiK / hh) - (1.0 / factorial);
        }

        // Predictor weights on the history differences D1s[k] = (m_{-(k+1)} − m0)/rk.
        Span<double> rhosP = stackalloc double[MaxOrder];
        int historyTerms = order - 1;
        if (historyTerms > 0)
        {
            if (order == 2)
            {
                rhosP[0] = 0.5;
            }
            else
            {
                SolveSquare(rMatrix, order, historyTerms, bVec, rhosP);
            }
        }

        using Tensor baseState = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, baseState, z, model0, (float)(sigmaT / sigmaPrev), (float)-hPhi1);
        backend.Scale(z, baseState, 1.0f);
        AddHistoryDifferences(backend, z, model0, rhosP, historyTerms, -bH, rks);
        if (!useCorrector)
        {
            return null;
        }

        Span<double> rhosC = stackalloc double[MaxOrder];
        if (order == 1)
        {
            rhosC[0] = 0.5;
        }
        else
        {
            SolveSquare(rMatrix, order, order, bVec, rhosC);
        }
        Tensor modelT = Denoise(backend, predictor, z, sigmaT, stepIndex);
        backend.Scale(z, baseState, 1.0f);
        AddHistoryDifferences(backend, z, model0, rhosC, historyTerms, -bH, rks);
        double last = -bH * rhosC[order - 1];
        SamplerOps.MixInto(backend, z, modelT, model0, 1.0f, (float)last, (float)-last);
        return modelT;
    }

    /// <summary><c>target += scale·Σ rho_k·(m_{-(k+1)} − m0)/r_k</c>.</summary>
    private void AddHistoryDifferences(IBackend backend, Tensor target, Tensor model0, ReadOnlySpan<double> rhos, int terms,
        double scale, ReadOnlySpan<double> rks)
    {
        double model0Weight = 0.0;
        for (int k = 0; k < terms; k++)
        {
            double w = scale * rhos[k] / rks[k];
            SamplerOps.MixInto(backend, target, _models[_count - 2 - k]!, 1.0f, (float)w);
            model0Weight -= w;
        }
        if (terms > 0)
        {
            SamplerOps.MixInto(backend, target, model0, 1.0f, (float)model0Weight);
        }
    }

    private void Push(IBackend backend, Tensor model, double sigma)
    {
        Keep(backend, ref _models[_count], model);
        _times[_count] = sigma;
        _count++;
    }

    /// <summary>Drops the oldest entry and appends <paramref name="model"/> (which may be null at the final step).</summary>
    private void Shift(IBackend backend, Tensor? model, double sigma)
    {
        Release(ref _models[0]);
        for (int k = 0; k < _count - 1; k++)
        {
            _models[k] = _models[k + 1];
            _times[k] = _times[k + 1];
        }
        _models[_count - 1] = null;
        _times[_count - 1] = sigma;
        if (model is not null)
        {
            Keep(backend, ref _models[_count - 1], model);
        }
    }

    /// <summary>Solves the leading <paramref name="n"/>×<paramref name="n"/> block of <paramref name="matrix"/>.</summary>
    private static void SolveSquare(ReadOnlySpan<double> matrix, int stride, int n, ReadOnlySpan<double> rhs, Span<double> result)
    {
        Span<double> a = stackalloc double[MaxOrder * MaxOrder];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < n; c++)
            {
                a[(r * n) + c] = matrix[(r * stride) + c];
            }
            result[r] = rhs[r];
        }
        for (int col = 0; col < n; col++)
        {
            int pivot = col;
            for (int r = col + 1; r < n; r++)
            {
                if (Math.Abs(a[(r * n) + col]) > Math.Abs(a[(pivot * n) + col]))
                {
                    pivot = r;
                }
            }
            if (pivot != col)
            {
                for (int c = 0; c < n; c++)
                {
                    (a[(col * n) + c], a[(pivot * n) + c]) = (a[(pivot * n) + c], a[(col * n) + c]);
                }
                (result[col], result[pivot]) = (result[pivot], result[col]);
            }
            for (int r = col + 1; r < n; r++)
            {
                double f = a[(r * n) + col] / a[(col * n) + col];
                for (int c = col; c < n; c++)
                {
                    a[(r * n) + c] -= f * a[(col * n) + c];
                }
                result[r] -= f * result[col];
            }
        }
        for (int r = n - 1; r >= 0; r--)
        {
            double sum = result[r];
            for (int c = r + 1; c < n; c++)
            {
                sum -= a[(r * n) + c] * result[c];
            }
            result[r] = sum / a[(r * n) + r];
        }
    }
}
