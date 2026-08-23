using System;
using HartsyInference.Audio.Dsp;
using HartsyInference.Audio.Layers;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Vits;

/// <summary>VITS stochastic duration predictor (<c>StochasticDurationPredictor</c>, used when <c>use_sdp=True</c> — Piper's default).</summary>
/// <remarks>Inference (reverse) path: <c>pre</c> Conv1d → <c>convs</c> DDSConv (depthwise-separable dilated conv stack, LayerNorm + GELU) → <c>proj</c> as the flow conditioner; then sample <c>z ~ randn * noise_scale_w</c> of shape <c>[1,2,T]</c> and run it BACKWARD through the flow stack <c>[Flip, (ConvFlow, Flip)…, ElementwiseAffine]</c>. Each <c>ConvFlow</c> builds spline parameters with its own DDSConv and applies an unconstrained piecewise-rational-quadratic-spline transform (10 bins, linear tails) in reverse. The first latent channel is <c>logw</c> (log-durations, one per text frame); the downstream <see cref="VitsLengthRegulator"/> applies <c>exp</c>.</remarks>
public sealed unsafe class VitsStochasticDurationPredictor
{
    private const int NumBins = 10;
    private const float TailBound = 5.0f;
    private const float MinBinWidth = 1e-3f;
    private const float MinBinHeight = 1e-3f;
    private const float MinDerivative = 1e-3f;

    private readonly int _hidden, _filter, _kernel, _ddsLayers, _flowLayers, _nFlows;
    private Tensor? _preW, _preB, _projW, _projB, _condW, _condB;
    private DdsConv? _convs;
    private ConvFlow[]? _flows;
    private Tensor? _eaM, _eaLogs;
    private float[]? _eaScale;     // exp(-logs) per channel; folded into a graph constant by the ONNX export

    public VitsStochasticDurationPredictor(VitsConfig cfg)
    {
        _hidden = cfg.HiddenChannels;
        _filter = cfg.HiddenChannels;     // SDP filter_channels == hidden (192)
        _kernel = 3;
        _ddsLayers = 3;
        _flowLayers = 3;
        _nFlows = cfg.SdpFlows;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "sdp")
    {
        _preW = VitsWeights.Conv(w, $"{prefix}.pre"); _preB = VitsWeights.Bias(w, $"{prefix}.pre");
        _projW = VitsWeights.Conv(w, $"{prefix}.proj"); _projB = VitsWeights.Bias(w, $"{prefix}.proj");
        if (w.ContainsKey($"{prefix}.cond.weight") || w.ContainsKey($"{prefix}.cond.weight_g"))
        { _condW = VitsWeights.Conv(w, $"{prefix}.cond"); _condB = VitsWeights.Bias(w, $"{prefix}.cond"); }
        _convs = new DdsConv(_filter, _kernel, _ddsLayers);
        _convs.LoadWeights(w, $"{prefix}.convs");

        // self.flows = [ElementwiseAffine(2)] + n_flows×[ConvFlow, Flip]; ConvFlow at index 2*j+1.
        // The inference reverse drops the first ConvFlow (index 1) and folds a zero ElementwiseAffine `logs`, so a
        // real Piper export omits `flows.1.*` and `flows.0.logs`; both are optional here (logs absent => identity).
        _eaM = w[$"{prefix}.flows.0.m"];
        _eaLogs = w.TryGetValue($"{prefix}.flows.0.logs", out Tensor? eaLogs) ? eaLogs : null;
        _eaScale = ResolveEaScale(w, prefix);
        _flows = new ConvFlow[_nFlows];
        for (int j = 0; j < _nFlows; j++)
        {
            if (!w.ContainsKey($"{prefix}.flows.{2 * j + 1}.pre.weight")) continue; // unused dropped flow (index 1)
            _flows[j] = new ConvFlow(_filter, _kernel, _flowLayers, NumBins);
            _flows[j].LoadWeights(w, $"{prefix}.flows.{2 * j + 1}");
        }
    }

    // The ElementwiseAffine reverse needs exp(-logs). When the raw `logs` parameter is present, compute it; when the
    // ONNX export folded exp(-logs) into a graph constant (named like ".../flows.0/Exp_output_0"), use it directly.
    private static float[]? ResolveEaScale(IReadOnlyDictionary<string, Tensor> w, string prefix)
    {
        if (w.TryGetValue($"{prefix}.flows.0.logs", out Tensor? logs))
        {
            float* lp = (float*)logs.DataPointer;
            return [MathF.Exp(-lp[0]), MathF.Exp(-lp[1])];
        }
        foreach (KeyValuePair<string, Tensor> kv in w)
        {
            if (kv.Key.Contains("flows.0", StringComparison.Ordinal) && kv.Key.Contains("Exp", StringComparison.Ordinal)
                && kv.Value.ElementCount == 2)
            {
                float* ep = (float*)kv.Value.DataPointer;
                return [ep[0], ep[1]];
            }
        }
        return null; // identity (logs == 0)
    }

    /// <summary>Predicts log-durations from the encoder hidden <c>[1, hidden, T]</c>; when a speaker embedding <paramref name="g"/> <c>[1, gin, 1]</c> and a <c>cond</c> layer are present, <c>cond(g)</c> is added after <c>pre(x)</c> (MeloTTS/Bert-VITS2).</summary>
    public float[] Forward(IBackend backend, Tensor x, int t, float noiseScaleW, int seed, Tensor? g = null)
    {
        if (_preW is null || _convs is null || _flows is null)
            throw new InvalidOperationException("VitsStochasticDurationPredictor weights not loaded.");

        // Conditioner: cond = proj(convs(pre(x) + cond(g))) — feeds every ConvFlow's spline-parameter network.
        Tensor h = new(new TensorShape(1, _filter, t), DType.F32);
        backend.Conv1d(h, x, _preW!, _preB, 1, 0, 0, 1, 1);
        if (g is not null && _condW is not null)
        {
            Tensor cg = new(new TensorShape(1, _filter, 1), DType.F32);
            backend.Conv1d(cg, g, _condW, _condB, 1, 0, 0, 1, 1);
            float* hp = (float*)h.DataPointer, cp = (float*)cg.DataPointer;
            for (int ch = 0; ch < _filter; ch++)
                for (int j = 0; j < t; j++) hp[(long)ch * t + j] += cp[ch];
            cg.Dispose();
        }
        Tensor convOut = _convs.Forward(backend, h, t);
        h.Dispose();
        Tensor cond = new(new TensorShape(1, _filter, t), DType.F32);
        backend.Conv1d(cond, convOut, _projW!, _projB, 1, 0, 0, 1, 1);
        convOut.Dispose();

        // z ~ randn * noise_scale_w, shape [1, 2, T].
        uint rng = DeterministicRng.Seed(seed);
        Tensor z = new(new TensorShape(1, 2, t), DType.F32);
        float* zp = (float*)z.DataPointer;
        for (long n = 0; n < 2L * t; n++) zp[n] = DeterministicRng.NextGaussian(ref rng) * noiseScaleW;

        // Reverse flow order: list(reversed(flows))[:-2] + [flows[-1]] →
        // [Flip, ConvFlow_{n-1}, Flip, …, ConvFlow_1, Flip, ElementwiseAffine].
        // Channel-0 of z holds logw at the end.
        for (int j = _nFlows - 1; j >= 1; j--)
        {
            FlipInPlace(zp, t);
            _flows[j].ReverseInPlace(backend, z, cond, t);
        }
        FlipInPlace(zp, t);
        ElementwiseAffineReverse(zp, t);

        cond.Dispose();
        float[] logw = new float[t];
        new Span<float>(zp, t).CopyTo(logw);     // z0 (first channel) == logw
        z.Dispose();
        return logw;
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] heads = [_preW, _preB, _projW, _projB, _eaM, _eaLogs, _condW, _condB];
        foreach (Tensor? t in heads) if (t is not null) yield return t;
        if (_convs is not null) foreach (Tensor t in _convs.EnumerateWeights()) yield return t;
        if (_flows is not null) foreach (ConvFlow f in _flows) foreach (Tensor t in f.EnumerateWeights()) yield return t;
    }

    /// <summary>Flip reverses channel order; with 2 channels that swaps channel 0 and channel 1 elementwise.</summary>
    private static void FlipInPlace(float* z, int t)
    {
        for (int j = 0; j < t; j++)
        {
            float tmp = z[j];
            z[j] = z[(long)t + j];
            z[(long)t + j] = tmp;
        }
    }

    /// <summary>ElementwiseAffine reverse: <c>x = (x - m) * exp(-logs)</c>, per channel (m/logs shape [2,1]).</summary>
    private void ElementwiseAffineReverse(float* z, int t)
    {
        float* m = (float*)_eaM!.DataPointer;
        for (int c = 0; c < 2; c++)
        {
            float mc = m[c], invs = _eaScale is not null ? _eaScale[c] : 1f;
            for (int j = 0; j < t; j++) z[(long)c * t + j] = (z[(long)c * t + j] - mc) * invs;
        }
    }

    /// <summary>Unconstrained piecewise-rational-quadratic-spline transform (Neural Spline Flow) for a single scalar input: outside <c>[-B, B]</c> the linear tails pass the input through unchanged, inside the (forward or inverse) rational-quadratic map is applied. Public + static so the inverse pass can be unit tested directly.</summary>
    /// <returns>The transformed value (logabsdet is irrelevant at inference and dropped).</returns>
    public static float UnconstrainedRationalQuadraticSpline(float input, ReadOnlySpan<float> unnormWidths,
        ReadOnlySpan<float> unnormHeights, ReadOnlySpan<float> unnormDerivatives, bool inverse,
        float tailBound = TailBound)
    {
        if (input < -tailBound || input > tailBound)
            return input;     // linear tails — identity outside the spline support
        return RationalQuadraticSpline(input, unnormWidths, unnormHeights, unnormDerivatives, inverse,
            -tailBound, tailBound, -tailBound, tailBound);
    }

    /// <summary>The bounded rational-quadratic spline on <c>[left,right]×[bottom,top]</c>: forward maps the input domain to the output range, <paramref name="inverse"/> solves the per-bin quadratic to invert.</summary>
    private static float RationalQuadraticSpline(float input, ReadOnlySpan<float> unnormWidths,
        ReadOnlySpan<float> unnormHeights, ReadOnlySpan<float> unnormDerivatives, bool inverse,
        float left, float right, float bottom, float top)
    {
        int numBins = unnormWidths.Length;

        // widths = softmax(unnorm_widths); cumwidths span [left, right]; clamp min width.
        Span<float> cumWidths = stackalloc float[numBins + 1];
        Span<float> widths = stackalloc float[numBins];
        Softmax(unnormWidths, widths);
        for (int i = 0; i < numBins; i++) widths[i] = MinBinWidth + (1f - MinBinWidth * numBins) * widths[i];
        cumWidths[0] = left;
        float acc = 0f;
        for (int i = 0; i < numBins; i++) { acc += widths[i]; cumWidths[i + 1] = left + (right - left) * acc; }
        cumWidths[numBins] = right;
        for (int i = 0; i < numBins; i++) widths[i] = cumWidths[i + 1] - cumWidths[i];

        // derivatives = min_deriv + softplus(unnorm_derivatives), length numBins+1.
        Span<float> derivatives = stackalloc float[numBins + 1];
        for (int i = 0; i <= numBins; i++) derivatives[i] = MinDerivative + Softplus(unnormDerivatives[i]);

        // heights = softmax(unnorm_heights); cumheights span [bottom, top]; clamp min height.
        Span<float> cumHeights = stackalloc float[numBins + 1];
        Span<float> heights = stackalloc float[numBins];
        Softmax(unnormHeights, heights);
        for (int i = 0; i < numBins; i++) heights[i] = MinBinHeight + (1f - MinBinHeight * numBins) * heights[i];
        cumHeights[0] = bottom;
        acc = 0f;
        for (int i = 0; i < numBins; i++) { acc += heights[i]; cumHeights[i + 1] = bottom + (top - bottom) * acc; }
        cumHeights[numBins] = top;
        for (int i = 0; i < numBins; i++) heights[i] = cumHeights[i + 1] - cumHeights[i];

        // Locate the bin: inverse searches the output (cumheights), forward searches the input (cumwidths).
        int bin = inverse ? SearchSorted(cumHeights, input, numBins) : SearchSorted(cumWidths, input, numBins);

        float inputCumWidths = cumWidths[bin];
        float inputBinWidth = widths[bin];
        float inputCumHeights = cumHeights[bin];
        float deltaHeight = heights[bin];
        float inputDerivative = derivatives[bin];
        float inputDerivativePlusOne = derivatives[bin + 1];
        float inputDeltaSlope = deltaHeight / inputBinWidth;     // s_k = (y_{k+1}-y_k)/(x_{k+1}-x_k)

        if (inverse)
        {
            // Solve the quadratic a·ξ² + b·ξ + c = 0 for the bin position ξ (Durkan et al. NSF, eq. for inverse).
            float a = deltaHeight * (inputDeltaSlope - inputDerivative)
                + (input - inputCumHeights) * (inputDerivative + inputDerivativePlusOne - 2f * inputDeltaSlope);
            float b = deltaHeight * inputDerivative
                - (input - inputCumHeights) * (inputDerivative + inputDerivativePlusOne - 2f * inputDeltaSlope);
            float c = -inputDeltaSlope * (input - inputCumHeights);

            float discriminant = b * b - 4f * a * c;
            if (discriminant < 0f) discriminant = 0f;
            float root = (2f * c) / (-b - MathF.Sqrt(discriminant));
            return root * inputBinWidth + inputCumWidths;
        }
        else
        {
            float theta = (input - inputCumWidths) / inputBinWidth;
            float thetaOneMinusTheta = theta * (1f - theta);
            float numerator = deltaHeight * (inputDeltaSlope * theta * theta + inputDerivative * thetaOneMinusTheta);
            float denominator = inputDeltaSlope
                + (inputDerivative + inputDerivativePlusOne - 2f * inputDeltaSlope) * thetaOneMinusTheta;
            return inputCumHeights + numerator / denominator;
        }
    }

    private static int SearchSorted(ReadOnlySpan<float> boundaries, float value, int numBins)
    {
        // bin index = count of bin-edges (excluding the first) <= value, clamped to [0, numBins-1].
        int bin = 0;
        for (int i = 1; i <= numBins; i++) { if (boundaries[i] <= value) bin = i; else break; }
        if (bin > numBins - 1) bin = numBins - 1;
        return bin;
    }

    private static void Softmax(ReadOnlySpan<float> src, Span<float> dst)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < src.Length; i++) if (src[i] > max) max = src[i];
        float sum = 0f;
        for (int i = 0; i < src.Length; i++) { float e = MathF.Exp(src[i] - max); dst[i] = e; sum += e; }
        float inv = 1f / sum;
        for (int i = 0; i < dst.Length; i++) dst[i] *= inv;
    }

    private static float Softplus(float x)
    {
        // Numerically stable log(1 + e^x): for large x it saturates to x, avoiding exp overflow.
        return x > 20f ? x : MathF.Log(1f + MathF.Exp(x));
    }

    /// <summary>Depthwise-separable dilated conv stack (<c>DDSConv</c>): per layer a depthwise conv (groups=channels, dilation=kernel^i) → LayerNorm → GELU → pointwise 1×1 conv → LayerNorm → GELU, added as a residual; LayerNorm is over the channel dim.</summary>
    private sealed class DdsConv
    {
        private readonly int _ch, _kernel, _layers;
        private readonly Tensor?[] _sepW, _sepB, _oneW, _oneB, _n1G, _n1B, _n2G, _n2B;

        public DdsConv(int channels, int kernel, int layers)
        {
            _ch = channels; _kernel = kernel; _layers = layers;
            _sepW = new Tensor?[layers]; _sepB = new Tensor?[layers];
            _oneW = new Tensor?[layers]; _oneB = new Tensor?[layers];
            _n1G = new Tensor?[layers]; _n1B = new Tensor?[layers];
            _n2G = new Tensor?[layers]; _n2B = new Tensor?[layers];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            for (int i = 0; i < _layers; i++)
            {
                _sepW[i] = VitsWeights.Conv(w, $"{prefix}.convs_sep.{i}"); _sepB[i] = VitsWeights.Bias(w, $"{prefix}.convs_sep.{i}");
                _oneW[i] = VitsWeights.Conv(w, $"{prefix}.convs_1x1.{i}"); _oneB[i] = VitsWeights.Bias(w, $"{prefix}.convs_1x1.{i}");
                _n1G[i] = w[$"{prefix}.norms_1.{i}.gamma"]; _n1B[i] = w[$"{prefix}.norms_1.{i}.beta"];
                _n2G[i] = w[$"{prefix}.norms_2.{i}.gamma"]; _n2B[i] = w[$"{prefix}.norms_2.{i}.beta"];
            }
        }

        public unsafe Tensor Forward(IBackend backend, Tensor x, int t)
        {
            Tensor cur = Clone(x);
            for (int i = 0; i < _layers; i++)
            {
                int dilation = Pow(_kernel, i);
                int pad = dilation * (_kernel - 1) / 2;
                Tensor sep = new(new TensorShape(1, _ch, t), DType.F32);
                backend.Conv1d(sep, cur, _sepW[i]!, _sepB[i], 1, pad, pad, dilation, _ch);     // depthwise
                ChannelLnGeluInPlace(sep, _n1G[i]!, _n1B[i]!, _ch, t);
                Tensor one = new(new TensorShape(1, _ch, t), DType.F32);
                backend.Conv1d(one, sep, _oneW[i]!, _oneB[i], 1, 0, 0, 1, 1);     // pointwise
                sep.Dispose();
                ChannelLnGeluInPlace(one, _n2G[i]!, _n2B[i]!, _ch, t);
                AddInPlace(cur, one);     // residual: x = x + conv(x)
                one.Dispose();
            }
            return cur;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[][] groups = [_sepW, _sepB, _oneW, _oneB, _n1G, _n1B, _n2G, _n2B];
            foreach (Tensor?[] g in groups) foreach (Tensor? t in g) if (t is not null) yield return t;
        }

        private static unsafe void ChannelLnGeluInPlace(Tensor x, Tensor gamma, Tensor beta, int ch, int t)
        {
            float* xp = (float*)x.DataPointer, g = (float*)gamma.DataPointer, be = (float*)beta.DataPointer;
            for (int j = 0; j < t; j++)
            {
                double mean = 0;
                for (int c = 0; c < ch; c++) mean += xp[(long)c * t + j];
                mean /= ch;
                double var = 0;
                for (int c = 0; c < ch; c++) { double d = xp[(long)c * t + j] - mean; var += d * d; }
                var /= ch;
                float inv = (float)(1.0 / Math.Sqrt(var + 1e-5));
                for (int c = 0; c < ch; c++)
                {
                    float v = ((xp[(long)c * t + j] - (float)mean) * inv) * g[c] + be[c];
                    // Exact erf GELU, not the tanh approximation — the latter biases the spline parameters
                    // enough to shift durations.
                    xp[(long)c * t + j] = Activations.ErfGelu(v);
                }
            }
        }

        private static unsafe void AddInPlace(Tensor dst, Tensor src)
        {
            float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
            for (long n = 0; n < dst.ElementCount; n++) d[n] += s[n];
        }

        private static int Pow(int b, int e) { int r = 1; for (int i = 0; i < e; i++) r *= b; return r; }

        private static unsafe Tensor Clone(Tensor t)
        {
            Tensor c = new(t.Shape, DType.F32);
            Buffer.MemoryCopy((void*)t.DataPointer, (void*)c.DataPointer, t.ElementCount * 4, t.ElementCount * 4);
            return c;
        }
    }

    /// <summary>A single <c>ConvFlow</c> coupling layer: a DDSConv-backed network predicts per-frame spline parameters (widths/heights/derivatives) from one half of the channels, and the rational-quadratic spline transforms the other half; here the channels are split <c>1|1</c> (the SDP operates on 2 channels).</summary>
    private sealed class ConvFlow
    {
        private readonly int _filter, _kernel, _layers, _numBins;
        private Tensor? _preW, _preB, _projW, _projB;
        private DdsConv? _convs;

        public ConvFlow(int filter, int kernel, int layers, int numBins)
        {
            _filter = filter; _kernel = kernel; _layers = layers; _numBins = numBins;
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix)
        {
            _preW = VitsWeights.Conv(w, $"{prefix}.pre"); _preB = VitsWeights.Bias(w, $"{prefix}.pre");
            _projW = VitsWeights.Conv(w, $"{prefix}.proj"); _projB = VitsWeights.Bias(w, $"{prefix}.proj");
            _convs = new DdsConv(_filter, _kernel, _layers);
            _convs.LoadWeights(w, $"{prefix}.convs");
        }

        /// <summary>Reverse coupling on <paramref name="z"/> <c>[1,2,T]</c>: channel 0 (x0) conditions the spline applied (inverse) to channel 1 (x1), in place; channel 0 is unchanged.</summary>
        public unsafe void ReverseInPlace(IBackend backend, Tensor z, Tensor cond, int t)
        {
            float* zp = (float*)z.DataPointer;

            // h = pre(x0) + cond (broadcast); the SDP conditions every ConvFlow on the shared `proj` output.
            Tensor x0 = new(new TensorShape(1, 1, t), DType.F32);
            Buffer.MemoryCopy(zp, (void*)x0.DataPointer, (long)t * 4, (long)t * 4);
            Tensor h = new(new TensorShape(1, _filter, t), DType.F32);
            backend.Conv1d(h, x0, _preW!, _preB, 1, 0, 0, 1, 1);
            x0.Dispose();
            AddInPlace(h, cond);
            Tensor convOut = _convs!.Forward(backend, h, t);
            h.Dispose();

            // proj → [1, numBins*3 - 1, T]: widths/heights scaled by 1/sqrt(filter), raw derivatives.
            int paramCount = _numBins * 3 - 1;
            Tensor p = new(new TensorShape(1, paramCount, t), DType.F32);
            backend.Conv1d(p, convOut, _projW!, _projB, 1, 0, 0, 1, 1);
            convOut.Dispose();

            float* pp = (float*)p.DataPointer;
            float invSqrtF = 1f / MathF.Sqrt(_filter);
            Span<float> uw = stackalloc float[_numBins];
            Span<float> uh = stackalloc float[_numBins];
            Span<float> ud = stackalloc float[_numBins + 1];
            for (int j = 0; j < t; j++)
            {
                for (int b = 0; b < _numBins; b++) uw[b] = pp[(long)b * t + j] * invSqrtF;
                for (int b = 0; b < _numBins; b++) uh[b] = pp[(long)(_numBins + b) * t + j] * invSqrtF;
                // Linear tails: pad the derivative array with a boundary constant at both ends so the edge
                // slopes are exactly 1; the proj emits only the numBins-1 interior derivatives.
                ud[0] = Constants;
                ud[_numBins] = Constants;
                for (int b = 0; b < _numBins - 1; b++) ud[b + 1] = pp[(long)(2 * _numBins + b) * t + j];
                float x1 = zp[(long)t + j];
                zp[(long)t + j] = UnconstrainedRationalQuadraticSpline(x1, uw, uh, ud, inverse: true);
            }
            p.Dispose();
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] heads = [_preW, _preB, _projW, _projB];
            foreach (Tensor? t in heads) if (t is not null) yield return t;
            if (_convs is not null) foreach (Tensor t in _convs.EnumerateWeights()) yield return t;
        }

        // Linear-tails boundary constant: log(exp(1 - min_derivative) - 1) with min_derivative=1e-3, so after
        // the transform's `min_derivative + softplus(.)` the edge derivative equals exactly 1.
        private const float Constants = 0.5399810f;

        private static unsafe void AddInPlace(Tensor dst, Tensor src)
        {
            float* d = (float*)dst.DataPointer, s = (float*)src.DataPointer;
            for (long n = 0; n < dst.ElementCount; n++) d[n] += s[n];
        }
    }
}
