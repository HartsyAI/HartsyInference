using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>The mask and IoU heads of P3-SAM (<c>Hunyuan3D-Part/P3-SAM/model.py</c>), plain MLPs of <c>Linear, GELU, Linear, GELU, Linear</c> over per-point
/// features: stage 1 (<c>seg_mlp_1..3</c>) reads <c>[feature, point, prompt]</c>; stage 2 (<c>seg_s2_mlp_g</c>, <c>seg_s2_mlp_1..3</c>) adds the stage-1
/// logits and a max-pooled global vector; <c>iou_mlp</c> and <c>iou_mlp_out</c> predict the IoU of the three masks.
/// <para>The first layer of every head is linear in a concatenation whose per-point part (feature and point) does not depend on the prompt, so that part
/// is computed once per mesh (<see cref="Prepare"/>) and only the prompt and logit terms are added per prompt. The result is the same function as
/// upstream's concatenate-then-MLP, evaluated in a different order.</para></summary>
public sealed class P3SamHeads
{
    private const int PointWidth = 3, PromptWidth = 3, Masks = 3;
    private readonly Mlp _seg1, _seg2, _seg3, _g, _s21, _s22, _s23, _iou, _iouOut;
    private readonly int _featureDim, _hidden1, _hidden2;

    /// <summary>Reads the heads from <paramref name="weights"/> (keys such as <c>seg_mlp_1.0.weight</c>, the names in <c>p3sam.safetensors</c>).</summary>
    public P3SamHeads(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _seg1 = new Mlp(weights, "seg_mlp_1");
        _seg2 = new Mlp(weights, "seg_mlp_2");
        _seg3 = new Mlp(weights, "seg_mlp_3");
        _g = new Mlp(weights, "seg_s2_mlp_g");
        _s21 = new Mlp(weights, "seg_s2_mlp_1");
        _s22 = new Mlp(weights, "seg_s2_mlp_2");
        _s23 = new Mlp(weights, "seg_s2_mlp_3");
        _iou = new Mlp(weights, "iou_mlp");
        _iouOut = new Mlp(weights, "iou_mlp_out");
        _featureDim = _seg1.In - PointWidth - PromptWidth;
        _hidden1 = _seg1.Hidden;
        _hidden2 = _g.Out;
        Expect(_g.In == _featureDim + PointWidth + PromptWidth + Masks, "seg_s2_mlp_g input");
        foreach (Mlp s2 in new[] { _s21, _s22, _s23 })
            Expect(s2.In == _hidden2 + _featureDim + PointWidth + PromptWidth + Masks && s2.Out == 1, "seg_s2_mlp input");
        Expect(_iou.In == _hidden2 + _featureDim + PointWidth + PromptWidth + Masks && _iouOut.Out == Masks, "iou head input");
    }

    /// <summary>Per-point feature width (512 for P3-SAM).</summary>
    public int FeatureDim => _featureDim;

    /// <summary>The prompt-independent first-layer terms of every head for one mesh.</summary>
    public sealed class Prepared
    {
        internal float[] Seg1 = [], Seg2 = [], Seg3 = [], G = [], S21 = [], S22 = [], S23 = [], Iou = [];
        internal float[] Points = [];
        internal int Count;
    }

    /// <summary>The three mask probabilities per point (index mask, then point) and the three predicted IoUs of one prompt.</summary>
    public sealed record Result(float[][] Masks, float[] Iou);

    /// <summary>Computes the terms that do not depend on the prompt for <paramref name="count"/> points: <paramref name="features"/> <c>[count, FeatureDim]</c> and
    /// <paramref name="points"/> <c>[count, 3]</c>.</summary>
    public Prepared Prepare(IBackend backend, float[] features, float[] points, int count)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (features.Length != (long)count * _featureDim || points.Length != count * PointWidth)
            throw new ArgumentException("Features and points must match the point count.");
        Prepared p = new() { Count = count, Points = points };
        int d = _featureDim;
        p.Seg1 = FirstLayer(backend, _seg1, features, points, count, 0, d);
        p.Seg2 = FirstLayer(backend, _seg2, features, points, count, 0, d);
        p.Seg3 = FirstLayer(backend, _seg3, features, points, count, 0, d);
        p.G = FirstLayer(backend, _g, features, points, count, 0, d);
        // Stage-2 and IoU inputs are [global, feature, point, prompt, logits]: the feature block starts after the global vector.
        p.S21 = FirstLayer(backend, _s21, features, points, count, _hidden2, d);
        p.S22 = FirstLayer(backend, _s22, features, points, count, _hidden2, d);
        p.S23 = FirstLayer(backend, _s23, features, points, count, _hidden2, d);
        p.Iou = FirstLayer(backend, _iou, features, points, count, _hidden2, d);
        return p;
    }

    /// <summary>Predicts the three masks and their IoUs for one prompt point.</summary>
    public Result Segment(IBackend backend, Prepared p, ReadOnlySpan<float> prompt)
    {
        if (prompt.Length != PromptWidth)
            throw new ArgumentException("A prompt is one 3D point.", nameof(prompt));
        int n = p.Count;
        // Stage 1: three heads, each [feature, point, prompt] -> one logit per point.
        float[] stage1 = new float[n * Masks];
        Head(backend, _seg1, p.Seg1, n, Prompt(_seg1, _featureDim + PointWidth, prompt), stage1, 0);
        Head(backend, _seg2, p.Seg2, n, Prompt(_seg2, _featureDim + PointWidth, prompt), stage1, 1);
        Head(backend, _seg3, p.Seg3, n, Prompt(_seg3, _featureDim + PointWidth, prompt), stage1, 2);

        // Stage 2: the global vector is the point-wise max of the g-MLP over [feature, point, prompt, stage-1 logits].
        float[] g = AddTerms(p.G, n, _g.Hidden, Prompt(_g, _featureDim + PointWidth, prompt), stage1, _g, _featureDim + PointWidth + PromptWidth);
        float[] globalVector = MaxRows(Tail(backend, _g, g, n, activateLast: false), n, _hidden2);

        float[] stage2 = new float[n * Masks];
        float[][] s2Prompt = [PromptTerm(_s21, globalVector, prompt), PromptTerm(_s22, globalVector, prompt), PromptTerm(_s23, globalVector, prompt)];
        Mlp[] s2 = [_s21, _s22, _s23];
        float[][] s2Base = [p.S21, p.S22, p.S23];
        for (int h = 0; h < Masks; h++)
        {
            float[] hidden = AddTerms(s2Base[h], n, s2[h].Hidden, s2Prompt[h], stage1, s2[h], _hidden2 + _featureDim + PointWidth + PromptWidth);
            Logits(backend, s2[h], hidden, n, stage2, h);
        }

        // IoU head over [global, feature, point, prompt, stage-2 logits], max over points, then the output MLP.
        float[] iouHidden = AddTerms(p.Iou, n, _iou.Hidden, PromptTerm(_iou, globalVector, prompt), stage2, _iou, _hidden2 + _featureDim + PointWidth + PromptWidth);
        float[] pooled = MaxRows(Tail(backend, _iou, iouHidden, n, activateLast: false), n, _iou.Out);
        float[] iou = _iouOut.ForwardRow(backend, pooled);
        for (int i = 0; i < iou.Length; i++)
            iou[i] = Sigmoid(iou[i]);

        float[][] masks = new float[Masks][];
        for (int m = 0; m < Masks; m++)
        {
            masks[m] = new float[n];
            for (int i = 0; i < n; i++)
                masks[m][i] = Sigmoid(stage2[i * Masks + m]);
        }

        return new Result(masks, iou);
    }

    // The first linear layer restricted to the prompt-independent input columns: W[:, featureStart .. featureStart + featureDim) applied to the features,
    // the three point columns that follow applied to the points, plus the bias. Returns [count, hidden].
    private static float[] FirstLayer(IBackend backend, Mlp mlp, float[] features, float[] points, int count, int featureStart, int featureDim)
    {
        int hidden = mlp.Hidden;
        float[] w = mlp.W0;
        float[] featureWeights = new float[hidden * featureDim];
        for (int r = 0; r < hidden; r++)
            Array.Copy(w, (long)r * mlp.In + featureStart, featureWeights, (long)r * featureDim, featureDim);
        float[] result = Matmul(backend, features, count, featureDim, featureWeights, hidden, null);
        int pointStart = featureStart + featureDim;
        for (int i = 0; i < count; i++)
            for (int r = 0; r < hidden; r++)
            {
                float acc = mlp.B0[r];
                for (int c = 0; c < PointWidth; c++)
                    acc += w[(long)r * mlp.In + pointStart + c] * points[i * PointWidth + c];
                result[(long)i * hidden + r] += acc;
            }

        return result;
    }

    // The prompt columns of the first layer applied to the prompt, [hidden]; `promptStart` is the column of the first prompt value.
    private static float[] Prompt(Mlp mlp, int promptStart, ReadOnlySpan<float> prompt)
    {
        float[] term = new float[mlp.Hidden];
        for (int r = 0; r < mlp.Hidden; r++)
            for (int c = 0; c < PromptWidth; c++)
                term[r] += mlp.W0[(long)r * mlp.In + promptStart + c] * prompt[c];
        return term;
    }

    // Stage-2 and IoU first layers: the global vector sits in columns [0, hidden2), so its term is added together with the prompt term.
    private float[] PromptTerm(Mlp mlp, float[] globalVector, ReadOnlySpan<float> prompt)
    {
        float[] term = Prompt(mlp, _hidden2 + _featureDim + PointWidth, prompt);
        for (int r = 0; r < mlp.Hidden; r++)
            for (int c = 0; c < _hidden2; c++)
                term[r] += mlp.W0[(long)r * mlp.In + c] * globalVector[c];
        return term;
    }

    // base[i] + promptTerm + W0[:, logitStart ..] . logits[i]  (activation not yet applied).
    private static float[] AddTerms(float[] baseTerms, int n, int hidden, float[] promptTerm, float[] logits, Mlp mlp, int logitStart)
    {
        float[] result = new float[(long)n * hidden];
        for (int i = 0; i < n; i++)
            for (int r = 0; r < hidden; r++)
            {
                float acc = baseTerms[(long)i * hidden + r] + promptTerm[r];
                for (int c = 0; c < Masks; c++)
                    acc += mlp.W0[(long)r * mlp.In + logitStart + c] * logits[i * Masks + c];
                result[(long)i * hidden + r] = acc;
            }

        return result;
    }

    // Layers 2 and 3 (the last without its activation) of a head whose first layer output is `firstLayer` before the GELU.
    private static float[] Tail(IBackend backend, Mlp mlp, float[] firstLayer, int n, bool activateLast)
    {
        float[] h = Gelu(backend, firstLayer);
        h = Matmul(backend, h, n, mlp.Hidden, mlp.W1, mlp.Hidden, mlp.B1);
        h = Gelu(backend, h);
        float[] output = Matmul(backend, h, n, mlp.Hidden, mlp.W2, mlp.Out, mlp.B2);
        return activateLast ? Gelu(backend, output) : output;
    }

    private static void Head(IBackend backend, Mlp mlp, float[] baseTerms, int n, float[] promptTerm, float[] destination, int column)
    {
        float[] first = new float[baseTerms.Length];
        for (int i = 0; i < n; i++)
            for (int r = 0; r < mlp.Hidden; r++)
                first[(long)i * mlp.Hidden + r] = baseTerms[(long)i * mlp.Hidden + r] + promptTerm[r];
        Logits(backend, mlp, first, n, destination, column);
    }

    private static void Logits(IBackend backend, Mlp mlp, float[] first, int n, float[] destination, int column)
    {
        float[] logits = Tail(backend, mlp, first, n, activateLast: false);
        for (int i = 0; i < n; i++)
            destination[i * Masks + column] = logits[i];
    }

    private static float[] MaxRows(float[] values, int n, int width)
    {
        float[] max = new float[width];
        Array.Fill(max, float.NegativeInfinity);
        for (int i = 0; i < n; i++)
            for (int c = 0; c < width; c++)
                max[c] = Math.Max(max[c], values[(long)i * width + c]);
        return max;
    }

    // nn.GELU (exact, erf based) through the backend's own kernel.
    private static float[] Gelu(IBackend backend, float[] x)
    {
        using Tensor input = new(new TensorShape(x.Length), DType.F32);
        using Tensor output = new(new TensorShape(x.Length), DType.F32);
        Copy(x, input);
        backend.GeluErf(output, input);
        float[] y = new float[x.Length];
        Read(output, y);
        return y;
    }

    private static float Sigmoid(float x) => (float)(1.0 / (1.0 + Math.Exp(-x)));

    private static float[] Matmul(IBackend backend, float[] input, int rows, int inDim, float[] weight, int outDim, float[]? bias)
    {
        using Tensor x = new(new TensorShape(rows, inDim), DType.F32);
        using Tensor w = new(new TensorShape(outDim, inDim), DType.F32);
        using Tensor y = new(new TensorShape(rows, outDim), DType.F32);
        Tensor? b = bias is null ? null : new Tensor(new TensorShape(outDim), DType.F32);
        try
        {
            Copy(input, x);
            Copy(weight, w);
            if (b is not null)
                Copy(bias!, b);
            backend.Linear(y, x, w, b);
            float[] result = new float[(long)rows * outDim];
            Read(y, result);
            return result;
        }
        finally
        {
            b?.Dispose();
        }
    }

    private static unsafe void Copy(float[] source, Tensor destination) =>
        new ReadOnlySpan<float>(source).CopyTo(new Span<float>((float*)destination.DataPointer, source.Length));

    private static unsafe void Read(Tensor source, float[] destination) =>
        new ReadOnlySpan<float>((float*)source.DataPointer, destination.Length).CopyTo(destination);

    private static void Expect(bool condition, string what)
    {
        if (!condition)
            throw new HartsyInferenceException($"P3-SAM head shapes do not fit together: {what}.");
    }

    private sealed class Mlp
    {
        public readonly float[] W0, B0, W1, B1, W2, B2;
        public readonly int In, Hidden, Out;

        public Mlp(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            Tensor w0 = Get(weights, prefix, 0, "weight"), w1 = Get(weights, prefix, 2, "weight"), w2 = Get(weights, prefix, 4, "weight");
            Hidden = (int)w0.Shape[0];
            In = (int)w0.Shape[1];
            Out = (int)w2.Shape[0];
            if (w1.Shape[0] != Hidden || w1.Shape[1] != Hidden || w2.Shape[1] != Hidden)
                throw new HartsyInferenceException($"P3-SAM MLP '{prefix}' has inconsistent layer shapes.");
            W0 = ToArray(w0);
            B0 = ToArray(Get(weights, prefix, 0, "bias"));
            W1 = ToArray(w1);
            B1 = ToArray(Get(weights, prefix, 2, "bias"));
            W2 = ToArray(w2);
            B2 = ToArray(Get(weights, prefix, 4, "bias"));
        }

        // One row through the whole MLP (GELU between layers), for the small output heads.
        public float[] ForwardRow(IBackend backend, float[] row)
        {
            float[] h = Matmul(backend, row, 1, In, W0, Hidden, B0);
            h = Gelu(backend, h);
            h = Matmul(backend, h, 1, Hidden, W1, Hidden, B1);
            h = Gelu(backend, h);
            return Matmul(backend, h, 1, Hidden, W2, Out, B2);
        }

        private static Tensor Get(IReadOnlyDictionary<string, Tensor> weights, string prefix, int layer, string kind)
        {
            string key = $"{prefix}.{layer}.{kind}";
            if (!weights.TryGetValue(key, out Tensor? tensor))
                throw new HartsyInferenceException($"P3-SAM weight '{key}' is missing.");
            if (tensor.DType != DType.F32)
                throw new HartsyInferenceException($"P3-SAM weight '{key}' must be F32, got {tensor.DType.Name}.");
            return tensor;
        }

        private static unsafe float[] ToArray(Tensor t)
        {
            float[] values = new float[t.ElementCount];
            new ReadOnlySpan<float>((float*)t.DataPointer, values.Length).CopyTo(values);
            return values;
        }
    }
}
