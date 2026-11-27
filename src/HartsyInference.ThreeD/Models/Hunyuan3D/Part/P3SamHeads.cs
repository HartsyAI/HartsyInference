using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Part;

/// <summary>The mask and IoU heads of P3-SAM (<c>Hunyuan3D-Part/P3-SAM/model.py</c>), plain MLPs of <c>Linear, GELU, Linear, GELU, Linear</c> over per-point
/// features: stage 1 (<c>seg_mlp_1..3</c>) reads <c>[feature, point, prompt]</c>; stage 2 (<c>seg_s2_mlp_g</c>, <c>seg_s2_mlp_1..3</c>) adds the stage-1
/// logits and a max-pooled global vector; <c>iou_mlp</c> and <c>iou_mlp_out</c> predict the IoU of the three masks.
/// <para>The first layer of every head is linear in a concatenation whose per-point part (feature and point) does not depend on the prompt, so that part
/// is computed once per mesh (<see cref="Prepare"/>, kept as tensors the backend holds on its device) and only the prompt and logit terms are added per
/// prompt. The result is the same function as upstream's concatenate-then-MLP, evaluated in a different order. A prompt runs entirely on the backend; only
/// the per-point logits (<c>[points, 1]</c>) and the two max-pool inputs come back to the host.</para></summary>
public sealed class P3SamHeads : IDisposable
{
    private const int PointWidth = 3, PromptWidth = 3, Masks = 3;
    private readonly P3SamMlp _seg1, _seg2, _seg3, _g, _s21, _s22, _s23, _iou, _iouOut;
    private readonly int _featureDim, _hidden2;

    /// <summary>Reads the heads from <paramref name="weights"/> (keys such as <c>seg_mlp_1.0.weight</c>, the names in <c>p3sam.safetensors</c>).</summary>
    public P3SamHeads(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        List<P3SamMlp> made = [];
        try
        {
            P3SamMlp Make(string prefix)
            {
                P3SamMlp mlp = new(weights, prefix);
                made.Add(mlp);
                return mlp;
            }

            _seg1 = Make("seg_mlp_1");
            _seg2 = Make("seg_mlp_2");
            _seg3 = Make("seg_mlp_3");
            _g = Make("seg_s2_mlp_g");
            _s21 = Make("seg_s2_mlp_1");
            _s22 = Make("seg_s2_mlp_2");
            _s23 = Make("seg_s2_mlp_3");
            _iou = Make("iou_mlp");
            _iouOut = Make("iou_mlp_out");
            _featureDim = _seg1.In - PointWidth - PromptWidth;
            _hidden2 = _g.Out;
            Expect(_g.In == _featureDim + PointWidth + PromptWidth + Masks, "seg_s2_mlp_g input");
            foreach (P3SamMlp s2 in new[] { _s21, _s22, _s23 })
                Expect(s2.In == _hidden2 + _featureDim + PointWidth + PromptWidth + Masks && s2.Out == 1, "seg_s2_mlp input");
            Expect(_iou.In == _hidden2 + _featureDim + PointWidth + PromptWidth + Masks && _iouOut.Out == Masks, "iou head input");
        }
        catch
        {
            foreach (P3SamMlp mlp in made)
                mlp.Dispose();
            throw;
        }
    }

    /// <summary>Releases the layer tensors of every head.</summary>
    public void Dispose()
    {
        foreach (P3SamMlp mlp in new[] { _seg1, _seg2, _seg3, _g, _s21, _s22, _s23, _iou, _iouOut })
            mlp.Dispose();
    }

    /// <summary>Per-point feature width (512 for P3-SAM).</summary>
    public int FeatureDim => _featureDim;

    /// <summary>The prompt-independent first-layer terms of every head for one mesh, with the work buffers a prompt needs, as backend tensors. Dispose it
    /// when the mesh is done. At 100,000 points it holds about 2.3 GB on the device (8 first-layer terms and 3 work buffers of <c>[points, 512]</c> in F32).
    /// The work buffers are shared, so one instance serves one <see cref="Segment"/> call at a time.</summary>
    public sealed class Prepared : IDisposable
    {
        internal Tensor Seg1 = null!, Seg2 = null!, Seg3 = null!, G = null!, S21 = null!, S22 = null!, S23 = null!, Iou = null!;
        // The logit columns of the first layer of g, s21, s22, s23 and iou, and a column of ones for the heads that have no logits.
        internal Tensor[] LogitWeights = [];
        internal Tensor Ones = null!;
        internal readonly int Count;
        private readonly Dictionary<(int Slot, int Width), Tensor> _buffers = [];

        internal Prepared(int count) => Count = count;

        // A work tensor [Count, width] reused across prompts; `slot` tells apart buffers that are live at the same time.
        internal Tensor Buffer(int slot, int width)
        {
            if (!_buffers.TryGetValue((slot, width), out Tensor? tensor))
            {
                tensor = new Tensor(new TensorShape(Count, width), DType.F32);
                _buffers[(slot, width)] = tensor;
            }

            return tensor;
        }

        /// <summary>Releases the per-mesh tensors.</summary>
        public void Dispose()
        {
            foreach (Tensor tensor in _buffers.Values.Concat(LogitWeights))
                tensor.Dispose();
            foreach (Tensor? tensor in new[] { Seg1, Seg2, Seg3, G, S21, S22, S23, Iou, Ones })
                tensor?.Dispose();
        }
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
        int d = _featureDim, width = d + PointWidth;
        // The feature and point columns of every first layer are adjacent, so one product over [feature, point] rows gives both terms and the bias.
        Tensor columns = new(new TensorShape(count, width), DType.F32);
        List<Tensor> temporaries = [columns];
        Prepared p = new(count);
        try
        {
            P3SamMlp.CopyRows(features, d, points, PointWidth, count, columns);
            p.Seg1 = FirstLayer(backend, _seg1, columns, count, 0, width, temporaries);
            p.Seg2 = FirstLayer(backend, _seg2, columns, count, 0, width, temporaries);
            p.Seg3 = FirstLayer(backend, _seg3, columns, count, 0, width, temporaries);
            p.G = FirstLayer(backend, _g, columns, count, 0, width, temporaries);
            // Stage-2 and IoU inputs are [global, feature, point, prompt, logits]: the feature block starts after the global vector.
            p.S21 = FirstLayer(backend, _s21, columns, count, _hidden2, width, temporaries);
            p.S22 = FirstLayer(backend, _s22, columns, count, _hidden2, width, temporaries);
            p.S23 = FirstLayer(backend, _s23, columns, count, _hidden2, width, temporaries);
            p.Iou = FirstLayer(backend, _iou, columns, count, _hidden2, width, temporaries);
            int logitStart = _featureDim + PointWidth + PromptWidth;
            p.LogitWeights = [LogitWeights(_g, logitStart), .. new[] { _s21, _s22, _s23, _iou }.Select(s2 => LogitWeights(s2, _hidden2 + logitStart))];
            p.Ones = new Tensor(new TensorShape(count, 1), DType.F32);
            P3SamMlp.Fill(p.Ones, 1f);
            backend.Sync();
        }
        catch
        {
            p.Dispose();
            throw;
        }
        finally
        {
            foreach (Tensor tensor in temporaries)
                tensor.Dispose();
        }

        return p;
    }

    /// <summary>Predicts the three masks and their IoUs for one prompt point.</summary>
    public Result Segment(IBackend backend, Prepared p, ReadOnlySpan<float> prompt)
    {
        if (prompt.Length != PromptWidth)
            throw new ArgumentException("A prompt is one 3D point.", nameof(prompt));
        int n = p.Count;
        List<Tensor> temporaries = [];
        try
        {
            // Stage 1: three heads, each [feature, point, prompt] -> one logit per point.
            float[] stage1 = new float[n * Masks];
            P3SamMlp[] s1 = [_seg1, _seg2, _seg3];
            Tensor[] s1Base = [p.Seg1, p.Seg2, p.Seg3];
            for (int h = 0; h < Masks; h++)
            {
                // The prompt term of every row: ones [n, 1] times the term as a [hidden, 1] weight.
                Tensor term = Small(Prompt(s1[h], _featureDim + PointWidth, prompt), temporaries, s1[h].Hidden, 1);
                Tensor bufferTerm = p.Buffer(2, s1[h].Hidden);
                backend.Linear(bufferTerm, p.Ones, term, null);
                Tensor logits = Tail(backend, p, s1[h], s1Base[h], bufferTerm);
                ReadColumn(logits, n, stage1, h);
            }

            Tensor stage1Tensor = Columns(stage1, n, temporaries);

            // Stage 2: the global vector is the point-wise max of the g-MLP over [feature, point, prompt, stage-1 logits].
            Tensor gOut = TailWithLogits(backend, p, _g, p.G, p.LogitWeights[0], stage1Tensor, Prompt(_g, _featureDim + PointWidth, prompt), temporaries);
            float[] globalVector = P3SamMlp.MaxRows(gOut, n, _hidden2);

            float[] stage2 = new float[n * Masks];
            P3SamMlp[] s2 = [_s21, _s22, _s23];
            Tensor[] s2Base = [p.S21, p.S22, p.S23];
            for (int h = 0; h < Masks; h++)
            {
                Tensor logits = TailWithLogits(backend, p, s2[h], s2Base[h], p.LogitWeights[1 + h], stage1Tensor, PromptTerm(s2[h], globalVector, prompt), temporaries);
                ReadColumn(logits, n, stage2, h);
            }

            // IoU head over [global, feature, point, prompt, stage-2 logits], max over points, then the output MLP.
            Tensor stage2Tensor = Columns(stage2, n, temporaries);
            Tensor iouOut = TailWithLogits(backend, p, _iou, p.Iou, p.LogitWeights[4], stage2Tensor, PromptTerm(_iou, globalVector, prompt), temporaries);
            float[] pooled = P3SamMlp.MaxRows(iouOut, n, _iou.Out);
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
        finally
        {
            foreach (Tensor tensor in temporaries)
                tensor.Dispose();
        }
    }

    // The first linear layer restricted to the prompt-independent input columns: W[:, featureStart .. featureStart + width) applied to [feature, point] rows,
    // plus the bias. Returns [count, hidden].
    private static Tensor FirstLayer(IBackend backend, P3SamMlp mlp, Tensor columns, int count, int featureStart, int width, List<Tensor> temporaries)
    {
        float[] slice = new float[(long)mlp.Hidden * width];
        for (int r = 0; r < mlp.Hidden; r++)
            Array.Copy(mlp.W0, (long)r * mlp.In + featureStart, slice, (long)r * width, width);
        Tensor weight = P3SamMlp.Own(slice, mlp.Hidden, width);
        temporaries.Add(weight);
        Tensor result = new(new TensorShape(count, mlp.Hidden), DType.F32);
        try
        {
            backend.Linear(result, columns, weight, mlp.TB0);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
    }

    // The columns [start, start + 3) of the first layer of `mlp` as a [hidden, 3] weight, which multiplies the stage logits.
    private static Tensor LogitWeights(P3SamMlp mlp, int start)
    {
        float[] slice = new float[mlp.Hidden * Masks];
        for (int r = 0; r < mlp.Hidden; r++)
            Array.Copy(mlp.W0, (long)r * mlp.In + start, slice, (long)r * Masks, Masks);
        return P3SamMlp.Own(slice, mlp.Hidden, Masks);
    }

    // base + W_logits . logits + shift, then layers 2 and 3 of `mlp`. `shift` carries the prompt term (and the global-vector term).
    private static Tensor TailWithLogits(IBackend backend, Prepared p, P3SamMlp mlp, Tensor baseTerms, Tensor logitWeights, Tensor logits, float[] shift,
        List<Tensor> temporaries)
    {
        Tensor bias = Small(shift, temporaries, mlp.Hidden);
        Tensor term = p.Buffer(2, mlp.Hidden);
        backend.Linear(term, logits, logitWeights, bias);
        return Tail(backend, p, mlp, baseTerms, term);
    }

    // first = base + term (in buffer 0), then GELU, the second layer, GELU and the last layer; returns the output buffer [n, Out].
    // Buffer roles: 0 and 1 ping-pong through the layers, 2 holds the prompt/logit term, 3 is the output. The output is shared by every head, so the caller
    // reads it (logit column or max-pool) before the next head runs.
    private static Tensor Tail(IBackend backend, Prepared p, P3SamMlp mlp, Tensor baseTerms, Tensor term)
    {
        Tensor a = p.Buffer(0, mlp.Hidden), b = p.Buffer(1, mlp.Hidden), output = p.Buffer(3, mlp.Out);
        backend.Add(a, baseTerms, term);
        backend.GeluErf(b, a);
        backend.Linear(a, b, mlp.T1, mlp.TB1);
        backend.GeluErf(b, a);
        backend.Linear(output, b, mlp.T2, mlp.TB2);
        return output;
    }

    // A small tensor of `values` with the given shape; released with the other temporaries of the prompt.
    private static Tensor Small(float[] values, List<Tensor> temporaries, params long[] shape)
    {
        Tensor tensor = new(new TensorShape(shape), DType.F32) { ForceF32Gemm = true };
        P3SamMlp.Copy(values, tensor);
        temporaries.Add(tensor);
        return tensor;
    }

    // [n, 3] logits as a tensor.
    private static Tensor Columns(float[] values, int n, List<Tensor> temporaries)
    {
        Tensor tensor = new(new TensorShape(n, Masks), DType.F32);
        P3SamMlp.Copy(values, tensor);
        temporaries.Add(tensor);
        return tensor;
    }

    // Reads a [n, 1] logit tensor into column `column` of an [n, 3] host array.
    private static void ReadColumn(Tensor logits, int n, float[] destination, int column)
    {
        float[] values = new float[n];
        P3SamMlp.Read(logits, values);
        for (int i = 0; i < n; i++)
            destination[i * Masks + column] = values[i];
    }

    // The prompt columns of the first layer applied to the prompt, [hidden]; `promptStart` is the column of the first prompt value.
    private static float[] Prompt(P3SamMlp mlp, int promptStart, ReadOnlySpan<float> prompt)
    {
        float[] term = new float[mlp.Hidden];
        for (int r = 0; r < mlp.Hidden; r++)
            for (int c = 0; c < PromptWidth; c++)
                term[r] += mlp.W0[(long)r * mlp.In + promptStart + c] * prompt[c];
        return term;
    }

    // Stage-2 and IoU first layers: the global vector sits in columns [0, hidden2), so its term is added together with the prompt term.
    private float[] PromptTerm(P3SamMlp mlp, float[] globalVector, ReadOnlySpan<float> prompt)
    {
        float[] term = Prompt(mlp, _hidden2 + _featureDim + PointWidth, prompt);
        for (int r = 0; r < mlp.Hidden; r++)
            for (int c = 0; c < _hidden2; c++)
                term[r] += mlp.W0[(long)r * mlp.In + c] * globalVector[c];
        return term;
    }

    private static float Sigmoid(float x) => (float)(1.0 / (1.0 + Math.Exp(-x)));

    private static void Expect(bool condition, string what)
    {
        if (!condition)
            throw new HartsyInferenceException($"P3-SAM head shapes do not fit together: {what}.");
    }
}
