using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Decision.Clef;

/// <summary>Port of Cloudflare Clef's <c>JointSchemaHead</c> (<c>joint_schema_model.py</c>): scores every allowed option of
/// every question in one pass from the backbone's final hidden states. Computed in F32 with host-side attention; the large
/// projections go through <see cref="IBackend.Linear"/>.</summary>
public sealed unsafe class ClefJointHead : IDisposable
{
    private const float LayerNormEps = 1e-5f;
    private const float UnitEps = 1e-8f;
    private const float NormalizeEps = 1e-12f;

    private readonly ClefJointHeadConfig _cfg;
    private IReadOnlyDictionary<string, Tensor> _w = new Dictionary<string, Tensor>();
    private readonly List<Tensor> _owned = [];
    private float _priorScale, _jointScale, _gate;
    private int _disposed;

    public ClefJointHead(ClefJointHeadConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.Width % cfg.Heads != 0)
        {
            throw new ArgumentException("Head width must divide evenly by the head count.", nameof(cfg));
        }
        _cfg = cfg;
    }

    /// <summary>Loads the state dict of <c>JointSchemaHead</c> (<c>joint_head.safetensors</c>), converted to F32.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        Dictionary<string, Tensor> f32 = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> kv in weights)
        {
            if (kv.Value.DType == DType.F32)
            {
                f32[kv.Key] = kv.Value;
                continue;
            }
            Tensor converted = kv.Value.CastTo(DType.F32);
            _owned.Add(converted);
            f32[kv.Key] = converted;
        }
        _w = f32;
        _priorScale = MathF.Exp(MathF.Min(Scalar("prior_logit_scale"), MathF.Log(100f)));
        _jointScale = MathF.Exp(MathF.Min(Scalar("joint_logit_scale"), MathF.Log(100f)));
        _gate = 1f / (1f + MathF.Exp(-Scalar("residual_gate")));
    }

    /// <summary>Returns one logit per option for each question, in question order.</summary>
    /// <param name="hidden">Backbone last hidden states, <c>[seq, HiddenSize]</c> row-major.</param>
    /// <param name="inputIds">Token ids of the encoded record (length <paramref name="seq"/>).</param>
    /// <param name="outputEmbedding">Row accessor for the backbone's output embedding (<c>lm_head</c>) matrix.</param>
    public float[][] Forward(IBackend backend, float[] hidden, int seq, ReadOnlySpan<int> inputIds,
        IReadOnlyList<ClefQuestionSpans> questions, Func<int, float[]> outputEmbedding)
    {
        int hs = _cfg.HiddenSize, width = _cfg.Width;
        if (hidden.Length != seq * hs || inputIds.Length != seq)
        {
            throw new ArgumentException("Hidden states and input ids must both cover the sequence.");
        }
        if (questions.Count == 0)
        {
            throw new ArgumentException("A record needs at least one question.", nameof(questions));
        }
        float[] normHidden = LayerNorm(hidden, seq, hs, "hidden_norm");
        float[] memory = Linear(backend, normHidden, seq, hs, width, "memory_projection.weight", null);
        float[] global = normHidden.AsSpan((seq - 1) * hs, hs).ToArray();

        int qCount = questions.Count;
        float[] questionVectors = new float[qCount * hs];
        int totalOptions = 0;
        foreach (ClefQuestionSpans q in questions)
        {
            totalOptions += q.OptionSpans.Length;
        }
        float[] optionContext = new float[totalOptions * hs], lexical = new float[totalOptions * hs];
        int[] optionQuestion = new int[totalOptions];
        int o = 0;
        for (int qi = 0; qi < qCount; qi++)
        {
            MeanSpan(normHidden, hs, questions[qi].QuestionSpan, questionVectors.AsSpan(qi * hs, hs));
            foreach ((int start, int end) in questions[qi].OptionSpans)
            {
                MeanSpan(normHidden, hs, (start, end), optionContext.AsSpan(o * hs, hs));
                Span<float> lex = lexical.AsSpan(o * hs, hs);
                for (int t = start; t < end; t++)
                {
                    float[] row = outputEmbedding(inputIds[t]);
                    for (int c = 0; c < hs; c++)
                    {
                        lex[c] += row[c];
                    }
                }
                for (int c = 0; c < hs; c++)
                {
                    lex[c] /= end - start;
                }
                optionQuestion[o++] = qi;
            }
        }

        float[] ctxProj = Linear(backend, optionContext, totalOptions, hs, width, "option_context_projection.weight", null);
        float[] lexProj = Linear(backend, lexical, totalOptions, hs, width, "option_lexical_projection.weight", null);
        float[] qOptProj = Linear(backend, questionVectors, qCount, hs, width, "option_question_projection.weight", null);
        float[] routed = new float[totalOptions * width];
        for (int i = 0; i < totalOptions; i++)
        {
            for (int c = 0; c < width; c++)
            {
                routed[i * width + c] = ctxProj[i * width + c] + lexProj[i * width + c] + qOptProj[optionQuestion[i] * width + c];
            }
        }

        for (int layer = 0; layer < _cfg.RoutingLayers; layer++)
        {
            routed = EvidenceLayer(backend, routed, totalOptions, memory, seq, $"evidence_layers.{layer}");
        }

        float[] baseFields = Linear(backend, questionVectors, qCount, hs, width, "question_projection.weight", null);
        float[] summaries = new float[qCount * width];
        int offset = 0;
        double scale = Math.Sqrt(width);
        for (int qi = 0; qi < qCount; qi++)
        {
            int count = questions[qi].OptionSpans.Length;
            double[] scores = new double[count];
            double max = double.NegativeInfinity;
            for (int i = 0; i < count; i++)
            {
                double dot = 0;
                for (int c = 0; c < width; c++)
                {
                    dot += routed[(offset + i) * width + c] * (double)baseFields[qi * width + c];
                }
                scores[i] = dot / scale;
                max = Math.Max(max, scores[i]);
            }
            double sum = 0;
            for (int i = 0; i < count; i++)
            {
                scores[i] = Math.Exp(scores[i] - max);
                sum += scores[i];
            }
            for (int i = 0; i < count; i++)
            {
                float weight = (float)(scores[i] / sum);
                for (int c = 0; c < width; c++)
                {
                    summaries[qi * width + c] += weight * routed[(offset + i) * width + c];
                }
            }
            offset += count;
        }
        float[] summaryNorm = LayerNorm(summaries, qCount, width, "option_summary_norm");
        float[] globalProj = Linear(backend, global, 1, hs, width, "global_projection.weight", null);
        float[] typeEmbedding = Weight("type_embedding.weight");
        float[] fields = new float[qCount * width];
        for (int qi = 0; qi < qCount; qi++)
        {
            for (int c = 0; c < width; c++)
            {
                fields[qi * width + c] = baseFields[qi * width + c] + summaryNorm[qi * width + c] + globalProj[c]
                    + typeEmbedding[questions[qi].QuestionType * width + c];
            }
        }
        for (int layer = 0; layer < _cfg.Layers; layer++)
        {
            fields = DecoderLayer(backend, fields, qCount, memory, seq, $"layers.{layer}");
        }
        fields = LayerNorm(fields, qCount, width, "field_norm");
        float[] optionNorm = LayerNorm(routed, totalOptions, width, "option_norm");

        float[][] logits = new float[qCount][];
        offset = 0;
        for (int qi = 0; qi < qCount; qi++)
        {
            int count = questions[qi].OptionSpans.Length;
            float[] anchor = new float[hs];
            for (int c = 0; c < hs; c++)
            {
                anchor[c] = questionVectors[qi * hs + c] + global[c];
            }
            Normalize(anchor);
            float[] features = new float[count * width * 4];
            float[] cosine = new float[count], prior = new float[count];
            ReadOnlySpan<float> field = fields.AsSpan(qi * width, width);
            for (int i = 0; i < count; i++)
            {
                float[] lex = lexical.AsSpan((offset + i) * hs, hs).ToArray();
                Normalize(lex);
                float dotPrior = 0;
                for (int c = 0; c < hs; c++)
                {
                    dotPrior += lex[c] * anchor[c];
                }
                prior[i] = _priorScale * dotPrior;
                ReadOnlySpan<float> option = optionNorm.AsSpan((offset + i) * width, width);
                cosine[i] = CosineSimilarity(field, option);
                for (int c = 0; c < width; c++)
                {
                    features[(i * 4 + 0) * width + c] = field[c];
                    features[(i * 4 + 1) * width + c] = option[c];
                    features[(i * 4 + 2) * width + c] = field[c] * option[c];
                    features[(i * 4 + 3) * width + c] = MathF.Abs(field[c] - option[c]);
                }
            }
            float[] hiddenScore = Linear(backend, features, count, width * 4, width, "residual_scorer.0.weight", "residual_scorer.0.bias");
            Gelu(hiddenScore);
            float[] residual = Linear(backend, hiddenScore, count, width, 1, "residual_scorer.3.weight", "residual_scorer.3.bias");
            float[] result = new float[count];
            for (int i = 0; i < count; i++)
            {
                result[i] = prior[i] + _gate * (_jointScale * cosine[i] + residual[i]);
            }
            logits[qi] = result;
            offset += count;
        }
        return logits;
    }

    /// <summary>Releases the F32 conversions this head made of the checkpoint tensors.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor t in _owned)
        {
            t.Dispose();
        }
        _owned.Clear();
    }

    private float[] EvidenceLayer(IBackend backend, float[] queries, int n, float[] memory, int memLen, string p)
    {
        int width = _cfg.Width;
        float[] q = LayerNorm(queries, n, width, $"{p}.query_norm");
        float[] mem = LayerNorm(memory, memLen, width, $"{p}.memory_norm");
        float[] routed = Attention(backend, q, n, mem, memLen, $"{p}.attention");
        for (int i = 0; i < routed.Length; i++)
        {
            routed[i] += queries[i];
        }
        float[] ff = LayerNorm(routed, n, width, $"{p}.feedforward_norm");
        float[] hidden = Linear(backend, ff, n, width, _cfg.Feedforward, $"{p}.feedforward.0.weight", $"{p}.feedforward.0.bias");
        Gelu(hidden);
        float[] output = Linear(backend, hidden, n, _cfg.Feedforward, width, $"{p}.feedforward.3.weight", $"{p}.feedforward.3.bias");
        for (int i = 0; i < output.Length; i++)
        {
            output[i] += routed[i];
        }
        return output;
    }

    private float[] DecoderLayer(IBackend backend, float[] x, int n, float[] memory, int memLen, string p)
    {
        int width = _cfg.Width;
        float[] a = LayerNorm(x, n, width, $"{p}.norm1");
        float[] selfOut = Attention(backend, a, n, a, n, $"{p}.self_attn");
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += selfOut[i];
        }
        float[] b = LayerNorm(x, n, width, $"{p}.norm2");
        float[] cross = Attention(backend, b, n, memory, memLen, $"{p}.multihead_attn");
        for (int i = 0; i < x.Length; i++)
        {
            x[i] += cross[i];
        }
        float[] c = LayerNorm(x, n, width, $"{p}.norm3");
        float[] hidden = Linear(backend, c, n, width, _cfg.Feedforward, $"{p}.linear1.weight", $"{p}.linear1.bias");
        Gelu(hidden);
        float[] output = Linear(backend, hidden, n, _cfg.Feedforward, width, $"{p}.linear2.weight", $"{p}.linear2.bias");
        for (int i = 0; i < output.Length; i++)
        {
            output[i] += x[i];
        }
        return output;
    }

    /// <summary><c>nn.MultiheadAttention</c> (batch-first, no mask): packed in-projection, scaled dot-product, out-projection.</summary>
    private float[] Attention(IBackend backend, float[] query, int nq, float[] keyValue, int nk, string p)
    {
        int width = _cfg.Width, heads = _cfg.Heads, hd = width / heads;
        float[] inW = Weight($"{p}.in_proj_weight"), inB = Weight($"{p}.in_proj_bias");
        float[] q = LinearRaw(backend, query, nq, width, width, inW.AsSpan(0, width * width), inB.AsSpan(0, width));
        float[] k = LinearRaw(backend, keyValue, nk, width, width, inW.AsSpan(width * width, width * width), inB.AsSpan(width, width));
        float[] v = LinearRaw(backend, keyValue, nk, width, width, inW.AsSpan(2 * width * width, width * width), inB.AsSpan(2 * width, width));
        float[] context = new float[nq * width];
        float scale = 1f / MathF.Sqrt(hd);
        Parallel.For(0, heads * nq, idx =>
        {
            int h = idx / nq, i = idx % nq;
            float[] scores = new float[nk];
            float max = float.NegativeInfinity;
            for (int j = 0; j < nk; j++)
            {
                float dot = 0;
                int qo = i * width + h * hd, ko = j * width + h * hd;
                for (int c = 0; c < hd; c++)
                {
                    dot += q[qo + c] * k[ko + c];
                }
                scores[j] = dot * scale;
                max = MathF.Max(max, scores[j]);
            }
            float sum = 0;
            for (int j = 0; j < nk; j++)
            {
                scores[j] = MathF.Exp(scores[j] - max);
                sum += scores[j];
            }
            int co = i * width + h * hd;
            for (int j = 0; j < nk; j++)
            {
                float weight = scores[j] / sum;
                int vo = j * width + h * hd;
                for (int c = 0; c < hd; c++)
                {
                    context[co + c] += weight * v[vo + c];
                }
            }
        });
        return Linear(backend, context, nq, width, width, $"{p}.out_proj.weight", $"{p}.out_proj.bias");
    }

    private float[] Linear(IBackend backend, float[] x, int n, int inDim, int outDim, string weightKey, string? biasKey)
    {
        Tensor weight = _w[weightKey];
        if (weight.Shape[0] != outDim || weight.Shape[1] != inDim)
        {
            throw new InvalidDataException($"'{weightKey}' is {weight.Shape}, expected [{outDim}, {inDim}].");
        }
        using Tensor input = new(new TensorShape(1, n, inDim), DType.F32);
        x.AsSpan(0, n * inDim).CopyTo(new Span<float>((void*)input.DataPointer, n * inDim));
        using Tensor output = new(new TensorShape(1, n, outDim), DType.F32);
        backend.Linear(output, input, weight, biasKey is null ? null : _w[biasKey]);
        backend.Sync();
        return new Span<float>((void*)output.DataPointer, n * outDim).ToArray();
    }

    private static float[] LinearRaw(IBackend backend, float[] x, int n, int inDim, int outDim,
        ReadOnlySpan<float> weight, ReadOnlySpan<float> bias)
    {
        using Tensor w = new(new TensorShape(outDim, inDim), DType.F32);
        weight.CopyTo(new Span<float>((void*)w.DataPointer, outDim * inDim));
        using Tensor b = new(new TensorShape(outDim), DType.F32);
        bias.CopyTo(new Span<float>((void*)b.DataPointer, outDim));
        using Tensor input = new(new TensorShape(1, n, inDim), DType.F32);
        x.AsSpan(0, n * inDim).CopyTo(new Span<float>((void*)input.DataPointer, n * inDim));
        using Tensor output = new(new TensorShape(1, n, outDim), DType.F32);
        backend.Linear(output, input, w, b);
        backend.Sync();
        return new Span<float>((void*)output.DataPointer, n * outDim).ToArray();
    }

    private float[] LayerNorm(float[] x, int n, int dim, string p)
    {
        float[] weight = Weight($"{p}.weight"), bias = Weight($"{p}.bias");
        float[] result = new float[n * dim];
        for (int i = 0; i < n; i++)
        {
            double mean = 0;
            for (int c = 0; c < dim; c++)
            {
                mean += x[i * dim + c];
            }
            mean /= dim;
            double variance = 0;
            for (int c = 0; c < dim; c++)
            {
                double d = x[i * dim + c] - mean;
                variance += d * d;
            }
            float inv = (float)(1.0 / Math.Sqrt(variance / dim + LayerNormEps));
            for (int c = 0; c < dim; c++)
            {
                result[i * dim + c] = (float)((x[i * dim + c] - mean) * inv) * weight[c] + bias[c];
            }
        }
        return result;
    }

    private static void MeanSpan(float[] values, int dim, (int Start, int End) span, Span<float> destination)
    {
        if (span.End <= span.Start)
        {
            throw new InvalidDataException("Question and option spans must cover at least one token.");
        }
        for (int t = span.Start; t < span.End; t++)
        {
            for (int c = 0; c < dim; c++)
            {
                destination[c] += values[t * dim + c];
            }
        }
        for (int c = 0; c < dim; c++)
        {
            destination[c] /= span.End - span.Start;
        }
    }

    private static void Gelu(float[] x)
    {
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(0.5 * x[i] * (1.0 + Erf(x[i] / Math.Sqrt(2.0))));
        }
    }

    private static void Normalize(float[] x)
    {
        double sum = 0;
        foreach (float v in x)
        {
            sum += (double)v * v;
        }
        float inv = (float)(1.0 / Math.Max(Math.Sqrt(sum), NormalizeEps));
        for (int i = 0; i < x.Length; i++)
        {
            x[i] *= inv;
        }
    }

    private static float CosineSimilarity(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (float)(dot / (Math.Max(Math.Sqrt(na), UnitEps) * Math.Max(Math.Sqrt(nb), UnitEps)));
    }

    // Abramowitz-Stegun 7.1.26 is too coarse for parity; use the series/continued-fraction split.
    private static double Erf(double x)
    {
        double ax = Math.Abs(x);
        double result;
        if (ax < 2.5)
        {
            double sum = ax, term = ax;
            for (int n = 1; n < 100; n++)
            {
                term *= -ax * ax / n;
                double add = term / (2 * n + 1);
                sum += add;
                if (Math.Abs(add) < 1e-17)
                {
                    break;
                }
            }
            result = 2.0 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double f = 0;
            for (int k = 60; k >= 1; k--)
            {
                f = k / 2.0 / (ax + f);
            }
            result = 1.0 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + f);
        }
        return x < 0 ? -result : result;
    }

    private float Scalar(string key) => ((float*)_w[key].DataPointer)[0];

    private float[] Weight(string key)
    {
        Tensor t = _w[key];
        return new Span<float>((void*)t.DataPointer, (int)t.ElementCount).ToArray();
    }
}
