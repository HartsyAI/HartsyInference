using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.CoLa3D;
using Xunit;

namespace HartsyInference.ThreeD.Tests.CoLa3D;

/// <summary>Checks one block against a scalar transcription of the Python <c>_forward_impl</c> rope path, so head layout,
/// RoPE orientation and the RMS gains are verified by value rather than only by invariance.</summary>
public sealed class CoLa3DBlockTests
{
    private const float Tolerance = 2e-4f;

    [Fact]
    public void Block_MatchesTheScalarTranscriptionOfTheRopePath()
    {
        CoLa3DConfig config = CoLa3DTestSupport.Tiny(CoLa3DConfig.QuerySourceKind.Fixed);
        Dictionary<string, Tensor> weights = CoLa3DTestSupport.RandomWeights(config, seed: 21);
        int c = config.HiddenDim, np = 5, nq = 3;
        Random random = new(22);
        float[] p = RandomArray(random, np * c), q = RandomArray(random, nq * c);
        int[] pPos = [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7];
        int[] qPos = [2, 5, 6];
        try
        {
            using CpuBackend backend = new();
            using CoLa3DRopeTable pRope = CoLa3DRope.Build(CoLa3DRope.Plan(config.HeadDim), ToFloat(pPos), np);
            float[] qPositions = qPos.SelectMany(v => new[] { (float)v, v, v }).ToArray();
            using CoLa3DRopeTable qRope = CoLa3DRope.Build(CoLa3DRope.Plan(config.HeadDim), qPositions, nq);
            using CoLa3DBlock block = new(weights, 0, config);

            Tensor pIn = new(new TensorShape(1, np, c), DType.F32);
            Tensor qIn = new(new TensorShape(1, nq, c), DType.F32);
            p.AsSpan().CopyTo(pIn.AsSpan<float>());
            q.AsSpan().CopyTo(qIn.AsSpan<float>());
            (Tensor pOut, Tensor qOut) = block.Forward(backend, pIn, qIn, pRope, qRope, CoLa3DRope.RotaryDim(config.HeadDim));

            Reference reference = new(weights, 0, config, pRope, qRope);
            (float[] expectedP, float[] expectedQ) = reference.Run(p, np, q, nq);
            AssertClose(expectedP, pOut.AsReadOnlySpan<float>());
            AssertClose(expectedQ, qOut.AsReadOnlySpan<float>());
            pOut.Dispose();
            qOut.Dispose();
        }
        finally
        {
            CoLa3DTestSupport.Dispose(weights.Values);
        }
    }

    private static float[] RandomArray(Random random, int n)
    {
        float[] values = new float[n];
        for (int i = 0; i < n; i++)
            values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static float[] ToFloat(int[] values) => values.Select(v => (float)v).ToArray();

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= Tolerance * MathF.Max(1f, MathF.Abs(expected[i])),
                $"Element {i}: expected {expected[i]}, got {actual[i]}.");
    }

    /// <summary>Row-major scalar code for one block, written from the Python: every step follows its order.</summary>
    private sealed class Reference
    {
        private const float Eps = 1e-6f;
        private readonly IReadOnlyDictionary<string, Tensor> _w;
        private readonly string _p;
        private readonly int _c, _h, _d, _pairs;
        private readonly float[] _pCos, _pSin, _qCos, _qSin;

        internal Reference(IReadOnlyDictionary<string, Tensor> w, int layer, CoLa3DConfig config, CoLa3DRopeTable pRope, CoLa3DRopeTable qRope)
        {
            _w = w;
            _p = $"backbone.blocks.{layer}.";
            _c = config.HiddenDim;
            _h = config.NumHeads;
            _d = config.HeadDim;
            _pairs = CoLa3DRope.RotaryDim(config.HeadDim) / 2;
            _pCos = pRope.Cos.AsReadOnlySpan<float>().ToArray();
            _pSin = pRope.Sin.AsReadOnlySpan<float>().ToArray();
            _qCos = qRope.Cos.AsReadOnlySpan<float>().ToArray();
            _qSin = qRope.Sin.AsReadOnlySpan<float>().ToArray();
        }

        internal (float[] P, float[] Q) Run(float[] p0, int np, float[] q0, int nq)
        {
            int c = _c;
            // Query self-attention.
            float[] qn = Ln(q0, nq, "qSelfNorm");
            float[] qkv = Lin(qn, nq, "qSelfQkv", 3 * c);
            float[] qq = Cols(qkv, nq, 3 * c, 0, c), qk = Cols(qkv, nq, 3 * c, c, c), qv = Cols(qkv, nq, 3 * c, 2 * c, c);
            Rope(qq, nq, _qCos, _qSin);
            Rope(qk, nq, _qCos, _qSin);
            float[] self = Lin(Attn(qq, nq, qk, nq, qv), nq, "qSelfOut", c);
            float[] q1 = Add(q0, self);

            // Queries read points.
            float[] qr = Lin(Ln(q1, nq, "qReadNorm"), nq, "qReadQuery", c);
            Rope(qr, nq, _qCos, _qSin);
            float[] kv = Lin(Ln(p0, np, "pReadNorm"), np, "pReadKv", 2 * c);
            float[] pk = Cols(kv, np, 2 * c, 0, c), pv = Cols(kv, np, 2 * c, c, c);
            Rope(pk, np, _pCos, _pSin);
            float[] read = Lin(Attn(qr, nq, pk, np, pv), nq, "qReadOut", c);
            float[] q2 = Add(q1, read);

            // Points read the updated queries.
            float[] pq = Lin(Ln(p0, np, "pWriteNorm"), np, "pWriteQuery", c);
            Gain(pq, np, "pWriteQkNorm.gamma");
            Rope(pq, np, _pCos, _pSin);
            float[] kvq = Lin(q2, nq, "qWriteKv", 2 * c);
            float[] qk2 = Cols(kvq, nq, 2 * c, 0, c), qv2 = Cols(kvq, nq, 2 * c, c, c);
            Gain(qk2, nq, "qWriteKNorm.gamma");
            Rope(qk2, nq, _qCos, _qSin);
            float[] write = Lin(Attn(pq, np, qk2, nq, qv2), np, "pWriteOut", c);
            float[] p2 = Add(p0, write);

            // Feed-forward on both streams.
            float[] p3 = Add(p2, Ffn(p2, np, "pFfn"));
            float[] q3 = Add(q2, Ffn(q2, nq, "qFfn"));
            return (p3, q3);
        }

        private float[] Ffn(float[] x, int rows, string prefix)
        {
            float[] hidden = Lin(Ln(x, rows, prefix + "Norm"), rows, prefix + "Up", (int)_w[_p + prefix + "Up.weight"].Shape[0]);
            for (int i = 0; i < hidden.Length; i++)
                hidden[i] = Gelu(hidden[i]);
            return Lin(hidden, rows, prefix + "Down", _c);
        }

        private float[] Ln(float[] x, int rows, string name)
        {
            float[] g = W(name + ".weight"), b = W(name + ".bias");
            float[] o = new float[rows * _c];
            for (int r = 0; r < rows; r++)
            {
                double mean = 0, var = 0;
                for (int i = 0; i < _c; i++)
                    mean += x[r * _c + i];
                mean /= _c;
                for (int i = 0; i < _c; i++)
                    var += (x[r * _c + i] - mean) * (x[r * _c + i] - mean);
                var /= _c;
                for (int i = 0; i < _c; i++)
                    o[r * _c + i] = (float)((x[r * _c + i] - mean) / Math.Sqrt(var + Eps)) * g[i] + b[i];
            }
            return o;
        }

        private float[] Lin(float[] x, int rows, string name, int outDim)
        {
            float[] weight = W(name + ".weight"), bias = W(name + ".bias");
            int inDim = weight.Length / outDim;
            float[] o = new float[rows * outDim];
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < outDim; j++)
                {
                    double sum = bias[j];
                    for (int i = 0; i < inDim; i++)
                        sum += x[r * inDim + i] * weight[j * inDim + i];
                    o[r * outDim + j] = (float)sum;
                }
            return o;
        }

        /// <summary>Per-head RMS gain: <c>x / rms(x) · gamma[h, d]</c>, which is <c>F.normalize · gamma · sqrt(D)</c>.</summary>
        private void Gain(float[] x, int rows, string gammaName)
        {
            float[] gamma = W(gammaName);
            for (int r = 0; r < rows; r++)
                for (int h = 0; h < _h; h++)
                {
                    int at = (r * _h + h) * _d;
                    double sq = 0;
                    for (int i = 0; i < _d; i++)
                        sq += x[at + i] * x[at + i];
                    float inv = (float)(1.0 / Math.Sqrt(sq / _d));
                    for (int i = 0; i < _d; i++)
                        x[at + i] = x[at + i] * inv * gamma[h * _d + i];
                }
        }

        /// <summary>Interleaved RoPE on <c>[rows, H, D]</c>: pair <c>j</c> of each head rotates by the table's angle <c>j</c>.</summary>
        private void Rope(float[] x, int rows, float[] cos, float[] sin)
        {
            for (int r = 0; r < rows; r++)
                for (int h = 0; h < _h; h++)
                    for (int j = 0; j < _pairs; j++)
                    {
                        int at = (r * _h + h) * _d + 2 * j;
                        float cj = cos[r * _pairs + j], sj = sin[r * _pairs + j];
                        float a = x[at], b = x[at + 1];
                        x[at] = a * cj - b * sj;
                        x[at + 1] = a * sj + b * cj;
                    }
        }

        /// <summary>Softmax attention per head over flat <c>[rows, H·D]</c> operands, scale <c>1/sqrt(D)</c>.</summary>
        private float[] Attn(float[] q, int lq, float[] k, int lk, float[] v)
        {
            float[] o = new float[lq * _c];
            float scale = 1f / MathF.Sqrt(_d);
            for (int h = 0; h < _h; h++)
                for (int i = 0; i < lq; i++)
                {
                    float[] score = new float[lk];
                    float max = float.NegativeInfinity;
                    for (int j = 0; j < lk; j++)
                    {
                        float dot = 0;
                        for (int t = 0; t < _d; t++)
                            dot += q[i * _c + h * _d + t] * k[j * _c + h * _d + t];
                        score[j] = dot * scale;
                        max = MathF.Max(max, score[j]);
                    }
                    float total = 0;
                    for (int j = 0; j < lk; j++)
                    {
                        score[j] = MathF.Exp(score[j] - max);
                        total += score[j];
                    }
                    for (int t = 0; t < _d; t++)
                    {
                        float acc = 0;
                        for (int j = 0; j < lk; j++)
                            acc += score[j] / total * v[j * _c + h * _d + t];
                        o[i * _c + h * _d + t] = acc;
                    }
                }
            return o;
        }

        private static float[] Cols(float[] x, int rows, int width, int offset, int count)
        {
            float[] o = new float[rows * count];
            for (int r = 0; r < rows; r++)
                Array.Copy(x, r * width + offset, o, r * count, count);
            return o;
        }

        private static float[] Add(float[] a, float[] b)
        {
            float[] o = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
                o[i] = a[i] + b[i];
            return o;
        }

        private static float Gelu(float x) =>
            0.5f * x * (1f + MathF.Tanh(MathF.Sqrt(2f / MathF.PI) * (x + 0.044715f * x * x * x)));

        private float[] W(string name) => _w[_p + name].AsReadOnlySpan<float>().ToArray();
    }
}
