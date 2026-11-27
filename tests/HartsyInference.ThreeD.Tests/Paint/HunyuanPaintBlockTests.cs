using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Models.Hunyuan3D.Paint;

namespace HartsyInference.ThreeD.Tests.Paint;

/// <summary>Unit lane: the 2.5D block against a double-precision host transcription of the upstream forward (material attention, reference attention with the
/// concatenated per-material values viewed as (heads, materials * headDim), rotary multiview attention, DINO attention, GEGLU). The transcription shares this
/// repository's reading of the Python; it is not a parity reference.</summary>
public sealed unsafe class HunyuanPaintBlockTests
{
    private const int C = 32, Heads = 2, HeadDim = 16, Views = 2, Tokens = 3, RefTokens = 5, Text = 4, DinoTokens = 3, Cross = 16;

    private static readonly string[] Materials = ["albedo", "mr"];
    private readonly Random _random = new(21);

    private double[][] Matrix(int rows, int cols, double scale) => Enumerable.Range(0, rows).Select(_ => Enumerable.Range(0, cols).Select(_ => (_random.NextDouble() * 2 - 1) * scale).ToArray()).ToArray();

    private static Tensor ToTensor(double[][] m)
    {
        Tensor t = new(new TensorShape(m.Length, m[0].Length), DType.F32);
        for (int r = 0; r < m.Length; r++)
            for (int c = 0; c < m[0].Length; c++)
                ((float*)t.DataPointer)[r * m[0].Length + c] = (float)m[r][c];
        return t;
    }

    private static Tensor ToTensor(double[] v)
    {
        Tensor t = new(new TensorShape(v.Length), DType.F32);
        for (int i = 0; i < v.Length; i++)
            ((float*)t.DataPointer)[i] = (float)v[i];
        return t;
    }

    private static Tensor Tokens3(double[][][] x)
    {
        Tensor t = new(new TensorShape(x.Length, x[0].Length, x[0][0].Length), DType.F32);
        int i = 0;
        foreach (double[][] a in x)
            foreach (double[] b in a)
                foreach (double v in b)
                    ((float*)t.DataPointer)[i++] = (float)v;
        return t;
    }

    [Fact]
    public void Forward_MatchesTheHostTranscriptionOfTheUpstreamBlock()
    {
        HunyuanPaintConfig cfg = HunyuanPaintTestWeights.TinyConfig(v21: true);
        Dictionary<string, Tensor> w = [];
        Dictionary<string, double[][]> m = [];
        void Mat(string key, int rows, int cols, double scale = 0.2) { m[key] = Matrix(rows, cols, scale); w[key] = ToTensor(m[key]); }
        void Vec(string key, int n, double center, double scale) { double[] v = Enumerable.Range(0, n).Select(_ => center + (_random.NextDouble() * 2 - 1) * scale).ToArray(); m[key] = [v]; w[key] = ToTensor(v); }
        const string p = "b", t = p + ".transformer";
        foreach (string n in new[] { "norm1", "norm2", "norm3" })
        {
            Vec($"{t}.{n}.weight", C, 1, 0.2); Vec($"{t}.{n}.bias", C, 0, 0.1);
        }
        foreach ((string prefix, int kv) in new[] { ($"{t}.attn1", C), ($"{t}.attn2", Cross), ($"{p}.attn_multiview", C), ($"{p}.attn_refview", C), ($"{p}.attn_dino", Cross) })
        {
            Mat($"{prefix}.to_q.weight", C, C); Mat($"{prefix}.to_k.weight", C, kv); Mat($"{prefix}.to_v.weight", C, kv);
            Mat($"{prefix}.to_out.0.weight", C, C); Vec($"{prefix}.to_out.0.bias", C, 0, 0.1);
        }
        Mat($"{t}.attn1.processor.to_q_mr.weight", C, C); Mat($"{t}.attn1.processor.to_k_mr.weight", C, C); Mat($"{t}.attn1.processor.to_v_mr.weight", C, C);
        Mat($"{t}.attn1.processor.to_out_mr.0.weight", C, C); Vec($"{t}.attn1.processor.to_out_mr.0.bias", C, 0, 0.1);
        Mat($"{p}.attn_refview.processor.to_v_mr.weight", C, C);
        Mat($"{p}.attn_refview.processor.to_out_mr.0.weight", C, C); Vec($"{p}.attn_refview.processor.to_out_mr.0.bias", C, 0, 0.1);
        Mat($"{t}.ff.net.0.proj.weight", C * 8, C, 0.15); Vec($"{t}.ff.net.0.proj.bias", C * 8, 0, 0.1);
        Mat($"{t}.ff.net.2.weight", C, C * 4, 0.15); Vec($"{t}.ff.net.2.bias", C, 0, 0.1);

        double[][][][] x = Materials.Select(_ => Enumerable.Range(0, Views).Select(_ => Matrix(Tokens, C, 1)).ToArray()).ToArray();
        double[][] cond = Matrix(RefTokens, C, 1);
        double[][][] text = Materials.Select(_ => Matrix(Text, Cross, 1)).ToArray();
        double[][] dino = Matrix(DinoTokens, Cross, 1);
        int[] voxels = Enumerable.Range(0, Views * Tokens * 3).Select(_ => _random.Next(0, 64)).ToArray();
        (float[] cosF, float[] sinF) = HunyuanPaintRope.Tables(voxels, HeadDim);
        double[][] cos = Enumerable.Range(0, Views * Tokens).Select(r => Enumerable.Range(0, HeadDim).Select(c => (double)cosF[r * HeadDim + c]).ToArray()).ToArray();
        double[][] sin = Enumerable.Range(0, Views * Tokens).Select(r => Enumerable.Range(0, HeadDim).Select(c => (double)sinF[r * HeadDim + c]).ToArray()).ToArray();

        double[][][][] expected = Reference(m, x, cond, text, dino, cos, sin, referenceScale: 1.0, multiviewScale: 0.7);

        HunyuanPaintBlock block = new(cfg, C, Heads, extended: true, "b");
        block.LoadWeights(w, p);
        using CpuBackend backend = new();
        (Tensor Cos, Tensor Sin) tables = HunyuanPaintRope.TableTensors(voxels, HeadDim);
        Tensor condTensor = ToTensor(new[] { cond.SelectMany(r => r).ToArray() });
        condTensor = condTensor.Reshape(new TensorShape(1, RefTokens, C));
        HunyuanPaintBlockState state = new()
        {
            Embeds = new() { ["b"] = condTensor },
            MultiviewScale = 0.7f,
            Text = text.Select(a => Tokens3([a])).ToArray(),
            Dino = Tokens3([dino]),
            Rotation = new Dictionary<int, (Tensor, Tensor)> { [Views * Tokens] = tables },
        };
        Tensor[] actual = block.Forward(backend, x.Select(Tokens3).ToArray(), state);

        double maxDiff = 0;
        for (int mat = 0; mat < 2; mat++)
            for (int v = 0; v < Views; v++)
                for (int tok = 0; tok < Tokens; tok++)
                    for (int c = 0; c < C; c++)
                        maxDiff = Math.Max(maxDiff, Math.Abs(expected[mat][v][tok][c] - ((float*)actual[mat].DataPointer)[(v * Tokens + tok) * C + c]));
        Assert.True(maxDiff < 2e-4, $"max difference {maxDiff}");
    }

    private static double[][][][] Reference(Dictionary<string, double[][]> m, double[][][][] x, double[][] cond, double[][][] text, double[][] dino,
        double[][] cos, double[][] sin, double referenceScale, double multiviewScale)
    {
        const string t = "b.transformer", p = "b";
        double[][][][] y = x.Select(a => a.Select(b => b.Select(c => (double[])c.Clone()).ToArray()).ToArray()).ToArray();
        double[][][][] h1 = new double[2][][][];
        string[] suffix = ["", "_mr"];
        for (int mat = 0; mat < 2; mat++)
        {
            h1[mat] = y[mat].Select(view => view.Select(Norm1).ToArray()).ToArray();
            string q = mat == 0 ? $"{t}.attn1.to_q.weight" : $"{t}.attn1.processor.to_q_mr.weight";
            string k = mat == 0 ? $"{t}.attn1.to_k.weight" : $"{t}.attn1.processor.to_k_mr.weight";
            string v = mat == 0 ? $"{t}.attn1.to_v.weight" : $"{t}.attn1.processor.to_v_mr.weight";
            string o = mat == 0 ? $"{t}.attn1.to_out.0" : $"{t}.attn1.processor.to_out_mr.0";
            for (int view = 0; view < Views; view++)
                y[mat][view] = Add(y[mat][view], Attend(h1[mat][view], h1[mat][view], m[q], m[k], m[v], m[$"{o}.weight"], m[$"{o}.bias"][0], null, null));
        }

        // Reference attention: queries are the albedo rows of every view; values are the channel concatenation of both materials' projections.
        double[][] queryRows = h1[0].SelectMany(r => r).ToArray();
        double[][] kRef = MatMul(cond, m[$"{p}.attn_refview.to_k.weight"]);
        double[][] qRef = MatMul(queryRows, m[$"{p}.attn_refview.to_q.weight"]);
        double[][] vCat = MatMul(cond, m[$"{p}.attn_refview.to_v.weight"]).Zip(MatMul(cond, m[$"{p}.attn_refview.processor.to_v_mr.weight"]), (a, b) => a.Concat(b).ToArray()).ToArray();
        double[][] refOut = new double[queryRows.Length][];
        for (int r = 0; r < refOut.Length; r++)
            refOut[r] = new double[2 * C];
        for (int head = 0; head < Heads; head++)
        {
            double[][] scores = qRef.Select(qr => kRef.Select(kr => Enumerable.Range(0, HeadDim).Sum(d => qr[head * HeadDim + d] * kr[head * HeadDim + d]) / Math.Sqrt(HeadDim)).ToArray()).ToArray();
            for (int r = 0; r < scores.Length; r++)
            {
                double[] prob = Softmax(scores[r]);
                for (int d = 0; d < 2 * HeadDim; d++)
                    refOut[r][head * 2 * HeadDim + d] = prob.Select((pr, j) => pr * vCat[j][head * 2 * HeadDim + d]).Sum();
            }
        }
        for (int mat = 0; mat < 2; mat++)
        {
            double[][] merged = refOut.Select(row => Enumerable.Range(0, Heads).SelectMany(h => row.Skip(h * 2 * HeadDim + mat * HeadDim).Take(HeadDim)).ToArray()).ToArray();
            string o = mat == 0 ? $"{p}.attn_refview.to_out.0" : $"{p}.attn_refview.processor.to_out_mr.0";
            double[][] projected = AddBias(MatMul(merged, m[$"{o}.weight"]), m[$"{o}.bias"][0]);
            for (int view = 0; view < Views; view++)
                y[mat][view] = Add(y[mat][view], projected.Skip(view * Tokens).Take(Tokens).Select(r => r.Select(v => v * referenceScale).ToArray()).ToArray());
        }

        // Multiview attention with the rotary encoding, one material at a time.
        for (int mat = 0; mat < 2; mat++)
        {
            double[][] joined = h1[mat].SelectMany(r => r).ToArray();
            double[][] result = Attend(joined, joined, m[$"{p}.attn_multiview.to_q.weight"], m[$"{p}.attn_multiview.to_k.weight"], m[$"{p}.attn_multiview.to_v.weight"],
                m[$"{p}.attn_multiview.to_out.0.weight"], m[$"{p}.attn_multiview.to_out.0.bias"][0], cos, sin);
            for (int view = 0; view < Views; view++)
                y[mat][view] = Add(y[mat][view], result.Skip(view * Tokens).Take(Tokens).Select(r => r.Select(v => v * multiviewScale).ToArray()).ToArray());
        }

        for (int mat = 0; mat < 2; mat++)
            for (int view = 0; view < Views; view++)
            {
                double[][] n2 = y[mat][view].Select(r => Norm(r, m[$"{t}.norm2.weight"][0], m[$"{t}.norm2.bias"][0])).ToArray();
                y[mat][view] = Add(y[mat][view], Attend(n2, text[mat], m[$"{t}.attn2.to_q.weight"], m[$"{t}.attn2.to_k.weight"], m[$"{t}.attn2.to_v.weight"],
                    m[$"{t}.attn2.to_out.0.weight"], m[$"{t}.attn2.to_out.0.bias"][0], null, null));
                y[mat][view] = Add(y[mat][view], Attend(n2, dino, m[$"{p}.attn_dino.to_q.weight"], m[$"{p}.attn_dino.to_k.weight"], m[$"{p}.attn_dino.to_v.weight"],
                    m[$"{p}.attn_dino.to_out.0.weight"], m[$"{p}.attn_dino.to_out.0.bias"][0], null, null));
                double[][] n3 = y[mat][view].Select(r => Norm(r, m[$"{t}.norm3.weight"][0], m[$"{t}.norm3.bias"][0])).ToArray();
                double[][] proj = AddBias(MatMul(n3, m[$"{t}.ff.net.0.proj.weight"]), m[$"{t}.ff.net.0.proj.bias"][0]);
                double[][] gated = proj.Select(r => Enumerable.Range(0, C * 4).Select(i => r[i] * Gelu(r[C * 4 + i])).ToArray()).ToArray();
                y[mat][view] = Add(y[mat][view], AddBias(MatMul(gated, m[$"{t}.ff.net.2.weight"]), m[$"{t}.ff.net.2.bias"][0]));
            }
        _ = suffix;
        return y;

        double[] Norm1(double[] r) => Norm(r, m[$"{t}.norm1.weight"][0], m[$"{t}.norm1.bias"][0]);
    }

    private static double[][] Attend(double[][] qIn, double[][] kvIn, double[][] wq, double[][] wk, double[][] wv, double[][] wo, double[] bo, double[][]? cos, double[][]? sin)
    {
        double[][] q = MatMul(qIn, wq), k = MatMul(kvIn, wk), v = MatMul(kvIn, wv);
        if (cos is not null && sin is not null)
        {
            q = Rotate(q, cos!, sin!);
            k = Rotate(k, cos!, sin!);
        }
        double[][] merged = new double[q.Length][];
        for (int r = 0; r < q.Length; r++)
            merged[r] = new double[C];
        for (int head = 0; head < Heads; head++)
            for (int r = 0; r < q.Length; r++)
            {
                double[] prob = Softmax(k.Select(kr => Enumerable.Range(0, HeadDim).Sum(d => q[r][head * HeadDim + d] * kr[head * HeadDim + d]) / Math.Sqrt(HeadDim)).ToArray());
                for (int d = 0; d < HeadDim; d++)
                    merged[r][head * HeadDim + d] = prob.Select((pr, j) => pr * v[j][head * HeadDim + d]).Sum();
            }
        return AddBias(MatMul(merged, wo), bo);
    }

    // Interleaved pairs per head: (a, b) -> (a cos - b sin, b cos + a sin) with the row's table entries for that channel pair.
    private static double[][] Rotate(double[][] x, double[][] cos, double[][] sin) =>
        x.Select((row, r) => row.Select((value, c) =>
        {
            int head = c / HeadDim, within = c % HeadDim, pair = within / 2 * 2;
            double a = row[head * HeadDim + pair], b = row[head * HeadDim + pair + 1];
            return within % 2 == 0 ? a * cos[r][within] - b * sin[r][within] : b * cos[r][within] + a * sin[r][within];
        }).ToArray()).ToArray();

    private static double[][] MatMul(double[][] input, double[][] weight) => input.Select(r => weight.Select(wr => r.Zip(wr, (a, b) => a * b).Sum()).ToArray()).ToArray();
    private static double[][] AddBias(double[][] x, double[] b) => x.Select(r => r.Select((v, i) => v + b[i]).ToArray()).ToArray();
    private static double[][] Add(double[][] a, double[][] b) => a.Select((r, i) => r.Select((v, j) => v + b[i][j]).ToArray()).ToArray();

    private static double[] Norm(double[] r, double[] g, double[] b)
    {
        double mean = r.Average(), variance = r.Select(v => (v - mean) * (v - mean)).Average();
        return r.Select((v, i) => (v - mean) / Math.Sqrt(variance + 1e-5) * g[i] + b[i]).ToArray();
    }

    private static double[] Softmax(double[] s)
    {
        double max = s.Max();
        double[] e = s.Select(v => Math.Exp(v - max)).ToArray();
        return e.Select(v => v / e.Sum()).ToArray();
    }

    private static double Gelu(double x) => 0.5 * x * (1 + Erf(x / Math.Sqrt(2)));

    // Series for |x| < 3, continued fraction beyond: accurate far below the test tolerance.
    private static double Erf(double x)
    {
        double ax = Math.Abs(x), result;
        if (ax < 3)
        {
            double term = ax, sum = ax;
            for (int n = 1; n < 80; n++)
            {
                term *= -ax * ax / n;
                sum += term / (2 * n + 1);
            }
            result = 2 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double f = 0;
            for (int n = 60; n >= 1; n--)
                f = n / 2.0 / (ax + f);
            result = 1 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + f);
        }
        return x < 0 ? -result : result;
    }
}
