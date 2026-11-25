using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.FastFrames;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The FastFrames KV refinement network and pixel decoder against loop transcriptions of their Python reference
/// (ml-fastframes a65c592), written here in double precision from the stacked <c>[layers, in, out]</c> checkpoint layout, on tiny
/// random weights and the CPU backend. They show the engine code computes what the reference's formulas say; they do not show the
/// released weights reproduce the paper's videos, which needs the real checkpoints and a reference dump.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class FastFramesRefinerDecoderTests
{
    private const int Tokens = 6;

    // ---------------------------------------------------------------- refiner

    private static readonly FastFramesKvRefinerConfig TinyRefiner = new(Dim: 16, Heads: 2, Layers: 3, AdapterDim: 16,
        StemBlocks: 2, StemHeads: 2, StemMlpMultiplier: 4, HeadMlpMultiplier: 2, FrequencyDim: 32);

    [Fact]
    public void TheRefinerMatchesItsReferenceFormulas()
    {
        FastFramesKvRefinerConfig c = TinyRefiner;
        Dictionary<string, Tensor> weights = RefinerWeights(c, seed: 5);
        Random rng = new(6);
        double[][] stem = RandomRows(rng, Tokens, c.Dim);
        double[][][] noisyK = Enumerable.Range(0, c.Layers).Select(_ => RandomRows(rng, Tokens, c.Dim)).ToArray();
        double[][][] noisyV = Enumerable.Range(0, c.Layers).Select(_ => RandomRows(rng, Tokens, c.Dim)).ToArray();
        const float timestep = 833.3333f;

        (double[][][] expectedK, double[][][] expectedV) = RefinerReference(c, weights, stem, noisyK, noisyV, timestep);

        using CpuBackend backend = new();
        using FastFramesKvRefiner refiner = new(c);
        refiner.LoadWeights(weights);
        // The refiner must not alias the checkpoint: the references were built before loading, so disposing the source now
        // would leave any dangling view to read freed memory.
        foreach (Tensor t in weights.Values)
            t.Dispose();
        weights.Clear();
        using FastFramesKvRefiner.Conditioning conditioning = refiner.Prepare(backend, timestep);
        using Tensor stemTokens = ToTensor(stem);
        using Tensor phi = refiner.ComputePhi(backend, stemTokens, conditioning);
        using Tensor phiV = refiner.ProjectPhi(backend, phi);

        for (int l = 0; l < c.Layers; l++)
        {
            using Tensor k = ToTensor(noisyK[l]);
            using Tensor v = ToTensor(noisyV[l]);
            (Tensor refinedK, Tensor refinedV) = refiner.RepairLayer(backend, l, k, v, phiV, conditioning);
            using (refinedK)
            using (refinedV)
            {
                AssertClose(Flatten(expectedK[l]), refinedK.AsReadOnlySpan<float>(), $"layer {l} keys");
                AssertClose(Flatten(expectedV[l]), refinedV.AsReadOnlySpan<float>(), $"layer {l} values");
            }
        }
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    [Fact]
    public void TheRefinerLoadsBf16WeightsAndDetectsItsShape()
    {
        FastFramesKvRefinerConfig c = TinyRefiner;
        Dictionary<string, Tensor> weights = RefinerWeights(c, seed: 8, bf16: true);

        FastFramesKvRefinerConfig detected = FastFramesKvRefiner.DetectConfig(weights, heads: c.Heads);

        Assert.Equal(c.Dim, detected.Dim);
        Assert.Equal(c.Layers, detected.Layers);
        Assert.Equal(c.AdapterDim, detected.AdapterDim);
        Assert.Equal(c.StemBlocks, detected.StemBlocks);
        using FastFramesKvRefiner refiner = new(detected with { StemHeads = c.StemHeads, FrequencyDim = c.FrequencyDim });
        refiner.LoadWeights(weights);
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    private static (double[][][] K, double[][][] V) RefinerReference(FastFramesKvRefinerConfig c, Dictionary<string, Tensor> w,
        double[][] stem, double[][][] noisyK, double[][][] noisyV, float timestep)
    {
        int dA = c.AdapterDim;
        // timestep embedding: sinusoid -> Linear -> SiLU -> Linear
        double[] sinus = new double[c.FrequencyDim];
        for (int i = 0; i < c.FrequencyDim / 2; i++)
        {
            double angle = (double)timestep * Math.Pow(10000.0, -(double)i / (c.FrequencyDim / 2));
            sinus[i] = (float)Math.Cos(angle);
            sinus[c.FrequencyDim / 2 + i] = (float)Math.Sin(angle);
        }
        double[] e = Silu(LinearVec(sinus, w["t_embed.mlp.0.weight"], w["t_embed.mlp.0.bias"]));
        e = LinearVec(e, w["t_embed.mlp.2.weight"], w["t_embed.mlp.2.bias"]);
        double[] silE = Silu(e);

        double[][] x = LinearRows(stem, w["stem_in.weight"], w["stem_in.bias"]);
        for (int b = 0; b < c.StemBlocks; b++)
        {
            string p = $"stem_blocks.{b}.";
            double[] mod = LinearVec(silE, w[p + "mod.lin.weight"], w[p + "mod.lin.bias"]);
            double[] Chunk(int i) => mod[(i * dA)..((i + 1) * dA)];
            double[] sh1 = Chunk(0), sc1 = Chunk(1), g1 = Chunk(2), sh2 = Chunk(3), sc2 = Chunk(4), g2 = Chunk(5);
            double[][] h = x.Select(r => ModulateRow(LayerNorm(r, 1e-5), sh1, sc1)).ToArray();
            double[][] qkv = LinearRows(h, w[p + "qkv.weight"], w[p + "qkv.bias"]);
            int heads = c.StemHeads, hd = dA / heads;
            double[][] attn = new double[Tokens][];
            for (int t = 0; t < Tokens; t++)
                attn[t] = new double[dA];
            for (int head = 0; head < heads; head++)
            {
                for (int t = 0; t < Tokens; t++)
                {
                    double[] scores = new double[Tokens];
                    for (int s = 0; s < Tokens; s++)
                        for (int d = 0; d < hd; d++)
                            scores[s] += qkv[t][head * hd + d] * qkv[s][dA + head * hd + d] / Math.Sqrt(hd);
                    double max = scores.Max();
                    double[] probs = scores.Select(sv => Math.Exp(sv - max)).ToArray();
                    double sum = probs.Sum();
                    for (int s = 0; s < Tokens; s++)
                        for (int d = 0; d < hd; d++)
                            attn[t][head * hd + d] += probs[s] / sum * qkv[s][2 * dA + head * hd + d];
                }
            }
            double[][] proj = LinearRows(attn, w[p + "proj.weight"], w[p + "proj.bias"]);
            for (int t = 0; t < Tokens; t++)
                for (int d = 0; d < dA; d++)
                    x[t][d] += g1[d] * proj[t][d];
            double[][] h2 = x.Select(r => ModulateRow(LayerNorm(r, 1e-5), sh2, sc2)).ToArray();
            double[][] mlp = LinearRows(LinearRows(h2, w[p + "mlp.0.weight"], w[p + "mlp.0.bias"]).Select(r => r.Select(GeluTanh).ToArray()).ToArray(),
                w[p + "mlp.2.weight"], w[p + "mlp.2.bias"]);
            for (int t = 0; t < Tokens; t++)
                for (int d = 0; d < dA; d++)
                    x[t][d] += g2[d] * mlp[t][d];
        }
        double[][] phi = x.Select(r => AffineNorm(r, w["stem_norm.weight"], w["stem_norm.bias"], 1e-5)).ToArray();
        double[][] phiV = LinearRows(phi, w[$"heads_grouped.phi_proj.{dA}.weight"], null);

        const string g = "heads_grouped.gemms.0.";
        double[][][] outK = new double[c.Layers][][], outV = new double[c.Layers][][];
        for (int l = 0; l < c.Layers; l++)
        {
            double[][] pk = StackedRows(noisyK[l], w[g + "proj_k_w"], w[g + "proj_k_b"], l);
            double[][] pv = StackedRows(noisyV[l], w[g + "proj_v_w"], w[g + "proj_v_b"], l);
            double[][] joined = Enumerable.Range(0, Tokens).Select(t => pk[t].Concat(pv[t]).Concat(phiV[t]).ToArray()).ToArray();
            double[][] fused = StackedRows(joined, w[g + "fuse_w"], w[g + "fuse_b"], l);
            double[] modE = StackedRows([silE], w[g + "mod_w"], w[g + "mod_b"], l)[0];
            double[] shift = modE[..dA], scale = modE[dA..];
            double[][] hh = fused.Select(r => ModulateRow(LayerNorm(r, 1e-5), shift, scale)).ToArray();
            double[][] m0 = StackedRows(hh, w[g + "m0_w"], w[g + "m0_b"], l).Select(r => r.Select(GeluTanh).ToArray()).ToArray();
            double[][] m2 = StackedRows(m0, w[g + "m2_w"], w[g + "m2_b"], l);
            double[][] dK = StackedRows(m2, w[g + "dK_w"], w[g + "dK_b"], l);
            double[][] dV = StackedRows(m2, w[g + "dV_w"], w[g + "dV_b"], l);
            double[] logScale = Row(w[g + "v_logscale"], l);
            int heads = c.Heads, hd = c.HeadDim;
            outK[l] = new double[Tokens][];
            outV[l] = new double[Tokens][];
            for (int t = 0; t < Tokens; t++)
            {
                outK[l][t] = new double[c.Dim];
                outV[l][t] = new double[c.Dim];
                for (int head = 0; head < heads; head++)
                {
                    double xn = 0, rn = 0;
                    for (int d = 0; d < hd; d++)
                    {
                        double kv = noisyK[l][t][head * hd + d] + dK[t][head * hd + d];
                        xn += kv * kv;
                        rn += noisyK[l][t][head * hd + d] * noisyK[l][t][head * hd + d];
                    }
                    double factor = Math.Sqrt(rn) / (Math.Sqrt(xn) + 1e-6);
                    for (int d = 0; d < hd; d++)
                        outK[l][t][head * hd + d] = (noisyK[l][t][head * hd + d] + dK[t][head * hd + d]) * factor;
                }
                for (int d = 0; d < c.Dim; d++)
                    outV[l][t][d] = (noisyV[l][t][d] + dV[t][d]) * Math.Exp(logScale[d]);
            }
        }
        return (outK, outV);
    }

    private static Dictionary<string, Tensor> RefinerWeights(FastFramesKvRefinerConfig c, int seed, bool bf16 = false)
    {
        Random rng = new(seed);
        int dA = c.AdapterDim, dim = c.Dim, L = c.Layers, mult = c.HeadMlpMultiplier;
        Dictionary<string, Tensor> w = [];
        void Add(string name, params long[] shape) => w[name] = RandomTensor(rng, bf16, 0.3f, shape);
        Add("t_embed.mlp.0.weight", dA, c.FrequencyDim); Add("t_embed.mlp.0.bias", dA);
        Add("t_embed.mlp.2.weight", dA, dA); Add("t_embed.mlp.2.bias", dA);
        Add("stem_in.weight", dA, dim); Add("stem_in.bias", dA);
        for (int b = 0; b < c.StemBlocks; b++)
        {
            string p = $"stem_blocks.{b}.";
            Add(p + "qkv.weight", 3 * dA, dA); Add(p + "qkv.bias", 3 * dA);
            Add(p + "proj.weight", dA, dA); Add(p + "proj.bias", dA);
            Add(p + "mlp.0.weight", dA * c.StemMlpMultiplier, dA); Add(p + "mlp.0.bias", dA * c.StemMlpMultiplier);
            Add(p + "mlp.2.weight", dA, dA * c.StemMlpMultiplier); Add(p + "mlp.2.bias", dA);
            Add(p + "mod.lin.weight", 6 * dA, dA); Add(p + "mod.lin.bias", 6 * dA);
        }
        Add("stem_norm.weight", dA); Add("stem_norm.bias", dA);
        Add($"heads_grouped.phi_proj.{dA}.weight", dA, dA);
        const string g = "heads_grouped.gemms.0.";
        Add(g + "proj_k_w", L, dim, dA); Add(g + "proj_k_b", L, 1, dA);
        Add(g + "proj_v_w", L, dim, dA); Add(g + "proj_v_b", L, 1, dA);
        Add(g + "fuse_w", L, 3 * dA, dA); Add(g + "fuse_b", L, 1, dA);
        Add(g + "mod_w", L, dA, 2 * dA); Add(g + "mod_b", L, 1, 2 * dA);
        Add(g + "m0_w", L, dA, dA * mult); Add(g + "m0_b", L, 1, dA * mult);
        Add(g + "m2_w", L, dA * mult, dA); Add(g + "m2_b", L, 1, dA);
        Add(g + "dK_w", L, dA, dim); Add(g + "dK_b", L, 1, dim);
        Add(g + "dV_w", L, dA, dim); Add(g + "dV_b", L, 1, dim);
        Add(g + "v_logscale", L, 1, dim);
        return w;
    }

    /// <summary>The key names and shapes in the headers of the four released files (read by range request from
    /// huggingface.co/apple/FastFrames-S and -L on 2026-11-24, no tensor data) are exactly what the loaders expect, so a rename or
    /// a shape change in the loaders, or a different checkpoint, fails here. S: refiner Dim 1536, 30 layers; L: Dim 5120, 40 layers.</summary>
    [Theory]
    [InlineData("s_refiner", 1536, 12, 30)]
    [InlineData("l_refiner", 5120, 40, 40)]
    public void TheReleasedRefinerHeadersMatchTheLoader(string file, int dim, int heads, int layers)
    {
        Dictionary<string, (string Dtype, long[] Shape)> released = ReleasedHeader(file);
        FastFramesKvRefinerConfig expected = new(dim, heads, layers);

        Assert.All(released.Values, v => Assert.Equal("BF16", v.Dtype));
        AssertShapes(FastFramesKvRefiner.ExpectedShapes(expected), released);
        using Tensor stemIn = new(new TensorShape(released["stem_in.weight"].Shape), DType.F32);
        using Tensor dK = new(new TensorShape(released["heads_grouped.gemms.0.dK_w"].Shape), DType.F32);
        Dictionary<string, Tensor> shapesOnly = new() { ["stem_in.weight"] = stemIn, ["heads_grouped.gemms.0.dK_w"] = dK };
        shapesOnly["stem_blocks.0.qkv.weight"] = stemIn;
        shapesOnly["stem_blocks.1.qkv.weight"] = stemIn;
        shapesOnly["stem_blocks.2.qkv.weight"] = stemIn;
        shapesOnly["stem_blocks.3.qkv.weight"] = stemIn;
        FastFramesKvRefinerConfig detected = FastFramesKvRefiner.DetectConfig(shapesOnly, heads);
        Assert.Equal(expected with { StemBlocks = 4 }, detected);
    }

    [Theory]
    [InlineData("s_decoder", 1536, false)]
    [InlineData("l_decoder", 5120, true)]
    public void TheReleasedDecoderHeadersMatchTheLoader(string file, int inputDim, bool prefixed)
    {
        Dictionary<string, (string Dtype, long[] Shape)> released = ReleasedHeader(file);
        FastFramesPixelDecoderConfig expected = new(Dim: 1536, InputDim: inputDim);

        Assert.All(released.Values, v => Assert.Equal("F32", v.Dtype));
        AssertShapes(FastFramesPixelDecoder.ExpectedShapes(expected, prefixed), released);
    }

    private static void AssertShapes(IReadOnlyDictionary<string, long[]> expected, Dictionary<string, (string Dtype, long[] Shape)> released)
    {
        Assert.Equal(expected.Keys.Order(StringComparer.Ordinal), released.Keys.Order(StringComparer.Ordinal));
        foreach ((string name, long[] shape) in expected)
            Assert.True(shape.SequenceEqual(released[name].Shape), $"{name}: loader expects [{string.Join(",", shape)}], the released file has [{string.Join(",", released[name].Shape)}]");
    }

    private static Dictionary<string, (string Dtype, long[] Shape)> ReleasedHeader(string file)
    {
        string path = Path.Combine(AppContext.BaseDirectory, "Fixtures", "fastframes_released_headers.json");
        using System.Text.Json.JsonDocument document = System.Text.Json.JsonDocument.Parse(File.ReadAllText(path));
        Dictionary<string, (string, long[])> result = [];
        foreach (System.Text.Json.JsonProperty tensor in document.RootElement.GetProperty(file).EnumerateObject())
            result[tensor.Name] = (tensor.Value.GetProperty("dtype").GetString()!, tensor.Value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64()).ToArray());
        return result;
    }

    // ---------------------------------------------------------------- decoder

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheDecoderDecodedFrameByFrameMatchesTheWholeClipReference(bool bridged)
    {
        const int dim = 8, grid = 3, gridW = 2, frames = 4;
        int inputDim = bridged ? 12 : dim;
        FastFramesPixelDecoderConfig c = new(Dim: dim, InputDim: inputDim, LowBlocks: 2, HighBlocks: 1, FfnMultiplier: 4);
        Dictionary<string, Tensor> weights = DecoderWeights(c, seed: 11, prefix: bridged ? "pixel_head." : "");
        Random rng = new(12);
        int tokens = grid * gridW;
        double[][][] hidden = Enumerable.Range(0, frames).Select(_ => RandomRows(rng, tokens, inputDim)).ToArray();
        double[][] temb = RandomRows(rng, frames, inputDim);

        double[][][][] expected = DecoderReference(c, weights, hidden, temb, grid, gridW, bridged ? "pixel_head." : "");   // [frame][p][c] flattened HxW

        FastFramesPixelDecoderConfig detected = FastFramesPixelDecoder.DetectConfig(weights);
        Assert.Equal(c.Dim, detected.Dim);
        Assert.Equal(c.InputDim, detected.InputDim);
        Assert.Equal(c.LowBlocks, detected.LowBlocks);
        Assert.Equal(c.HighBlocks, detected.HighBlocks);

        using CpuBackend backend = new();
        using FastFramesPixelDecoder decoder = new(detected);
        decoder.LoadWeights(weights);
        foreach (Tensor t in weights.Values)
            t.Dispose();
        weights.Clear();
        using FastFramesPixelDecoder.State state = decoder.CreateState();
        for (int f = 0; f < frames; f++)
        {
            using Tensor h = ToTensor(hidden[f]);
            using Tensor e = ToTensor([temb[f]]);
            using Tensor pixels = decoder.DecodeFrame(backend, h, e, grid, gridW, state);
            Assert.Equal(new TensorShape([4L, 3, grid * 16, gridW * 16]), pixels.Shape);
            AssertClose(expected[f].SelectMany(p => p.SelectMany(ch => ch)).ToArray(), pixels.AsReadOnlySpan<float>(), $"frame {f} pixels");
        }
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    private static double[][][][] DecoderReference(FastFramesPixelDecoderConfig c, Dictionary<string, Tensor> w, double[][][] hidden,
        double[][] temb, int gridH, int gridW, string prefix)
    {
        int dim = c.Dim, frames = hidden.Length, tokens = gridH * gridW;
        double[][] shiftScale = new double[frames][];
        double[][][] x = new double[frames][][];
        double[] modulation = Row(w[prefix + "modulation"], 0);   // [2·dim]: shift, then scale
        for (int f = 0; f < frames; f++)
        {
            double[][] tokensIn = hidden[f];
            double[] e = temb[f];
            if (c.HasBridge)
            {
                tokensIn = LinearRows(tokensIn, w["bridge.weight"], null);
                e = LinearVec(e, w["bridge.weight"], null);
            }
            double[] shift = Enumerable.Range(0, dim).Select(i => modulation[i] + e[i]).ToArray();
            double[] scale = Enumerable.Range(0, dim).Select(i => modulation[dim + i] + e[i]).ToArray();
            x[f] = tokensIn.Select(r => ModulateRow(LayerNorm(r, 1e-6), shift, scale)).ToArray();
            x[f] = LinearRows(x[f].Select(r => AffineNorm(r, w[prefix + "entry_norm.weight"], w[prefix + "entry_norm.bias"], 1e-6)).ToArray(),
                w[prefix + "entry_pw.weight"], w[prefix + "entry_pw.bias"]);
        }
        string[] names = [.. Enumerable.Range(0, c.LowBlocks).Select(i => $"{prefix}conv_low.{i}."), .. Enumerable.Range(0, c.HighBlocks).Select(i => $"{prefix}conv_high.{i}.")];
        foreach (string p in names)
        {
            double[] dw = ToDouble(w[p + "dw.weight"]);   // [dim, 1, 3, 3, 3]
            double[] dwBias = ToDouble(w[p + "dw.bias"]);
            double[][][] next = new double[frames][][];
            for (int f = 0; f < frames; f++)
            {
                double[][] conv = new double[tokens][];
                for (int t = 0; t < tokens; t++)
                    conv[t] = new double[dim];
                for (int ch = 0; ch < dim; ch++)
                {
                    for (int y = 0; y < gridH; y++)
                    {
                        for (int xx = 0; xx < gridW; xx++)
                        {
                            double acc = dwBias[ch];
                            for (int kt = 0; kt < 3; kt++)
                            {
                                int src = f - 2 + kt;   // left-only temporal padding of 2
                                if (src < 0)
                                    continue;
                                for (int ky = 0; ky < 3; ky++)
                                {
                                    for (int kx = 0; kx < 3; kx++)
                                    {
                                        int sy = y + ky - 1, sx = xx + kx - 1;
                                        if (sy < 0 || sy >= gridH || sx < 0 || sx >= gridW)
                                            continue;
                                        acc += dw[(((ch * 3 + kt) * 3 + ky) * 3) + kx] * x[src][sy * gridW + sx][ch];
                                    }
                                }
                            }
                            conv[y * gridW + xx][ch] = acc;
                        }
                    }
                }
                double[][] normed = conv.Select(r => AffineNorm(r, w[p + "norm.weight"], w[p + "norm.bias"], 1e-6)).ToArray();
                double[][] up = LinearRows(normed, w[p + "pw1.weight"], w[p + "pw1.bias"]).Select(r => r.Select(GeluErf).ToArray()).ToArray();
                double[][] down = LinearRows(up, w[p + "pw2.weight"], w[p + "pw2.bias"]);
                next[f] = x[f].Select((r, t) => r.Select((v, d) => v + down[t][d]).ToArray()).ToArray();
            }
            x = next;
        }
        double[][][][] result = new double[frames][][][];
        int height = gridH * 16, width = gridW * 16;
        for (int f = 0; f < frames; f++)
        {
            double[][] outTokens = LinearRows(x[f], w[prefix + "fc_out.weight"], w[prefix + "fc_out.bias"]);
            result[f] = new double[4][][];
            for (int p = 0; p < 4; p++)
            {
                result[f][p] = new double[3][];
                for (int ch = 0; ch < 3; ch++)
                    result[f][p][ch] = new double[height * width];
            }
            for (int h = 0; h < gridH; h++)
                for (int ww = 0; ww < gridW; ww++)
                    for (int p = 0; p < 4; p++)
                        for (int q = 0; q < 16; q++)
                            for (int r = 0; r < 16; r++)
                                for (int ch = 0; ch < 3; ch++)
                                    result[f][p][ch][(h * 16 + q) * width + ww * 16 + r] = outTokens[h * gridW + ww][((p * 16 + q) * 16 + r) * 3 + ch];
        }
        return result;
    }

    private static Dictionary<string, Tensor> DecoderWeights(FastFramesPixelDecoderConfig c, int seed, string prefix)
    {
        Random rng = new(seed);
        int dim = c.Dim, ffn = dim * c.FfnMultiplier;
        Dictionary<string, Tensor> w = [];
        void Add(string name, params long[] shape) => w[name] = RandomTensor(rng, false, 0.3f, shape);
        if (c.HasBridge)
            Add("bridge.weight", dim, c.InputDim);
        Add(prefix + "modulation", 1, 2, dim);
        Add(prefix + "entry_norm.weight", dim); Add(prefix + "entry_norm.bias", dim);
        Add(prefix + "entry_pw.weight", dim, dim, 1, 1, 1); Add(prefix + "entry_pw.bias", dim);
        for (int i = 0; i < c.LowBlocks + c.HighBlocks; i++)
        {
            string p = $"{prefix}{(i < c.LowBlocks ? "conv_low." + i : "conv_high." + (i - c.LowBlocks))}.";
            Add(p + "dw.weight", dim, 1, 3, 3, 3); Add(p + "dw.bias", dim);
            Add(p + "norm.weight", dim); Add(p + "norm.bias", dim);
            Add(p + "pw1.weight", ffn, dim, 1, 1, 1); Add(p + "pw1.bias", ffn);
            Add(p + "pw2.weight", dim, ffn, 1, 1, 1); Add(p + "pw2.bias", dim);
        }
        Add(prefix + "fc_out.weight", 4 * 16 * 16 * 3, dim); Add(prefix + "fc_out.bias", 4 * 16 * 16 * 3);
        return w;
    }

    [Fact]
    public void ARefinerLoadsOnceAndRefusesWorkBeforeItHasWeights()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RefinerWeights(TinyRefiner, seed: 9);
        FastFramesKvRefiner refiner = new(TinyRefiner);
        Assert.Throws<InvalidOperationException>(() => refiner.Prepare(backend, 833f));
        refiner.LoadWeights(weights);
        Assert.Throws<InvalidOperationException>(() => refiner.LoadWeights(weights));
        refiner.Dispose();
        Assert.Throws<ObjectDisposedException>(() => refiner.Prepare(backend, 833f));
        foreach (Tensor t in weights.Values)
            t.Dispose();
    }

    [Fact]
    public void ADecoderRefusesAFrameOfAnotherGridOrAWrongEmbedding()
    {
        FastFramesPixelDecoderConfig c = new(Dim: 8, InputDim: 8, LowBlocks: 1, HighBlocks: 1, FfnMultiplier: 4);
        Dictionary<string, Tensor> weights = DecoderWeights(c, seed: 13, prefix: "");
        using CpuBackend backend = new();
        using FastFramesPixelDecoder decoder = new(c);
        decoder.LoadWeights(weights);
        foreach (Tensor t in weights.Values)
            t.Dispose();
        weights.Clear();
        Random rng = new(14);
        using Tensor first = ToTensor(RandomRows(rng, 6, 8));
        using Tensor embedding = ToTensor([RandomRows(rng, 1, 8)[0]]);
        using FastFramesPixelDecoder.State state = decoder.CreateState();
        decoder.DecodeFrame(backend, first, embedding, 3, 2, state).Dispose();

        // The same token count on a 2x3 grid is still a different grid: refused before anything is written.
        using Tensor second = ToTensor(RandomRows(rng, 6, 8));
        Assert.Throws<ArgumentException>(() => decoder.DecodeFrame(backend, second, embedding, 2, 3, state));
        using Tensor wide = ToTensor([RandomRows(rng, 1, 16)[0]]);
        Assert.Throws<ArgumentException>(() => decoder.DecodeFrame(backend, first, wide, 3, 2, state));
        decoder.DecodeFrame(backend, second, embedding, 3, 2, state).Dispose();
        state.Dispose();
        Assert.Throws<ObjectDisposedException>(() => decoder.DecodeFrame(backend, first, embedding, 3, 2, state));
    }

    [Fact]
    public void ADisposedDecoderRefusesWork()
    {
        FastFramesPixelDecoder decoder = new(new FastFramesPixelDecoderConfig());
        decoder.Dispose();
        Assert.Throws<ObjectDisposedException>(() => decoder.CreateState());
    }

    [Fact]
    public void ARefinerAndDecoderRefuseHalfPrecisionInputs()
    {
        using CpuBackend backend = new();
        Dictionary<string, Tensor> weights = RefinerWeights(TinyRefiner, seed: 15);
        using FastFramesKvRefiner refiner = new(TinyRefiner);
        refiner.LoadWeights(weights);
        foreach (Tensor t in weights.Values)
            t.Dispose();
        using Tensor half = new(new TensorShape(Tokens, TinyRefiner.Dim), DType.BF16);
        using Tensor phi = new(new TensorShape(Tokens, TinyRefiner.AdapterDim), DType.F32);
        using FastFramesKvRefiner.Conditioning conditioning = refiner.Prepare(backend, 833f);
        Assert.Throws<ArgumentException>(() => refiner.RepairLayer(backend, 0, half, half, phi, conditioning));

        FastFramesPixelDecoderConfig c = new(Dim: 8, InputDim: 8, LowBlocks: 1, HighBlocks: 1, FfnMultiplier: 4);
        Dictionary<string, Tensor> decoderWeights = DecoderWeights(c, seed: 16, prefix: "");
        using FastFramesPixelDecoder decoder = new(c);
        decoder.LoadWeights(decoderWeights);
        foreach (Tensor t in decoderWeights.Values)
            t.Dispose();
        using FastFramesPixelDecoder.State state = decoder.CreateState();
        using Tensor hidden = new(new TensorShape(6, 8), DType.BF16);
        using Tensor embedding = new(new TensorShape(1, 8), DType.F32);
        Assert.Throws<ArgumentException>(() => decoder.DecodeFrame(backend, hidden, embedding, 3, 2, state));
    }

    [Fact]
    public void AFailedLoadLeavesNothingBehindAndTheCheckpointCanBeLoadedAgain()
    {
        Dictionary<string, Tensor> refinerWeights = RefinerWeights(TinyRefiner, seed: 17);
        Dictionary<string, Tensor> partialRefiner = new(refinerWeights);
        partialRefiner.Remove("stem_norm.bias");
        using FastFramesKvRefiner refiner = new(TinyRefiner);
        Assert.Throws<ArgumentException>(() => refiner.LoadWeights(partialRefiner));
        refiner.LoadWeights(refinerWeights);
        foreach (Tensor t in refinerWeights.Values)
            t.Dispose();

        FastFramesPixelDecoderConfig c = new(Dim: 8, InputDim: 8, LowBlocks: 1, HighBlocks: 1, FfnMultiplier: 4);
        Dictionary<string, Tensor> decoderWeights = DecoderWeights(c, seed: 18, prefix: "");
        Dictionary<string, Tensor> partialDecoder = new(decoderWeights);
        partialDecoder.Remove("fc_out.bias");
        using FastFramesPixelDecoder decoder = new(c);
        Assert.Throws<ArgumentException>(() => decoder.LoadWeights(partialDecoder));
        decoder.LoadWeights(decoderWeights);
        foreach (Tensor t in decoderWeights.Values)
            t.Dispose();
    }

    // ---------------------------------------------------------------- reference math (double precision)

    private static double[] LinearVec(double[] x, Tensor weight, Tensor? bias)
    {
        int outDim = (int)weight.Shape[0], inDim = (int)(weight.ElementCount / outDim);
        double[] w = ToDouble(weight), b = bias is null ? new double[outDim] : ToDouble(bias);
        double[] y = new double[outDim];
        for (int o = 0; o < outDim; o++)
        {
            double acc = b[o];
            for (int i = 0; i < inDim; i++)
                acc += w[o * inDim + i] * x[i];
            y[o] = acc;
        }
        return y;
    }

    private static double[][] LinearRows(double[][] x, Tensor weight, Tensor? bias) => x.Select(r => LinearVec(r, weight, bias)).ToArray();

    /// <summary>x @ W[layer] + b[layer] for a stacked [layers, in, out] weight (the checkpoint layout) and [layers, 1, out] bias.</summary>
    private static double[][] StackedRows(double[][] x, Tensor weight, Tensor bias, int layer)
    {
        int inDim = (int)weight.Shape[1], outDim = (int)weight.Shape[2];
        double[] w = ToDouble(weight), b = ToDouble(bias);
        return x.Select(row =>
        {
            double[] y = new double[outDim];
            for (int o = 0; o < outDim; o++)
            {
                double acc = b[layer * outDim + o];
                for (int i = 0; i < inDim; i++)
                    acc += row[i] * w[(layer * inDim + i) * outDim + o];
                y[o] = acc;
            }
            return y;
        }).ToArray();
    }

    private static double[] Row(Tensor stacked, int layer)
    {
        double[] all = ToDouble(stacked);
        int width = all.Length / (int)stacked.Shape[0];
        return all[(layer * width)..((layer + 1) * width)];
    }

    private static double[] LayerNorm(double[] x, double eps)
    {
        double mean = x.Average(), variance = x.Select(v => (v - mean) * (v - mean)).Average();
        return x.Select(v => (v - mean) / Math.Sqrt(variance + eps)).ToArray();
    }

    private static double[] AffineNorm(double[] x, Tensor weight, Tensor bias, double eps)
    {
        double[] n = LayerNorm(x, eps), w = ToDouble(weight), b = ToDouble(bias);
        return n.Select((v, i) => v * w[i] + b[i]).ToArray();
    }

    private static double[] ModulateRow(double[] normed, double[] shift, double[] scale) =>
        normed.Select((v, i) => v * (1 + scale[i]) + shift[i]).ToArray();

    private static double[] Silu(double[] x) => x.Select(v => v / (1 + Math.Exp(-v))).ToArray();

    private static double GeluTanh(double x) => 0.5 * x * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x)));

    private static double GeluErf(double x) => 0.5 * x * (1 + Erf(x / Math.Sqrt(2)));

    private static double Erf(double x)
    {
        // Abramowitz-Stegun 7.1.26 is too coarse for a 1e-4 comparison; use the series/continued-fraction split.
        double ax = Math.Abs(x);
        double result;
        if (ax < 3)
        {
            double sum = ax, term = ax;
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
            for (int k = 60; k >= 1; k--)
                f = k / 2.0 / (ax + f);
            result = 1 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / (ax + f);
        }
        return x < 0 ? -result : result;
    }

    // ---------------------------------------------------------------- helpers

    private static Tensor RandomTensor(Random rng, bool bf16, float scale, long[] shape)
    {
        Tensor f32 = new(new TensorShape(shape), DType.F32);
        Span<float> span = f32.AsSpan<float>();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        if (!bf16)
            return f32;
        Tensor converted = f32.CastTo(DType.BF16);
        f32.Dispose();
        return converted;
    }

    private static double[] ToDouble(Tensor t)
    {
        using Tensor f32 = t.DType == DType.F32 ? t.Reshape(t.Shape) : t.CastTo(DType.F32);
        return f32.AsReadOnlySpan<float>().ToArray().Select(v => (double)v).ToArray();
    }

    private static double[][] RandomRows(Random rng, int rows, int cols) =>
        Enumerable.Range(0, rows).Select(_ => Enumerable.Range(0, cols).Select(_ => (rng.NextDouble() * 2 - 1) * 0.8).ToArray()).ToArray();

    private static Tensor ToTensor(double[][] rows)
    {
        Tensor t = new(new TensorShape(rows.Length, rows[0].Length), DType.F32);
        Span<float> span = t.AsSpan<float>();
        for (int r = 0; r < rows.Length; r++)
            for (int c = 0; c < rows[r].Length; c++)
                span[r * rows[r].Length + c] = (float)rows[r][c];
        return t;
    }

    private static double[] Flatten(double[][] rows) => rows.SelectMany(r => r).ToArray();

    private static void AssertClose(double[] expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0, error = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{what} holds a non-finite value");
            scale += expected[i] * expected[i];
            error += (expected[i] - actual[i]) * (expected[i] - actual[i]);
        }
        Assert.True(scale > 1e-3, $"{what}: the reference is all zeros, so the comparison proves nothing");
        double relative = Math.Sqrt(error / Math.Max(scale, 1e-30));
        Assert.True(relative < 2e-4, $"{what}: relative RMS error {relative:E2} against the reference formulas");
    }
}
