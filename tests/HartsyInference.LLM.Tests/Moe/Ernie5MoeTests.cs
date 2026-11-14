using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Moe;

/// <summary>ERNIE-5.5's latent MoE block on the CPU, against a host reference written from the routing and expert definitions alone. The
/// reference shares no code with <see cref="MoeFeedForward"/>: sigmoid scores, selection-only bias, top-k with lower-index ties,
/// renormalisation, the routed scale, SiTU-GLU, the latent RMS norm after the combine, and the shared expert added once.</summary>
public sealed class Ernie5MoeTests
{
    private const float Tol = 1e-4f;
    private const string Prefix = "L1";
    private const int Block = 128;
    private static readonly Dims Small = new(12, 8, 6, 6, 6, 2);

    [Theory]
    [InlineData(false, false, true, true, 1f, 5)]
    [InlineData(true, false, true, true, 1f, 5)]
    [InlineData(true, true, true, true, 3f, 5)]
    [InlineData(false, false, false, true, 3f, 5)]
    [InlineData(true, false, false, false, 1f, 5)]
    [InlineData(false, true, true, true, 1f, 1)]
    public void Forward_MatchesTheReference_AcrossRoutingAndNormVariants(bool split, bool learnedScale, bool normTopK, bool latentNorm, float scale,
        int tokens)
    {
        Weights w = Weights.Create(11, Small, new Flags(split, learnedScale, normTopK, latentNorm, scale));
        float[] x = Input(tokens * Small.Hidden, 5);
        using Loaded moe = new(w);
        AssertClose(Reference(w, x, tokens), moe.Run(x, tokens), Tol, "output");
    }

    [Fact]
    public void Bias_ChangesSelectionOnly_NotTheRoutedWeights()
    {
        Flags flags = new(NormTopK: false);
        Weights plain = Weights.Create(11, Small, flags);
        Weights biased = Weights.Create(11, Small, flags);
        biased.Bias = Param.Vector(100f, 0f, 0f, 0f, 0f, 0f);
        const int n = 5;
        float[] x = Input(n * Small.Hidden, 3);
        float[] expected = Reference(biased, x, n);
        Assert.True(MaxDiff(expected, Reference(plain, x, n)) > 1e-2f, "the bias must move the selection for this seed");
        using Loaded moe = new(biased);
        AssertClose(expected, moe.Run(x, n), Tol, "output");
    }

    [Fact]
    public void Ties_ResolveToTheLowerExpertIndex_AndRepeatExactly()
    {
        // Every router score is sigmoid(0) = 0.5, so the selection ties at every expert and the top-2 are experts 0 and 1.
        Weights w = Weights.Create(13, Small, new Flags());
        w.Router = Param.Zeros(Small.Experts, Small.Hidden);
        w.Bias = Param.Zeros(Small.Experts);
        const int n = 4;
        float[] x = Input(n * Small.Hidden, 15);
        using Loaded moe = new(w);
        float[] first = moe.Run(x, n);
        AssertClose(Reference(w, x, n), first, Tol, "output");
        Assert.Equal(first, moe.Run(x, n));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedExpert_IsAddedOnce_AndIsNotScaledByTheRoutedFactor(bool learnedScale)
    {
        // With every routed down projection zero, the block's output is the shared expert alone, whatever the routed scale.
        Weights w = Weights.Create(17, Small, new Flags(Split: true, LearnedScale: learnedScale, Scale: 3f));
        for (int e = 0; e < Small.Experts; e++) w.Down[e] = Param.Zeros(Small.Latent, Small.Inter);
        const int n = 5;
        float[] x = Input(n * Small.Hidden, 9);
        using Loaded moe = new(w);
        AssertClose(SharedOnly(w, x, n), moe.Run(x, n), Tol, "output");
    }

    [Fact]
    public void ZeroTokensRoutedToAnExpert_LeaveItsWeightsOut()
    {
        // Expert 5 never wins selection, so scaling its weights a thousandfold must not change a bit of the output.
        Flags flags = new(Split: true);
        Weights plain = Weights.Create(19, Small, flags);
        plain.Bias = Param.Vector(0f, 0f, 0f, 0f, 0f, -1000f);
        Weights loud = Weights.Create(19, Small, flags);
        loud.Bias = plain.Bias;
        loud.Gate[5] = Param.Scaled(loud.Gate[5], 1000f);
        loud.Up[5] = Param.Scaled(loud.Up[5], 1000f);
        loud.Down[5] = Param.Scaled(loud.Down[5], 1000f);
        const int n = 6;
        float[] x = Input(n * Small.Hidden, 21);
        using Loaded quiet = new(plain);
        using Loaded loudMoe = new(loud);
        float[] actual = quiet.Run(x, n);
        AssertClose(Reference(plain, x, n), actual, Tol, "output");
        AssertClose(actual, loudMoe.Run(x, n), 1e-6f, "output with the idle expert scaled");
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void AllTokensRoutedToTheSameExperts_EachRowKeepsItsOwnOutput(int topK)
    {
        // Experts 0..topK-1 take every token, so one expert (or two) holds all rows and the rest hold none.
        float[] bias = new float[Small.Experts];
        for (int e = 0; e < topK; e++) bias[e] = 1000f;
        Weights w = Weights.Create(23, Small with { TopK = topK }, new Flags());
        w.Bias = Param.Vector(bias);
        const int n = 6;
        float[] x = Input(n * Small.Hidden, 25);
        using Loaded moe = new(w);
        float[] batched = moe.Run(x, n);
        AssertClose(Reference(w, x, n), batched, Tol, "batched output");
        for (int t = 0; t < n; t++)
        {
            float[] row = x.AsSpan(t * Small.Hidden, Small.Hidden).ToArray();
            float[] expected = batched.AsSpan(t * Small.Hidden, Small.Hidden).ToArray();
            AssertClose(expected, moe.Run(row, 1), 1e-5f, $"row {t} against its own single-token decode");
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, true)]
    public void Prefill_MatchesSingleTokenDecodes(bool split, bool learnedScale)
    {
        Weights w = Weights.Create(29, Small, new Flags(Split: split, LearnedScale: learnedScale, Scale: 3f));
        const int n = 7;
        float[] x = Input(n * Small.Hidden, 31);
        using Loaded moe = new(w);
        float[] prefill = moe.Run(x, n);
        float[] decode = new float[n * Small.Hidden];
        for (int t = 0; t < n; t++)
        {
            float[] row = moe.Run(x.AsSpan(t * Small.Hidden, Small.Hidden).ToArray(), 1);
            Array.Copy(row, 0, decode, t * Small.Hidden, Small.Hidden);
        }
        AssertClose(decode, prefill, 1e-5f, "prefill against decodes");
        AssertClose(Reference(w, x, n), prefill, Tol, "prefill against the reference");
    }

    [Fact]
    public void Load_RefusesAMissingExpertKey()
    {
        Weights w = Weights.Create(37, Small, new Flags());
        Assert.Throws<KeyNotFoundException>(() => new Loaded(w, tensors => tensors.Remove($"{Prefix}.mlp.experts.2.up_proj.weight")));
    }

    [Theory]
    [InlineData("gate")]
    [InlineData("down")]
    [InlineData("router")]
    public void Load_RefusesAWeightWithTheWrongShape(string which)
    {
        Weights w = Weights.Create(41, Small, new Flags());
        switch (which)
        {
            case "gate": w.Gate[0] = w.Gate[0].Reshape(Small.Latent, Small.Inter); break;
            case "down": w.Down[0] = w.Down[0].Reshape(Small.Inter, Small.Latent); break;
            default: w.Router = w.Router.Reshape(Small.Hidden, Small.Experts); break;
        }
        string key = which switch
        {
            "gate" => $"{Prefix}.mlp.experts.0.gate_proj.weight",
            "down" => $"{Prefix}.mlp.experts.0.down_proj.weight",
            _ => $"{Prefix}.mlp.gate.weight",
        };
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => new Loaded(w).Dispose());
        Assert.Contains(key, error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("pre_expert_proj.weight")]
    [InlineData("post_expert_proj.weight")]
    public void Load_RefusesALatentProjectionWithTheWrongModelWidth_ByName(string leaf)
    {
        // The latent width matches, the model-width side does not: pre_expert_proj is [latent, model width], post_expert_proj the reverse.
        Weights w = Weights.Create(53, Small, new Flags());
        string key = $"{Prefix}.mlp.{leaf}";
        bool pre = leaf.StartsWith("pre", StringComparison.Ordinal);
        using Tensor wrong = pre
            ? new Tensor(new TensorShape(Small.Latent, Small.Hidden + 2), DType.F32)
            : new Tensor(new TensorShape(Small.Hidden + 2, Small.Latent), DType.F32);
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => new Loaded(w, tensors => tensors[key] = wrong));
        Assert.Contains(key, error.Message, StringComparison.Ordinal);
        string expected = pre ? $"[{Small.Latent}, {Small.Hidden}]" : $"[{Small.Hidden}, {Small.Latent}]";
        Assert.Contains($"expected {expected}", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("gate_proj.weight", 7, 12)]   // intermediate side wrong
    [InlineData("gate_proj.weight", 6, 10)]   // model-width side wrong
    [InlineData("up_proj.weight", 7, 12)]
    [InlineData("up_proj.weight", 6, 10)]
    [InlineData("down_proj.weight", 12, 7)]   // intermediate side wrong
    [InlineData("down_proj.weight", 10, 6)]   // model-width side wrong
    public void Load_RefusesAWrongSharedExpertShape_ByName(string leaf, int rows, int cols)
    {
        Weights w = Weights.Create(59, Small, new Flags());
        string key = $"{Prefix}.mlp.shared_expert.{leaf}";
        using Tensor wrong = new(new TensorShape(rows, cols), DType.F32);
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => new Loaded(w, tensors => tensors[key] = wrong));
        Assert.Contains(key, error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("gate.e_score_correction_bias")]
    [InlineData("gate.routed_scaling_factor_param")]
    public void Load_RefusesARouterVectorThatIsNotF32_ByName(string leaf)
    {
        Weights w = Weights.Create(47, Small, new Flags(LearnedScale: true));
        string key = $"{Prefix}.mlp.{leaf}";
        using Tensor half = new(new TensorShape(Small.Experts), DType.F16);
        InvalidDataException error = Assert.Throws<InvalidDataException>(() => new Loaded(w, tensors => tensors[key] = half));
        Assert.Contains(key, error.Message, StringComparison.Ordinal);
        Assert.Contains("F16", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Fp8Experts_DecodeThroughTheirBlockGrid_AcrossPartialBlocks()
    {
        // 140-wide experts leave a 12-wide partial block in both directions; the reference reads the dequantized twin of the same codes.
        Dims dims = new(260, 140, 140, 6, 6, 2);
        Weights w = Weights.Create(43, dims, new Flags(Split: true), fp8Experts: true);
        const int n = 4;
        float[] x = Input(n * dims.Hidden, 45);
        using Loaded moe = new(w);
        AssertClose(Reference(w, x, n), moe.Run(x, n), Tol, "fp8 output");
    }

    // The host reference. Everything is double, and each step follows the ERNIE definition in the comments.
    private static float[] Reference(Weights w, float[] x, int n)
    {
        Dims d = w.Dims;
        Flags f = w.Flags;
        float[] output = new float[n * d.Hidden];
        for (int t = 0; t < n; t++)
        {
            double[] token = Row(x, t, d.Hidden);
            // Sigmoid scores; split routing sums two sigmoids per expert.
            double[] score = new double[d.Experts];
            for (int k = 0; k < d.Experts; k++)
            {
                score[k] = Sigmoid(Dot(w.Router, k, token));
                if (f.Split) score[k] += Sigmoid(Dot(w.Router2!, k, token));
            }
            // The correction bias joins the selection score only; ties take the lower index.
            int[] picked = Enumerable.Range(0, d.Experts)
                .OrderByDescending(k => score[k] + w.Bias.Values[k])
                .ThenBy(k => k)
                .Take(d.TopK)
                .ToArray();
            double wsum = picked.Sum(k => score[k]);
            double[] latent = MatVec(w.LatentDown, token);
            double[] combined = new double[d.Latent];
            foreach (int k in picked)
            {
                double weight = f.NormTopK ? score[k] / wsum : score[k];
                weight *= f.LearnedScale ? w.RoutedScale!.Values[k] : f.Scale;
                double[] y = Ffn(w.Gate[k], w.Up[k], w.Down[k], latent);
                for (int i = 0; i < d.Latent; i++) combined[i] += weight * y[i];
            }
            if (f.LatentNorm) combined = RmsNorm(combined, w.LatentNormGain!.Values);
            double[] routed = MatVec(w.LatentUp, combined);
            double[] shared = Ffn(w.SharedGate, w.SharedUp, w.SharedDown, token);
            for (int i = 0; i < d.Hidden; i++) output[t * d.Hidden + i] = (float)(routed[i] + shared[i]);
        }
        return output;
    }

    private static float[] SharedOnly(Weights w, float[] x, int n)
    {
        float[] output = new float[n * w.Dims.Hidden];
        for (int t = 0; t < n; t++)
        {
            double[] shared = Ffn(w.SharedGate, w.SharedUp, w.SharedDown, Row(x, t, w.Dims.Hidden));
            for (int i = 0; i < w.Dims.Hidden; i++) output[t * w.Dims.Hidden + i] = (float)shared[i];
        }
        return output;
    }

    // SiTU-GLU: gate factor beta * tanh(g / beta) * sigmoid(g), value factor beta_lin * tanh(u / beta_lin), with beta 4 and beta_lin 25.
    private static double[] Ffn(Param gate, Param up, Param down, double[] input)
    {
        double[] g = MatVec(gate, input), u = MatVec(up, input);
        double[] a = new double[g.Length];
        for (int i = 0; i < g.Length; i++)
            a[i] = 4.0 * Math.Tanh(g[i] / 4.0) * Sigmoid(g[i]) * (25.0 * Math.Tanh(u[i] / 25.0));
        return MatVec(down, a);
    }

    private static double[] RmsNorm(double[] v, float[] gain)
    {
        double meanSquare = 0;
        foreach (double value in v) meanSquare += value * value;
        meanSquare /= v.Length;
        double inverse = 1.0 / Math.Sqrt(meanSquare + 1e-6);
        return v.Select((value, i) => value * inverse * gain[i]).ToArray();
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Dot(Param p, int row, double[] v)
    {
        int cols = (int)p.Shape[1];
        double sum = 0;
        for (int c = 0; c < cols; c++) sum += p.Values[row * cols + c] * v[c];
        return sum;
    }

    private static double[] MatVec(Param p, double[] v)
    {
        int rows = (int)p.Shape[0];
        double[] result = new double[rows];
        for (int r = 0; r < rows; r++) result[r] = Dot(p, r, v);
        return result;
    }

    private static double[] Row(float[] x, int t, int width) => x.AsSpan(t * width, width).ToArray().Select(v => (double)v).ToArray();

    private static float[] Input(int count, int seed)
    {
        Random rng = new(seed);
        float[] values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(rng.NextDouble() * 2 - 1);
        return values;
    }

    private static void AssertClose(float[] expected, float[] actual, float tol, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol, $"{label}[{i}]: expected {expected[i]}, actual {actual[i]} (tolerance {tol})");
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    private static Tensor BuildTensor(Param p, List<Tensor> owned)
    {
        if (p.Codes is null)
        {
            Tensor dense = new(new TensorShape(p.Shape), DType.F32);
            owned.Add(dense);
            p.Values.AsSpan().CopyTo(dense.AsSpan<float>());
            return dense;
        }
        long gridRows = (p.Shape[0] + Block - 1) / Block, gridCols = (p.Shape[1] + Block - 1) / Block;
        Tensor grid = new(new TensorShape(gridRows, gridCols), DType.F32);
        owned.Add(grid);
        p.Grid!.AsSpan().CopyTo(grid.AsSpan<float>());
        Tensor fp8 = new(new TensorShape(p.Shape), DType.F8E4M3);
        owned.Add(fp8);
        p.Codes.AsSpan().CopyTo(fp8.AsSpan<byte>());
        fp8.QuantInfo = new QuantWeightInfo { Format = "fp8_block_scale_inv", Fp8BlockScaleInv = grid, Fp8BlockSize = Block };
        return fp8;
    }

    // IEEE-style E4M3: 1 sign, 4 exponent bits (bias 7), 3 mantissa bits; exponent 0 is subnormal.
    private static float DecodeE4m3(byte b)
    {
        int sign = b >> 7, exponent = (b >> 3) & 0xF, mantissa = b & 0x7;
        double magnitude = exponent == 0 ? mantissa / 8.0 * Math.Pow(2, -6) : (1 + mantissa / 8.0) * Math.Pow(2, exponent - 7);
        return (float)(sign == 1 ? -magnitude : magnitude);
    }

    private sealed record Dims(int Hidden, int Latent, int Inter, int Shared, int Experts, int TopK);

    private sealed record Flags(bool Split = false, bool LearnedScale = false, bool NormTopK = true, bool LatentNorm = true, float Scale = 1f);

    // One weight: the F32 values the reference reads, and for a block-FP8 weight the E4M3 codes and their 128x128 grid, whose decoded values
    // are Values.
    private sealed record Param(long[] Shape, float[] Values, byte[]? Codes = null, float[]? Grid = null)
    {
        public static Param Vector(params float[] values) => new([values.Length], values);

        public static Param Zeros(params long[] shape) => new(shape, new float[Count(shape)]);

        public static Param Uniform(Random rng, double bound, params long[] shape)
        {
            float[] values = new float[Count(shape)];
            for (int i = 0; i < values.Length; i++) values[i] = (float)((rng.NextDouble() * 2 - 1) * bound);
            return new Param(shape, values);
        }

        public static Param Fp8(Random rng, long rows, long cols)
        {
            int count = (int)(rows * cols), gridCols = (int)((cols + Block - 1) / Block);
            byte[] codes = new byte[count];
            for (int i = 0; i < count; i++)
            {
                int exponent = 4 + rng.Next(5), mantissa = rng.Next(8), sign = rng.Next(2);
                codes[i] = (byte)((sign << 7) | (exponent << 3) | mantissa);
            }
            float[] grid = new float[(int)((rows + Block - 1) / Block) * gridCols];
            for (int i = 0; i < grid.Length; i++) grid[i] = (float)(0.01 + rng.NextDouble() * 0.04);
            float[] values = new float[count];
            for (int i = 0; i < count; i++)
            {
                int r = i / (int)cols, c = i % (int)cols;
                values[i] = DecodeE4m3(codes[i]) * grid[(r / Block) * gridCols + c / Block];
            }
            return new Param([rows, cols], values, codes, grid);
        }

        public static Param Scaled(Param p, float factor) => p with { Values = p.Values.Select(v => v * factor).ToArray() };

        public Param Reshape(long rows, long cols) => this with { Shape = [rows, cols] };

        private static int Count(long[] shape) => (int)shape.Aggregate(1L, (a, s) => a * s);
    }

    private sealed class Weights
    {
        public required Dims Dims { get; init; }
        public required Flags Flags { get; init; }
        public required Param Router { get; set; }
        public Param? Router2 { get; set; }
        public required Param Bias { get; set; }
        public Param? RoutedScale { get; set; }
        public required Param LatentDown { get; set; }
        public Param? LatentNormGain { get; set; }
        public required Param LatentUp { get; set; }
        public required Param[] Gate { get; set; }
        public required Param[] Up { get; set; }
        public required Param[] Down { get; set; }
        public required Param SharedGate { get; set; }
        public required Param SharedUp { get; set; }
        public required Param SharedDown { get; set; }

        public static Weights Create(int seed, Dims d, Flags f, bool fp8Experts = false)
        {
            Random rng = new(seed);
            Param[] gate = new Param[d.Experts], up = new Param[d.Experts], down = new Param[d.Experts];
            for (int e = 0; e < d.Experts; e++)
            {
                gate[e] = fp8Experts ? Param.Fp8(rng, d.Inter, d.Latent) : Param.Uniform(rng, 1.0 / Math.Sqrt(d.Latent), d.Inter, d.Latent);
                up[e] = fp8Experts ? Param.Fp8(rng, d.Inter, d.Latent) : Param.Uniform(rng, 1.0 / Math.Sqrt(d.Latent), d.Inter, d.Latent);
                down[e] = fp8Experts ? Param.Fp8(rng, d.Latent, d.Inter) : Param.Uniform(rng, 1.0 / Math.Sqrt(d.Inter), d.Latent, d.Inter);
            }
            return new Weights
            {
                Dims = d,
                Flags = f,
                Router = Param.Uniform(rng, 1.0 / Math.Sqrt(d.Hidden), d.Experts, d.Hidden),
                Router2 = f.Split ? Param.Uniform(rng, 1.0 / Math.Sqrt(d.Hidden), d.Experts, d.Hidden) : null,
                Bias = Param.Uniform(rng, 0.3, d.Experts),
                RoutedScale = f.LearnedScale ? RandomPositive(rng, d.Experts) : null,
                LatentDown = Param.Uniform(rng, 1.0 / Math.Sqrt(d.Hidden), d.Latent, d.Hidden),
                LatentNormGain = f.LatentNorm ? RandomPositive(rng, d.Latent) : null,
                LatentUp = Param.Uniform(rng, 1.0 / Math.Sqrt(d.Latent), d.Hidden, d.Latent),
                Gate = gate,
                Up = up,
                Down = down,
                SharedGate = Param.Uniform(rng, 1.0 / Math.Sqrt(d.Hidden), d.Shared, d.Hidden),
                SharedUp = Param.Uniform(rng, 1.0 / Math.Sqrt(d.Hidden), d.Shared, d.Hidden),
                SharedDown = Param.Uniform(rng, 1.0 / Math.Sqrt(d.Shared), d.Hidden, d.Shared),
            };
        }

        public MoeConfig Config() => new()
        {
            NumExperts = Dims.Experts,
            NumExpertsPerTok = Dims.TopK,
            MoeIntermediateSize = Dims.Inter,
            NormTopKProb = Flags.NormTopK,
            Scoring = MoeScoring.Sigmoid,
            SharedExpertIntermediateSize = Dims.Shared,
            Activation = ActivationKind.Situ,
            ActivationSituBeta = 4f,
            ActivationSituLinearBeta = 25f,
            RoutedScalingFactor = Flags.Scale,
            SplitFeatureRouting = Flags.Split,
            LearnedRoutedScale = Flags.LearnedScale,
            LatentSize = Dims.Latent,
            LatentNorm = Flags.LatentNorm ? MoeLatentNorm.AfterCombine : MoeLatentNorm.None,
            LatentNormEps = 1e-6f,
        };

        // The keys the checkpoint loader hands the block (Ernie5WeightNames.MoeKeys, under the engine's names).
        public Dictionary<string, Tensor> Tensors(List<Tensor> owned)
        {
            string mlp = $"{Prefix}.mlp";
            Dictionary<string, Tensor> t = new(StringComparer.Ordinal)
            {
                [$"{mlp}.gate.weight"] = BuildTensor(Router, owned),
                [$"{mlp}.gate.e_score_correction_bias"] = BuildTensor(Bias, owned),
                [$"{mlp}.pre_expert_proj.weight"] = BuildTensor(LatentDown, owned),
                [$"{mlp}.post_expert_proj.weight"] = BuildTensor(LatentUp, owned),
                [$"{mlp}.shared_expert.gate_proj.weight"] = BuildTensor(SharedGate, owned),
                [$"{mlp}.shared_expert.up_proj.weight"] = BuildTensor(SharedUp, owned),
                [$"{mlp}.shared_expert.down_proj.weight"] = BuildTensor(SharedDown, owned),
            };
            if (Flags.Split) t[$"{mlp}.gate.weight_1"] = BuildTensor(Router2!, owned);
            if (Flags.LearnedScale) t[$"{mlp}.gate.routed_scaling_factor_param"] = BuildTensor(RoutedScale!, owned);
            if (Flags.LatentNorm) t[$"{mlp}.pre_expert_norm.weight"] = BuildTensor(LatentNormGain!, owned);
            for (int e = 0; e < Dims.Experts; e++)
            {
                t[$"{mlp}.experts.{e}.gate_proj.weight"] = BuildTensor(Gate[e], owned);
                t[$"{mlp}.experts.{e}.up_proj.weight"] = BuildTensor(Up[e], owned);
                t[$"{mlp}.experts.{e}.down_proj.weight"] = BuildTensor(Down[e], owned);
            }
            return t;
        }

        private static Param RandomPositive(Random rng, long count)
        {
            float[] values = new float[count];
            for (int i = 0; i < values.Length; i++) values[i] = (float)(0.5 + rng.NextDouble());
            return new Param([count], values);
        }
    }

    // Loads a block from the weights; the tensors it built are disposed with it. The edit hook changes the loader's dictionary first.
    private sealed class Loaded : IDisposable
    {
        private readonly List<Tensor> _owned = [];
        private readonly CpuBackend _backend = new();
        private readonly int _hidden;
        private readonly MoeFeedForward _block;

        public Loaded(Weights w, Action<Dictionary<string, Tensor>>? edit = null)
        {
            _hidden = w.Dims.Hidden;
            try
            {
                Dictionary<string, Tensor> tensors = w.Tensors(_owned);
                edit?.Invoke(tensors);
                _block = new MoeFeedForward(w.Config(), _hidden, lowVram: false);
                _block.LoadWeights(tensors, Prefix);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public float[] Run(float[] x, int n)
        {
            using Tensor input = new(new TensorShape(1, n, _hidden), DType.F32);
            x.AsSpan().CopyTo(input.AsSpan<float>());
            using Tensor y = _block.Forward(_backend, input, n);
            return y.AsReadOnlySpan<float>().ToArray();
        }

        public void Dispose()
        {
            foreach (Tensor t in _owned) t.Dispose();
            _owned.Clear();
        }
    }
}
