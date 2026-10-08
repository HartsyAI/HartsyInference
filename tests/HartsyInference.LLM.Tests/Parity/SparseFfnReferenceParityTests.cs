using HartsyInference.Core.Backends;
using HartsyInference.Core.Moe;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.LLM.Transformer;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>
/// Parity of the generic reference layer path (<see cref="SparseFfnReference"/>) against the production
/// <see cref="MoeFeedForward"/> on identical F32 weights. Tolerance is the bounded contract from docs/MOE_ARCHITECTURE.md:
/// the combine order and the scalar dot products differ from the backend's, so exact equality is not expected.
/// </summary>
public sealed unsafe class SparseFfnReferenceParityTests
{
    private const float Tolerance = 1e-4f;
    private static uint _rng = 0x5EED51u;

    private static float Rand()
    {
        _rng ^= _rng << 13; _rng ^= _rng >> 17; _rng ^= _rng << 5;
        return ((_rng & 0xFFFF) / 65535f - 0.5f) * 0.5f;
    }

    /// <summary>Uniform values in roughly ±<paramref name="scale"/>. Weights use a larger scale so outputs are order-one, not vanishing.</summary>
    private static float[] RandomArray(int n, float scale = 1f)
    {
        float[] r = new float[n];
        for (int i = 0; i < n; i++) r[i] = Rand() * scale;
        return r;
    }

    private static Tensor Matrix(float[] v, int rows, int cols)
    {
        Tensor t = new(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < v.Length; i++) p[i] = v[i];
        return t;
    }

    private static Tensor Vector(float[] v)
    {
        Tensor t = new(new TensorShape(v.Length), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < v.Length; i++) p[i] = v[i];
        return t;
    }

    private static Tensor Rank3(float[] v, int n, int hidden)
    {
        Tensor t = new(new TensorShape(1, n, hidden), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < v.Length; i++) p[i] = v[i];
        return t;
    }

    private static float[] Read(Tensor t)
    {
        float[] r = new float[t.ElementCount];
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < r.Length; i++) r[i] = p[i];
        return r;
    }

    /// <summary>Router logits <c>x · Wᵀ</c> on the host, the input both paths route from.</summary>
    private static float[] Logits(float[] x, float[] router, int n, int e, int hidden)
    {
        float[] r = new float[n * e];
        for (int t = 0; t < n; t++)
            for (int j = 0; j < e; j++)
            {
                float acc = 0f;
                for (int d = 0; d < hidden; d++) acc += x[t * hidden + d] * router[j * hidden + d];
                r[t * e + j] = acc;
            }
        return r;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float max = 0f;
        for (int i = 0; i < a.Length; i++) max = MathF.Max(max, MathF.Abs(a[i] - b[i]));
        return max;
    }

    [Fact]
    public void SoftmaxRouted_ReferenceMatchesMoeFeedForward()
    {
        const int hidden = 16, inter = 24, e = 4, topK = 2, n = 5;
        const string p = "model.layers.0";
        float[] x = RandomArray(n * hidden);
        float[] router = RandomArray(e * hidden, 6f);
        float[][] gate = new float[e][], up = new float[e][], down = new float[e][];
        Dictionary<string, Tensor> w = new() { [$"{p}.mlp.gate.weight"] = Matrix(router, e, hidden) };
        for (int i = 0; i < e; i++)
        {
            gate[i] = RandomArray(inter * hidden, 6f);
            up[i] = RandomArray(inter * hidden, 6f);
            down[i] = RandomArray(hidden * inter, 6f);
            w[$"{p}.mlp.experts.{i}.gate_proj.weight"] = Matrix(gate[i], inter, hidden);
            w[$"{p}.mlp.experts.{i}.up_proj.weight"] = Matrix(up[i], inter, hidden);
            w[$"{p}.mlp.experts.{i}.down_proj.weight"] = Matrix(down[i], hidden, inter);
        }

        MoeConfig moe = new()
        {
            NumExperts = e, NumExpertsPerTok = topK, MoeIntermediateSize = inter, NormTopKProb = true, Scoring = MoeScoring.Softmax,
        };
        using CpuBackend backend = new();
        MoeFeedForward production = new(moe, hidden, lowVram: false);
        production.LoadWeights(w, p);
        using Tensor xIn = Rank3(x, n, hidden);
        using Tensor actual = production.Forward(backend, xIn, n);

        MoeLayerDescriptor layer = new MoeLayerDescriptor(
            new RouterDescriptor(e, topK, topK, MoeRouteScoring.Softmax, Renormalize: true, RenormEpsilon: 0f).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, gate[i], up[i], down[i]).Validated());
        float[] expected = new float[n * hidden];
        SparseFfnReference.Run(layer, prefill: false, x, n, Logits(x, router, n, e, hidden), experts, expected);

        Assert.True(expected.Any(static v => MathF.Abs(v) > 1e-2f), "Reference output is degenerate; the parity check would be vacuous.");
        float diff = MaxAbsDiff(expected, Read(actual));
        Assert.True(diff <= Tolerance, $"Reference diverges from MoeFeedForward by {diff:E3}");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void GroupedSigmoidWithSelectionBias_ReferenceMatchesMoeFeedForward()
    {
        const int hidden = 16, inter = 12, e = 8, topK = 2, n = 6;
        const string p = "model.layers.0";
        float[] x = RandomArray(n * hidden);
        float[] router = RandomArray(e * hidden, 6f);
        float[] bias = RandomArray(e);
        float[][] gate = new float[e][], up = new float[e][], down = new float[e][];
        Dictionary<string, Tensor> w = new()
        {
            [$"{p}.mlp.gate.weight"] = Matrix(router, e, hidden),
            [$"{p}.mlp.gate.e_score_correction_bias"] = Vector(bias),
        };
        for (int i = 0; i < e; i++)
        {
            gate[i] = RandomArray(inter * hidden, 6f);
            up[i] = RandomArray(inter * hidden, 6f);
            down[i] = RandomArray(hidden * inter, 6f);
            w[$"{p}.mlp.experts.{i}.gate_proj.weight"] = Matrix(gate[i], inter, hidden);
            w[$"{p}.mlp.experts.{i}.up_proj.weight"] = Matrix(up[i], inter, hidden);
            w[$"{p}.mlp.experts.{i}.down_proj.weight"] = Matrix(down[i], hidden, inter);
        }

        MoeConfig moe = new()
        {
            NumExperts = e, NumExpertsPerTok = topK, MoeIntermediateSize = inter, NormTopKProb = true,
            Scoring = MoeScoring.SigmoidLogitAdd, ExpertGroupCount = 2, ExpertGroupUsedCount = 1,
        };
        using CpuBackend backend = new();
        MoeFeedForward production = new(moe, hidden, lowVram: false);
        production.LoadWeights(w, p);
        using Tensor xIn = Rank3(x, n, hidden);
        using Tensor actual = production.Forward(backend, xIn, n);

        MoeLayerDescriptor layer = new MoeLayerDescriptor(
            new RouterDescriptor(e, topK, topK, MoeRouteScoring.Sigmoid, GroupCount: 2, GroupsKept: 1, Renormalize: true,
                RenormEpsilon: 1e-20f, HasSelectionBias: true).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, gate[i], up[i], down[i]).Validated());
        float[] expected = new float[n * hidden];
        SparseFfnReference.Run(layer, prefill: false, x, n, Logits(x, router, n, e, hidden), experts, expected, bias);

        Assert.True(expected.Any(static v => MathF.Abs(v) > 1e-2f), "Reference output is degenerate; the parity check would be vacuous.");
        float diff = MaxAbsDiff(expected, Read(actual));
        Assert.True(diff <= Tolerance, $"Grouped reference diverges from MoeFeedForward by {diff:E3}");

        foreach (Tensor t in w.Values) t.Dispose();
    }

    [Fact]
    public void ExpertProgramReference_SiluAndGeluMatchClosedForms()
    {
        Assert.Equal(0f, ExpertProgramReference.Activate(ExpertActivation.Silu, 0f));
        Assert.Equal(2f / (1f + MathF.Exp(-2f)), ExpertProgramReference.Activate(ExpertActivation.Silu, 2f), 6);
        Assert.Equal(0f, ExpertProgramReference.Activate(ExpertActivation.GeluTanh, 0f));
        Assert.Equal(0f, ExpertProgramReference.Activate(ExpertActivation.Relu, -3f));
        Assert.Equal(9f, ExpertProgramReference.Activate(ExpertActivation.ReluSquared, 3f), 6);
    }

    [Fact]
    public void Run_RejectsMissingBiasAndMismatchedWeights()
    {
        const int hidden = 4, inter = 3, e = 4;
        MoeLayerDescriptor biased = new MoeLayerDescriptor(
            new RouterDescriptor(e, 2, 2, MoeRouteScoring.Sigmoid, HasSelectionBias: true).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, new float[inter * hidden], new float[inter * hidden],
                new float[hidden * inter]));
        float[] x = new float[hidden], logits = new float[e], y = new float[hidden];

        Assert.Throws<ArgumentException>(() => SparseFfnReference.Run(biased, false, x, 1, logits, experts, y));

        List<F32ExpertWeights> wrong = experts.ToList();
        wrong[1] = new F32ExpertWeights(hidden, inter + 1, new float[(inter + 1) * hidden], new float[(inter + 1) * hidden],
                new float[hidden * (inter + 1)]);
        Assert.Throws<ArgumentException>(() => SparseFfnReference.Run(biased, false, x, 1, logits, wrong, y, new float[e]));
    }

    [Fact]
    public void ClampedProgram_MatchesAnIndependentScalarComputation()
    {
        // Every expert is routed (topK == E), so the output is the softmax-weighted sum of plain expert evaluations.
        // The expert math below is written out here, independent of ExpertProgramReference, with the clamp applied
        // exactly where DeepSeek-V4.1 applies it: gate capped above, up clamped to +-limit.
        const int hidden = 8, inter = 6, e = 3, n = 4;
        const float limit = 0.4f;
        float[] x = RandomArray(n * hidden);
        float[] router = RandomArray(e * hidden, 2f);
        float[][] gate = new float[e][], up = new float[e][], down = new float[e][];
        for (int i = 0; i < e; i++)
        {
            gate[i] = RandomArray(inter * hidden, 6f);
            up[i] = RandomArray(inter * hidden, 6f);
            down[i] = RandomArray(hidden * inter, 6f);
        }

        MoeLayerDescriptor layer = new MoeLayerDescriptor(
            new RouterDescriptor(e, e, e, MoeRouteScoring.Softmax, Renormalize: true, RenormEpsilon: 0f).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.SwigluClamped(limit)).Validated();
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, gate[i], up[i], down[i]).Validated());

        float[] logits = Logits(x, router, n, e, hidden);
        float[] actual = new float[n * hidden];
        SparseFfnReference.Run(layer, prefill: false, x, n, logits, experts, actual);

        float[] expected = new float[n * hidden];
        bool clampActive = false;
        for (int t = 0; t < n; t++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < e; j++) max = MathF.Max(max, logits[t * e + j]);
            float sum = 0f;
            float[] weight = new float[e];
            for (int j = 0; j < e; j++) { weight[j] = MathF.Exp(logits[t * e + j] - max); sum += weight[j]; }
            for (int j = 0; j < e; j++)
            {
                weight[j] /= sum;
                for (int d = 0; d < hidden; d++)
                {
                    float y = 0f;
                    for (int i = 0; i < inter; i++)
                    {
                        float g = 0f, u = 0f;
                        for (int c = 0; c < hidden; c++)
                        {
                            g += gate[j][i * hidden + c] * x[t * hidden + c];
                            u += up[j][i * hidden + c] * x[t * hidden + c];
                        }
                        clampActive |= g > limit || u > limit || u < -limit;
                        float gc = MathF.Min(g, limit);
                        float uc = Math.Clamp(u, -limit, limit);
                        y += down[j][d * inter + i] * (gc / (1f + MathF.Exp(-gc)) * uc);
                    }
                    expected[t * hidden + d] += weight[j] * y;
                }
            }
        }

        Assert.True(clampActive, "The clamp never fired; the test would not distinguish a missing clamp.");
        float diff = MaxAbsDiff(expected, actual);
        Assert.True(diff <= Tolerance, $"Clamped reference diverges from the independent computation by {diff:E3}");
    }

    [Fact]
    public void Run_RejectsAStrayBiasAndSharedExpertsExplicitly()
    {
        const int hidden = 4, inter = 3, e = 4;
        ExpertGroupDescriptor routed = new(e, new ExpertDescriptor(hidden, inter, DType.F32));
        MoeLayerDescriptor softmax = new MoeLayerDescriptor(
            new RouterDescriptor(e, 2, 2,
                    MoeRouteScoring.Softmax).Validated(), routed, Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, new float[inter * hidden], new float[inter * hidden],
                new float[hidden * inter]));
        float[] x = new float[hidden], logits = new float[e], y = new float[hidden];

        Assert.Throws<ArgumentException>(() => SparseFfnReference.Run(softmax, false, x, 1, logits, experts, y, new float[e]));

        MoeLayerDescriptor withShared = new MoeLayerDescriptor(
            new RouterDescriptor(e, 2, 2, MoeRouteScoring.Softmax).Validated(), routed,
            new ExpertGroupDescriptor(1, new ExpertDescriptor(hidden, inter, DType.F32)), SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        Assert.Throws<NotSupportedException>(() => SparseFfnReference.Run(withShared, false, x, 1, logits, experts, y));
    }

    [Fact]
    public void LogitSpaceBias_AndWideRouters_AreExplicitlyUnsupported()
    {
        const int hidden = 4, inter = 3, e = 4;
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, new float[inter * hidden], new float[inter * hidden],
                new float[hidden * inter]));
        float[] x = new float[hidden], logits = new float[e], y = new float[hidden];

        MoeLayerDescriptor logitSpace = new MoeLayerDescriptor(
            new RouterDescriptor(e, 2, 2, MoeRouteScoring.Sigmoid, HasSelectionBias: true, BiasSpace: SelectionBiasSpace.Logit).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        Assert.Throws<NotSupportedException>(() => SparseFfnReference.Run(logitSpace, false, x, 1, logits, experts, y, new float[e]));

        const int wide = MoeRouteArgs.MaxExperts + 1;
        List<F32ExpertWeights> many = [];
        for (int i = 0; i < wide; i++) many.Add(new F32ExpertWeights(hidden, inter, new float[inter * hidden], new float[inter * hidden],
                new float[hidden * inter]));
        MoeLayerDescriptor tooWide = new MoeLayerDescriptor(
            new RouterDescriptor(wide, 2, 2, MoeRouteScoring.Softmax).Validated(),
            new ExpertGroupDescriptor(wide, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        Assert.Throws<NotSupportedException>(() => SparseFfnReference.Run(tooWide, false, x, 1, new float[wide], many, y));
    }

    [Fact]
    public void TokenKindBias_SelectsTheAlternateExpertForKindOneOnly()
    {
        // One token, top-1 over three experts, so the selected expert is unambiguous and its output is the whole result.
        const int hidden = 8, inter = 6, e = 3;
        float[] x = RandomArray(hidden);
        float[] logits = [1.0f, 0.5f, 0.2f];
        float[] baseBias = [0f, 0f, 0f];
        float[] altBias = [0f, 0f, 5f];
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++)
            experts.Add(new F32ExpertWeights(hidden, inter, RandomArray(inter * hidden, 6f), RandomArray(inter * hidden, 6f),
                    RandomArray(hidden * inter, 6f)).Validated());

        MoeLayerDescriptor layer = new MoeLayerDescriptor(
            new RouterDescriptor(e, 1, 1, MoeRouteScoring.Softmax, Renormalize: true, RenormEpsilon: 0f, HasSelectionBias: true,
                    HasTokenKindBias: true).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();

        float[] kindZero = new float[hidden], kindOne = new float[hidden];
        SparseFfnReference.Run(layer, false, x, 1, logits, experts, kindZero, baseBias, altBias, [0]);
        SparseFfnReference.Run(layer, false, x, 1, logits, experts, kindOne, baseBias, altBias, [1]);

        float[] expertZero = new float[hidden], expertTwo = new float[hidden];
        ExpertProgramReference.Apply(ExpertProgram.Swiglu, experts[0], x, 1, expertZero);
        ExpertProgramReference.Apply(ExpertProgram.Swiglu, experts[2], x, 1, expertTwo);

        Assert.True(MaxAbsDiff(expertZero, expertTwo) > 1e-2f, "The two experts must differ for the test to mean anything.");
        Assert.True(MaxAbsDiff(kindZero, expertZero) <= Tolerance, "Kind 0 must select the base-bias expert.");
        Assert.True(MaxAbsDiff(kindOne, expertTwo) <= Tolerance, "Kind 1 must select the alternate-bias expert.");

        Assert.Throws<ArgumentException>(() => SparseFfnReference.Run(layer, false, x, 1, logits, experts, kindOne, baseBias));
    }

    [Fact]
    public void Overflow_CheckUsesOnlyTheSelectedPhase()
    {
        // A large prefill top-k must not reject a decode call whose own buffers are small.
        const int hidden = 2, inter = 1, e = 256;
        List<F32ExpertWeights> experts = [];
        for (int i = 0; i < e; i++) experts.Add(new F32ExpertWeights(hidden, inter, [1f, 0f], [1f, 0f], [1f, 1f]));
        MoeLayerDescriptor layer = new MoeLayerDescriptor(
            new RouterDescriptor(e, 1, 256, MoeRouteScoring.Softmax).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32)).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        float[] x = [0.5f, -0.25f], y = new float[hidden], logits = new float[e];
        logits[7] = 3f;

        SparseFfnReference.Run(layer, prefill: false, x, 1, logits, experts, y);

        Assert.Contains(y, static v => MathF.Abs(v) > 1e-3f);
    }

    [Fact]
    public void Run_RejectsAnOverrideWhoseWidthDiffersFromTheLayerInput()
    {
        const int hidden = 4, inter = 3, e = 2;
        ExpertDescriptor narrowOverride = new ExpertDescriptor(hidden * 2, inter, DType.F32);
        MoeLayerDescriptor layer = new MoeLayerDescriptor(
            new RouterDescriptor(e, 1, 1, MoeRouteScoring.Softmax).Validated(),
            new ExpertGroupDescriptor(e, new ExpertDescriptor(hidden, inter, DType.F32), new Dictionary<int,
                    ExpertDescriptor> { [1] = narrowOverride }).Validated(),
            Shared: null, SharedIsGated: false, ExpertProgram.Swiglu).Validated();
        List<F32ExpertWeights> experts =
        [
            new F32ExpertWeights(hidden, inter, new float[inter * hidden], new float[inter * hidden], new float[hidden * inter]),
            new F32ExpertWeights(hidden * 2, inter, new float[inter * hidden * 2], new float[inter * hidden * 2], new float[hidden * 2 * inter]),
        ];
        float[] x = new float[hidden], logits = new float[e], y = new float[hidden];

        Assert.Throws<ArgumentException>(() => SparseFfnReference.Run(layer, false, x, 1, logits, experts, y));
    }
}
