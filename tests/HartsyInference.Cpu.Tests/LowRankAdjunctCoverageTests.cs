using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Cpu.Tests;

/// <summary>Enumerates every GEMM-shaped entry a LoRA'd weight can reach and pins each to one of two outcomes:
/// it APPLIES the adjunct, or it REFUSES the weight by name. There is no third option, because the third option is a
/// generation that succeeds while the LoRA silently does part of its job — the failure the zero-match refusal in
/// <c>RecipeLoraMerge</c> exists to prevent, and one that no output inspection would catch.
/// <para>The CUDA twins (the resident-int8 chain, the fused GELU epilogue, the decode-shaped GEMVs, the grouped
/// projections and a captured step graph) are in <c>HartsyInference.Cuda.Tests</c> under GpuIntegration; the entries
/// here are the ones every backend shares through <see cref="IBackend"/>'s defaults.</para></summary>
public sealed class LowRankAdjunctCoverageTests
{
    private const int Rows = 6, Cols = 8, Rank = 2, Batch = 3;

    /// <summary>How an <see cref="IBackend"/> entry that can receive a LoRA'd tensor is expected to behave.</summary>
    private enum AdjunctCoverage
    {
        /// <summary>Accumulates the adjunct into its result — pinned by a [Fact] here or in the CUDA twin.</summary>
        Applies,

        /// <summary>Throws by name when handed a tensor carrying one, because it has no shape to apply it in.</summary>
        Refuses,

        /// <summary>Cannot receive one: its weight is a rank-1 norm or gate vector, and
        /// <c>LoraStack.RequireRank2AdjunctTarget</c> means an adjunct is never attached to those.</summary>
        NotATarget,
    }

    /// <summary>Every <see cref="IBackend"/> entry that takes a tensor a LoRA could have been attached to, and which
    /// of the two outcomes it owes. Adding a row is the point: the test below fails until a new entry appears here.</summary>
    private static readonly Dictionary<string, AdjunctCoverage> _coverage = new(StringComparer.Ordinal)
    {
        ["Linear"] = AdjunctCoverage.Applies,
        ["LinearWeightRows"] = AdjunctCoverage.Applies,
        ["LinearGelu"] = AdjunctCoverage.Applies,
        ["LinearHeadGated"] = AdjunctCoverage.Applies,
        ["LinearMulti"] = AdjunctCoverage.Applies,
        ["QuantizedMatMul"] = AdjunctCoverage.Applies,
        ["MatMul"] = AdjunctCoverage.Refuses,
        ["BatchedMatMul"] = AdjunctCoverage.Refuses,
        ["Conv2D"] = AdjunctCoverage.Refuses,
        ["Conv1d"] = AdjunctCoverage.Refuses,
        ["Conv3d"] = AdjunctCoverage.Refuses,
        ["Conv2dDepthwise"] = AdjunctCoverage.Refuses,
        ["ConvTranspose1d"] = AdjunctCoverage.Refuses,
        ["ConvTranspose2d"] = AdjunctCoverage.Refuses,
        ["GroupNorm"] = AdjunctCoverage.NotATarget,
        ["GroupNormSilu"] = AdjunctCoverage.NotATarget,
        ["LayerNorm"] = AdjunctCoverage.NotATarget,
        ["ChannelLayerNorm3d"] = AdjunctCoverage.NotATarget,
        ["RmsNorm"] = AdjunctCoverage.NotATarget,
        ["RmsNormAdd"] = AdjunctCoverage.NotATarget,
        ["RmsNormEmitQ8"] = AdjunctCoverage.NotATarget,
        ["AddRmsNorm"] = AdjunctCoverage.NotATarget,
        ["AddRmsNormEmitQ8"] = AdjunctCoverage.NotATarget,
    };

    /// <summary>The [Fact]s below are hand-written, so on their own they only prove the entries someone remembered.
    /// This closes that: a new <see cref="IBackend"/> method taking a <c>Tensor weight</c> fails the suite until it is
    /// classified above, which forces the applies-or-refuses decision at the moment the entry is added rather than
    /// the moment a LoRA looks weak.
    /// <para>It pins the classification, not the behaviour — the signatures vary too much to invoke generically, so
    /// the [Fact]s remain what verifies that an <c>Applies</c> entry really does.</para></summary>
    [Fact]
    public void EveryBackendEntryTakingAWeightIsClassified()
    {
        string[] found = typeof(IBackend)
            .GetMethods(BindingFlags.Public | BindingFlags.Instance)
            .Where(m => m.GetParameters().Any(p =>
                p.ParameterType == typeof(Tensor) && p.Name is "weight" or "quantWeight"))
            .Select(m => m.Name)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .ToArray();

        string[] unclassified = found.Where(n => !_coverage.ContainsKey(n)).ToArray();
        Assert.True(unclassified.Length == 0,
            $"IBackend gained {string.Join(", ", unclassified)}, which take a weight a LoRA can be attached to. "
            + "Classify each in _coverage as Applies, Refuses or NotATarget — and if it is Applies or Refuses, add "
            + "the [Fact] that proves it. A silent miss reads as 'the LoRA looks weak' and never as an error.");

        // Three entries the predicate cannot reach: the matmuls name their operands `a`/`b`, and LinearMulti carries
        // its weights inside a LinearOp span. They are asserted to still exist so a rename cannot leave a stale row.
        string[] weightNotInASignature = ["MatMul", "BatchedMatMul", "LinearMulti"];
        string[] stale = _coverage.Keys
            .Where(k => !found.Contains(k, StringComparer.Ordinal)
                && !weightNotInASignature.Contains(k, StringComparer.Ordinal))
            .ToArray();
        Assert.True(stale.Length == 0, $"_coverage names {string.Join(", ", stale)}, which IBackend no longer has.");
        foreach (string name in weightNotInASignature)
        {
            Assert.Contains(typeof(IBackend).GetMethods(), m => m.Name == name);
        }
    }


    /// <summary>Each element of the delta: rank · down · up, with the file carrying no alpha (scale 1.0).</summary>
    private const float Down = 0.5f, Up = 0.25f;

    [Fact]
    public void Linear_AppliesTheAdjunct()
    {
        AssertMatchesMergedWeight((backend, output, input, weight) => backend.Linear(output, input, weight, null),
            Rows, rowOffset: 0);
    }

    [Fact]
    public void LinearWeightRows_AppliesTheWindowedAdjunct()
    {
        AssertMatchesMergedWeight(
            (backend, output, input, weight) => backend.LinearWeightRows(output, input, weight, null, 2, 3),
            outColumns: 3, rowOffset: 2);
    }

    [Fact]
    public void LinearGelu_AppliesTheAdjunctBeforeTheActivation()
    {
        // Order matters and is invisible in the result's shape: GELU of (base + delta) is not GELU(base) + delta.
        using Harness harness = new Harness();
        using Tensor adjunctOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.LinearGelu(adjunctOut, harness.Input, harness.Patched, null);

        using Tensor mergedOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.LinearGelu(mergedOut, harness.Input, harness.Merged, null);

        AssertClose(mergedOut, adjunctOut);
    }

    [Fact]
    public void LinearMulti_AppliesTheAdjunctOnTheOpThatCarriesOne()
    {
        using Harness harness = new Harness();
        using Tensor patchedOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        using Tensor plainOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        LinearOp[] ops =
        [
            new LinearOp(patchedOut, harness.Patched, null),
            new LinearOp(plainOut, harness.Base, null),
        ];
        harness.Backend.LinearMulti(harness.Input, ops);

        using Tensor mergedOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.Linear(mergedOut, harness.Input, harness.Merged, null);
        AssertClose(mergedOut, patchedOut);
        // The unpatched op in the same group must be untouched — a group-wide apply would be as wrong as a miss.
        using Tensor baseOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.Linear(baseOut, harness.Input, harness.Base, null);
        AssertClose(baseOut, plainOut);
    }

    [Fact]
    public void LinearHeadGated_AppliesTheAdjunctToTheGatedActivation()
    {
        // The gate mutates the activation in place before the projection, so the adjunct must consume the GATED
        // input. A backend that folds the gate into its activation quantization has to un-fuse for that reason.
        using Harness harness = new Harness();
        using Tensor logits = new Tensor(new TensorShape(Batch, 2), DType.F32);
        logits.AsSpan<float>().Fill(0.4f);
        using Tensor gatedInput = harness.Input.CastTo(DType.F32);
        using Tensor adjunctOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.LinearHeadGated(adjunctOut, gatedInput, harness.Patched, null, logits, 2, Cols / 2);

        using Tensor referenceInput = harness.Input.CastTo(DType.F32);
        using Tensor mergedOut = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.LinearHeadGated(mergedOut, referenceInput, harness.Merged, null, logits, 2, Cols / 2);

        AssertClose(mergedOut, adjunctOut);
    }

    [Fact]
    public void QuantizedMatMul_RefusesOnABackendWithoutOne()
    {
        using Harness harness = new Harness();
        using Tensor output = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        // The CPU/Vulkan default refuses every quantized weight, adjunct or not, which is a valid covered outcome:
        // the weight cannot reach a GEMM here at all.
        Assert.Throws<NotSupportedException>(
            () => harness.Backend.QuantizedMatMul(output, harness.Input, harness.Patched, null));
    }

    [Fact]
    public void MatMul_RefusesAnAdjunctOperandByName()
    {
        using Harness harness = new Harness();
        using Tensor output = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => harness.Backend.MatMul(output, harness.Input, harness.Patched));
        Assert.Contains("MatMul", error.Message);
        Assert.Contains("LoRA adjunct", error.Message);
    }

    [Fact]
    public void BatchedMatMul_RefusesAnAdjunctOperandByName()
    {
        using Harness harness = new Harness();
        using Tensor batched = new Tensor(new TensorShape(1, Batch, Cols), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, Batch, Rows), DType.F32);
        Assert.Throws<NotSupportedException>(
            () => harness.Backend.BatchedMatMul(output, batched, harness.Patched));
    }

    [Fact]
    public void Conv2D_RefusesAnAdjunctWeightByName()
    {
        using Harness harness = new Harness();
        using Tensor image = new Tensor(new TensorShape(1, Cols, 1, 1), DType.F32);
        using Tensor output = new Tensor(new TensorShape(1, Rows, 1, 1), DType.F32);
        // Attaching to a rank-4 weight is refused outright; a 1x1 VIEW of a patched Linear weight is the only way
        // an adjunct can reach a convolution at all, which is why the op refuses as well.
        using Tensor patchedKernel = harness.Patched.Reshape(new TensorShape(Rows, Cols, 1, 1));
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => harness.Backend.Conv2D(output, image, patchedKernel, null, 1, 1, 0, 0));
        Assert.Contains("Conv2D", error.Message);
    }

    private static void AssertMatchesMergedWeight(Action<IBackend, Tensor, Tensor, Tensor> run,
        int outColumns, int rowOffset)
    {
        using Harness harness = new Harness();
        using Tensor adjunctOut = new Tensor(new TensorShape(Batch, outColumns), DType.F32);
        run(harness.Backend, adjunctOut, harness.Input, harness.Patched);

        using Tensor mergedFull = new Tensor(new TensorShape(Batch, Rows), DType.F32);
        harness.Backend.Linear(mergedFull, harness.Input, harness.Merged, null);

        ReadOnlySpan<float> actual = adjunctOut.AsReadOnlySpan<float>();
        ReadOnlySpan<float> expected = mergedFull.AsReadOnlySpan<float>();
        for (int row = 0; row < Batch; row++)
        {
            for (int column = 0; column < outColumns; column++)
            {
                Assert.Equal(expected[(row * Rows) + rowOffset + column], actual[(row * outColumns) + column], 4);
            }
        }
    }

    private static void AssertClose(Tensor expected, Tensor actual)
    {
        ReadOnlySpan<float> e = expected.AsReadOnlySpan<float>();
        ReadOnlySpan<float> a = actual.AsReadOnlySpan<float>();
        Assert.Equal(e.Length, a.Length);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.Equal(e[i], a[i], 4);
        }
    }

    /// <summary>A base weight, the same weight carrying a one-term adjunct, and the dense merge of the two.</summary>
    private sealed class Harness : IDisposable
    {
        public Harness()
        {
            Backend = new CpuBackend();
            Input = Ramp(Batch, Cols, 0.31f);
            Base = Ramp(Rows, Cols, 0.17f);
            Tensor down = Filled(Rank, Cols, Down);
            Tensor up = Filled(Rows, Rank, Up);
            _factors = [down, up];
            Patched = Base.WithLowRankAdjunct(new LowRankAdjunct
            {
                Terms = [new LowRankAdjunctTerm { Down = down, Up = up, Scale = 1.0f }],
            });
            Merged = Base.CastTo(DType.F32);
            Span<float> merged = Merged.AsSpan<float>();
            float delta = Rank * Down * Up;
            for (int i = 0; i < merged.Length; i++)
            {
                merged[i] += delta;
            }
        }

        private readonly Tensor[] _factors;

        public IBackend Backend { get; }

        public Tensor Input { get; }

        public Tensor Base { get; }

        public Tensor Patched { get; }

        public Tensor Merged { get; }

        private static Tensor Ramp(int rows, int cols, float step)
        {
            Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
            Span<float> values = t.AsSpan<float>();
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = MathF.Sin(i * step);
            }
            return t;
        }

        private static Tensor Filled(long rows, long cols, float value)
        {
            Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
            t.AsSpan<float>().Fill(value);
            return t;
        }

        public void Dispose()
        {
            Merged.Dispose();
            Patched.Dispose();
            Base.Dispose();
            Input.Dispose();
            foreach (Tensor factor in _factors)
            {
                factor.Dispose();
            }
            Backend.Dispose();
        }
    }
}
