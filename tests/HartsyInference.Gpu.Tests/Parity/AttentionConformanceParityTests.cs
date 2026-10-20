using HartsyInference.Core.Backends;
using HartsyInference.Tests.Common;
using HartsyInference.Tests.Common.Attention;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Gpu.Tests.Parity;

/// <summary>Every attention case on every GPU backend against one oracle, the contract reference
/// (docs/ATTENTION_CONTRACT.md). Each backend used to be compared with its own inline reference, so the backends never
/// met and a bug a reference shared went unseen. Each row opens a fresh backend, so one that trips failure state cannot
/// poison the next; the tolerance is the worst precision that backend's legacy entry may use for the case.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class AttentionConformanceParityTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    /// <summary>(backend, case) rows for the GPU backends.</summary>
    public static IEnumerable<object[]> Rows => AttnCatalog.Rows(["cuda", "vulkan"]);

    [Theory]
    [MemberData(nameof(Rows))]
    public void Attention_Matches_The_Contract(string kind, string caseId)
    {
        AttnCase c = AttnCatalog.Get(caseId);
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? opened)) return;
        using IBackend backend = opened!;
        AttnInputs inputs = AttnInputs.Create(c);
        AttnOutcome outcome = AttnInvoker.Run(backend, c, inputs, out float[]? actual, out Exception? refusal);
        _out.WriteLine($"{kind} {c.Id} [{c.Origin}]: {outcome}{(refusal is null ? "" : " - " + refusal.Message)}");
        if (outcome == AttnOutcome.Unsupported) return;
        Assert.Equal(AttnInvoker.Expected(kind, c), outcome);
        if (outcome != AttnOutcome.Ran) return;

        AttnAchieved achieved = AttnTolerance.Worst(kind, c);
        int rowWidth = c.Layout == AttnLayout.TokenMajor ? c.Heads * c.HeadDim : c.HeadDim;
        (double rel, double cos, double max) = AttnAssert.Agrees(AttnOracle.CheckedRows(c, actual!), AttnOracle.Expected(c, inputs),
            rowWidth, AttnTolerance.For(achieved), because: $"on {kind} {c.Id} at {achieved}");
        _out.WriteLine($"  {achieved}: relL2 {rel:E2}  min cos {cos:F7}  max|Δ| {max:E2}");
    }

    /// <summary>Negative controls: each runs a case on a backend and asserts the output agrees with the right oracle AND
    /// differs from a wrong one, so the case is proven able to see the bug it names.</summary>
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void Negative_Controls(string kind)
    {
        if (!BackendGate.TryOpen(kind, _out.WriteLine, out IBackend? opened)) return;
        using IBackend backend = opened!;
        Control(backend, kind, "per-head mask reaches every head", AttnCatalog.Get("anchor-t5-perhead"),
            (q, k, v, m) => { long block = m!.Length / 8; for (long i = block; i < m.Length; i++) m[i] = m[i % block]; });
        Control(backend, kind, "per-batch mask reaches every batch item",
            new AttnCase("nc-perbatch", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.PerBatch, AttnSeq.Tiny, 2, 3, 3, 64, 9, 9, 9, 0, 0,
                false, false, false, "control"),
            (q, k, v, m) => { long block = m!.Length / 2; for (long i = block; i < m.Length; i++) m[i] = m[i - block]; });
        Control(backend, kind, "the mask is applied",
            new AttnCase("nc-mask-applied", AttnDType.F32, AttnLayout.HeadMajor, AttnMask.QueryKey, AttnSeq.Tiny, 1, 3, 3, 64, 9, 9, 9, 0, 0,
                false, false, false, "control"),
            (q, k, v, m) => Array.Clear(m!));
        Control(backend, kind, "query head h reads key/value head h/group",
            AttnCatalog.Get("anchor-llm-prefill"), SwapKvHeads(capacity: 96, d: 128));
        Control(backend, kind, "keys past the query rows count",
            AttnCatalog.Get("edge-skv-above-sq"),
            (q, k, v, m) => { for (int i = 5 * 64; i < 77 * 64; i++) v[i] = -v[i]; });
    }

    /// <summary>Runs <paramref name="c"/>, asserts agreement with the oracle, and asserts the oracle of the inputs
    /// <paramref name="wrong"/> rewrites differs from the output.</summary>
    private void Control(IBackend backend, string kind, string what, AttnCase c, Action<float[], float[], float[], float[]?> wrong)
    {
        AttnInputs inputs = AttnInputs.Create(c);
        Assert.Equal(AttnOutcome.Ran, AttnInvoker.Run(backend, c, inputs, out float[]? actual, out Exception? refusal));
        float[] rows = AttnOracle.CheckedRows(c, actual!);
        AttnAssert.Agrees(rows, AttnOracle.Expected(c, inputs), c.HeadDim, AttnTolerance.For(AttnTolerance.Worst(kind, c)), $"{kind}: {what}");
        double apart = AttnAssert.Differs(rows, AttnOracle.Expected(c, AttnInputs.Create(c, wrong)), c.HeadDim, 0.05, $"{kind}: {what}");
        _out.WriteLine($"{kind} control '{what}': {apart:E2} from the wrong oracle");
    }

    /// <summary>Swaps the key/value heads of each pair, so a kernel that maps query heads modulo instead of by group reads the
    /// other head's keys.</summary>
    private static Action<float[], float[], float[], float[]?> SwapKvHeads(int capacity, int d) => (q, k, v, m) =>
    {
        int block = capacity * d;
        foreach (float[] a in (float[][])[k, v])
            for (int i = 0; i < block; i++) (a[i], a[block + i]) = (a[block + i], a[i]);
    };

    [Fact]
    public void Report_Which_Backends_Are_Present()
    {
        foreach (string kind in BackendGate.Kinds)
            _out.WriteLine(BackendGate.UnavailableReason(kind) is { } reason ? $"{kind}: unavailable - {reason}" : $"{kind}: available");
    }
}
