using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity;

/// <summary>How a recorded output is judged against a re-run.</summary>
public enum AcceptanceKind
{
    /// <summary>Every byte equal. For copies, permutations and paths whose arithmetic is unchanged.</summary>
    BitIdentical,

    /// <summary>Within the stated absolute and relative tolerances, with the reason recorded in the manifest.</summary>
    Tolerance,

    /// <summary>Compared against the CPU backend's result on the same inputs rather than a fixture, for paths that
    /// were unsupported on the baseline or whose semantics the change corrects on purpose.</summary>
    HostReference,
}

/// <summary>The acceptance rule of one case: kind, tolerances and the reason a tolerance is justified.</summary>
public sealed record Acceptance(AcceptanceKind Kind, double MaxAbs = 0, double MaxRel = 0, string Reason = "")
{
    /// <summary>Byte equality.</summary>
    public static Acceptance Exact => new(AcceptanceKind.BitIdentical);

    /// <summary>A justified tolerance.</summary>
    public static Acceptance Within(double maxAbs, double maxRel, string reason) => new(AcceptanceKind.Tolerance, maxAbs, maxRel, reason);

    /// <summary>Judged against the CPU backend instead of a fixture.</summary>
    public static Acceptance AgainstHost(double maxAbs, double maxRel, string reason) => new(AcceptanceKind.HostReference, maxAbs, maxRel, reason);
}

/// <summary>Named input and output tensors of one case run; the case owns their lifetime.</summary>
public sealed class OpTensors : IDisposable
{
    /// <summary>Inputs the op reads, by the name they are stored under in the fixture.</summary>
    public Dictionary<string, Tensor> Inputs { get; } = new(StringComparer.Ordinal);

    /// <summary>Outputs the op writes, by fixture name.</summary>
    public Dictionary<string, Tensor> Outputs { get; } = new(StringComparer.Ordinal);

    /// <summary>Names of outputs where the contract allows non-finite values (masked logits, LSE of an excluded row).</summary>
    public HashSet<string> MayBeNonFinite { get; } = new(StringComparer.Ordinal);

    public void Dispose()
    {
        foreach (Tensor t in Inputs.Values) t.Dispose();
        foreach (Tensor t in Outputs.Values) t.Dispose();
    }
}

/// <summary>One recorded-and-checked op invocation: how to build its operands, which backends the baseline serves
/// it on, which must be covered, and how a re-run is judged.</summary>
/// <param name="Family">The op family (the <c>--op</c> selector), e.g. <c>attention</c>.</param>
/// <param name="Name">Unique within the family; the fixture file name.</param>
/// <param name="Build">Allocates seeded inputs and zeroed outputs.</param>
/// <param name="Run">Executes the op on the backend; outputs are read after <see cref="IBackend.Sync"/>.</param>
/// <param name="SupportedOn">True when the backend serves this case on the baseline; false records it as unsupported.</param>
/// <param name="RequiredBackends">Backend kinds (<c>cpu</c>, <c>cuda</c>, <c>vulkan</c>) a <c>check</c> must cover; a skip fails the run.</param>
/// <param name="Acceptance">How the re-run is judged.</param>
/// <param name="HostRun">The case's own host reference, writing the same outputs. Used as the oracle when the path is judged
/// against the host; a case without one is judged by the CPU backend's <see cref="Run"/>, which must serve it.</param>
public sealed record OpCase(
    string Family,
    string Name,
    Func<int, OpTensors> Build,
    Action<IBackend, OpTensors> Run,
    Func<IBackend, bool> SupportedOn,
    IReadOnlyList<string> RequiredBackends,
    Acceptance Acceptance,
    Action<OpTensors>? HostRun = null);
