using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>A rotation the transform stage applies from tables the caller supplies for this call.</summary>
public enum QkRotationKind
{
    /// <summary>No rotation.</summary>
    None,

    /// <summary>Interleaved pairs, <see cref="IBackend.WanRopeInterleaved"/> over each tensor with its own head count.</summary>
    InterleavedPairs,

    /// <summary>Rotate-half with tables broadcast over heads, <see cref="IBackend.ApplyRopeSingle"/>.</summary>
    RotateHalf,

    /// <summary>LTX-2's split rotation, fused with the full-width norm.</summary>
    FullWidthSplit,

    /// <summary>The LTX-0.9 lineage: interleaved pairs over the full inner width after the full-width norm.</summary>
    FullWidthInterleaved,

    /// <summary>Rotate-half on Q and K together, <see cref="IBackend.ApplyRope"/>: set on both, with one table pair.</summary>
    RotateHalfPaired,
}

/// <summary>One tensor's rotation tables for this call; <paramref name="Tokens"/> is the row count the tables cover.</summary>
public readonly record struct QkRotation(QkRotationKind Kind, Tensor? Cos, Tensor? Sin, int Tokens = 0)
{
    /// <summary>No rotation.</summary>
    public static QkRotation None => default;
}

/// <summary>Rotates token-major Q and K on the device before the head permute; a model's RoPE implements it.</summary>
/// <remarks>Device ops only, so it is capture-safe; the implementation leases its own tables around the call.</remarks>
public interface IQkRotation
{
    /// <summary>Rotates <c>[B, S, heads, D]</c> <paramref name="q"/> and <c>[B, S, kvHeads, D]</c> <paramref name="k"/> in place.</summary>
    void RotateTokenMajor(IBackend backend, Tensor q, Tensor k, int heads, int kvHeads);
}

/// <summary>Rotates head-major Q and K after the permute: the batched fallback of a model whose device rotation is single
/// batch item. Host math: it synchronizes and is never on a captured path.</summary>
public interface IHeadMajorQkRotation
{
    /// <summary>Rotates <c>[B, heads, S, D]</c> <paramref name="q"/> and <c>[B, kvHeads, S, D]</c> <paramref name="k"/> in place.</summary>
    void RotateHeadMajor(IBackend backend, Tensor q, Tensor k, int batch, int heads, int kvHeads, int tokens);
}

/// <summary>Replaces the rotated K and V before the head permute (a context-parallel exchange, a cached prefix, appended
/// references).</summary>
public interface IKvRewrite
{
    /// <summary>The K and V to attend over, <c>[B, Skv', kvHeads·D]</c> or <c>[B, Skv', kvHeads, D]</c> of any length; it takes
    /// over <paramref name="k"/> and <paramref name="v"/>.</summary>
    (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v);
}

/// <summary>Receives the rotated K and V a call attends over; the receiver owns them once the call returns.</summary>
public interface IKvCapture
{
    /// <summary>Takes <paramref name="k"/> and <paramref name="v"/>; the layer reads but no longer disposes them.</summary>
    void Capture(Tensor k, Tensor v);
}

/// <summary>Reads named intermediates of a call for diagnostics. It must not keep or modify them, and is never set on a
/// captured path.</summary>
public interface IAttentionProbe
{
    /// <summary>Called with each intermediate as it is produced: <c>q</c>, <c>k</c>, <c>v</c>, <c>qNormed</c>, <c>kNormed</c>
    /// and <c>attention</c> (the head-major output).</summary>
    void Observe(string point, Tensor value);
}

/// <summary>The per-call inputs of an attention layer. Every member is borrowed for the call.</summary>
public readonly record struct AttentionCallContext
{
    /// <summary>Table rotation of Q.</summary>
    public QkRotation QueryRotation { get; init; }

    /// <summary>Table rotation of K.</summary>
    public QkRotation KeyRotation { get; init; }

    /// <summary>A model rotation of Q and K together, used instead of the tables.</summary>
    public IQkRotation? Rotation { get; init; }

    /// <summary>A host rotation after the head permute; forces the head-major entry.</summary>
    public IHeadMajorQkRotation? HeadMajorRotation { get; init; }

    /// <summary>F32 additive bias broadcast over the scores.</summary>
    public Tensor? Bias { get; init; }

    /// <summary>A scale multiplied into V right after its projection, with V's shape and dtype (Krea 2's token weights).</summary>
    public Tensor? ValueRowScale { get; init; }

    /// <summary>Diagnostics; null in production.</summary>
    public IAttentionProbe? Probe { get; init; }

    /// <summary>Replaces the rotated K and V before the head permute.</summary>
    public IKvRewrite? KvRewrite { get; init; }

    /// <summary>Keeps the rotated K and V (after any rewrite) for the caller.</summary>
    public IKvCapture? KvCapture { get; init; }

    /// <summary>The layer disposes the query input, and a distinct key/value input, once the projections have read them; a
    /// failure after that point leaves nothing for the caller to dispose.</summary>
    public bool ReleaseInputs { get; init; }
}
