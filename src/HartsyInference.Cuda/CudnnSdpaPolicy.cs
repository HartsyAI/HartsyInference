using System.Collections.Concurrent;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>Mask kinds the cuDNN SDPA graph carries.</summary>
internal enum CudnnSdpaMask { None, Bias }

/// <summary>Bias axes that are stored rather than broadcast; the key axis always is.</summary>
[Flags]
internal enum CudnnBiasAxes { None = 0, Batch = 1, Head = 2, Query = 4 }

/// <summary>Dims of an additive fp32 bias <c>[B, H, Sq, Skv]</c>; <c>Sq == 0</c> is no bias.</summary>
internal readonly record struct CudnnBias(long B, long H, long Sq)
{
    public static CudnnBias None => default;

    public bool HasBias => Sq > 0;

    public CudnnBiasAxes Axes => (B > 1 ? CudnnBiasAxes.Batch : 0) | (H > 1 ? CudnnBiasAxes.Head : 0) | (Sq > 1 ? CudnnBiasAxes.Query : 0);

    /// <summary>Whether <paramref name="mask"/> is an fp32 <c>[1|Sq, Skv]</c>, <c>[1|H, 1|Sq, Skv]</c> or <c>[1|B, 1|H, 1|Sq, Skv]</c> bias; null is no bias.</summary>
    public static bool TryFrom(Tensor? mask, long b, long h, long sq, long skv, out CudnnBias bias)
    {
        bias = None;
        if (mask is null) return true;
        if (mask.DType != DType.F32) return false;
        TensorShape s = mask.Shape;
        int r = s.Rank;
        if (r < 2 || r > 4 || s[r - 1] != skv) return false;
        long rows = s[r - 2], heads = r >= 3 ? s[r - 3] : 1, batch = r == 4 ? s[0] : 1;
        if ((rows != 1 && rows != sq) || (heads != 1 && heads != h) || (batch != 1 && batch != b)) return false;
        // The graph declares 16-byte alignment for every tensor; a per-head bias's head and row starts need Skv % 4 == 0.
        if (heads > 1 && skv % 4 != 0) return false;
        bias = new CudnnBias(batch, heads, rows);
        return true;
    }
}

/// <summary>The graph variant a call needs, apart from its sizes; failures are remembered per signature.</summary>
internal readonly record struct CudnnSdpaSignature(CudnnSdpa.SdpaDtype Dtype, bool Gqa, CudnnSdpaMask Mask, CudnnBiasAxes BiasAxes, int HeadDim,
    bool Strided = false)
{
    /// <summary>F32 operands run the graph in fp16 behind a cast, so they share the F16 signature. <paramref name="strided"/> keeps a refused
    /// view from disabling the whole-tensor plans of its dtype and head dim.</summary>
    public static CudnnSdpaSignature For(DType io, long h, long hKv, long d, CudnnBias bias, bool strided = false) => new(
        io == DType.BF16 ? CudnnSdpa.SdpaDtype.BF16 : CudnnSdpa.SdpaDtype.F16, hKv != h,
        bias.HasBias ? CudnnSdpaMask.Bias : CudnnSdpaMask.None, bias.Axes, (int)d, strided);

    /// <summary>A mask the graph cannot carry reads as none; the row's mask check rejects that call first.</summary>
    public static CudnnSdpaSignature Of(in AttentionCall c)
    {
        CudnnBias.TryFrom(c.Mask.Bias, c.Batch, c.Heads, c.QueryTokens, c.KeyTokens, out CudnnBias bias);
        return For(c.Query.DType, c.Heads, c.KvHeads, c.QkHeadDim, bias, CudnnSdpa.IsStrided(c.QkHeadDim, CudnnSdpa.View.Of(c.Query),
            CudnnSdpa.View.Of(c.Key), CudnnSdpa.View.Of(c.Value), CudnnSdpa.View.Of(c.Output), CudnnSdpa.Offsets.Of(in c)));
    }

    public override string ToString() =>
        $"{Dtype} d{HeadDim}{(Gqa ? " gqa" : "")}{(Mask == CudnnSdpaMask.Bias ? $" bias({BiasAxes})" : "")}{(Strided ? " strided" : "")}";
}

/// <summary>A structural refusal holds for one shape: its signature plus the plan's sizes and strides.</summary>
internal readonly record struct CudnnSdpaShape(CudnnSdpaSignature Signature, DType Input, CudnnSdpa.PlanKey Plan)
{
    public override string ToString() =>
        $"b{Plan.B} h{Plan.H}/{Plan.Hkv} sq{Plan.Sq} skv{Plan.Sk} {Signature} q{Strides(Plan.Q)} k{Strides(Plan.K)} {Input.ToString().ToLowerInvariant()}";

    private static string Strides(in CudnnSdpa.View v) => $"[{v.BatchStride},{v.HeadStride},{v.TokenStride}]";
}

/// <summary>Per-signature failure, backoff and refusal state. Reads are lock-free; writes come from <c>TryCudnnSdpaLocked</c>,
/// under the backend's cuDNN lock.</summary>
internal sealed class CudnnSdpaHealth
{
    /// <param name="NextRetryAtTicks">Environment.TickCount64; 0 = eligible immediately.</param>
    private sealed record FailureState(int ConsecutiveFailures, long NextRetryAtTicks, bool Permanent);

    /// <summary>Shapes refused under one signature, with no success, before the signature is given up.</summary>
    internal const int RefusalsToDisable = 3;

    /// <summary>Refusals warned about per process.</summary>
    internal const int RefusalWarnings = 64;

    /// <summary>What <see cref="RecordFailure"/> decided, for the caller to log.</summary>
    internal readonly record struct FailureOutcome(bool Permanent, bool NewRefusal, bool Warn, bool SignatureDisabled, int RefusedUnderSignature,
        int ConsecutiveFailures, long BackoffMs);

    private readonly ConcurrentDictionary<CudnnSdpaSignature, FailureState> _state = new();
    private readonly ConcurrentDictionary<CudnnSdpaShape, byte> _refused = new();
    private readonly ConcurrentDictionary<CudnnSdpaSignature, byte> _working = new();

    public IReadOnlyCollection<string> RefusedShapes => [.. _refused.Keys.Select(k => k.ToString())];

    public int RefusedShapeCount => _refused.Count;

    /// <summary>Consecutive failures, permanence and retry time per signature.</summary>
    public IReadOnlyDictionary<CudnnSdpaSignature, (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt)> Diagnostics =>
        _state.ToDictionary(kv => kv.Key, kv => Describe(kv.Value));

    /// <summary>The same snapshot folded per head dim: the highest count, permanent if any signature is.</summary>
    public IReadOnlyDictionary<long, (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt)> DimDiagnostics =>
        _state.GroupBy(kv => (long)kv.Key.HeadDim).ToDictionary(g => g.Key, g =>
        {
            bool permanent = g.Any(kv => kv.Value.Permanent);
            DateTimeOffset? next = permanent ? null : g.Max(kv => Describe(kv.Value).NextRetryAt);
            return (g.Max(kv => kv.Value.ConsecutiveFailures), permanent, next);
        });

    private static (int ConsecutiveFailures, bool Permanent, DateTimeOffset? NextRetryAt) Describe(FailureState s) =>
        (s.ConsecutiveFailures, s.Permanent,
            s.Permanent ? null : DateTimeOffset.UtcNow.AddMilliseconds(s.NextRetryAtTicks - Environment.TickCount64));

    /// <summary>No failure on record, or not given up and past its backoff window.</summary>
    public bool Eligible(in CudnnSdpaSignature signature, long nowTicks)
    {
        if (!_state.TryGetValue(signature, out FailureState? s)) return true;
        return !s.Permanent && nowTicks >= s.NextRetryAtTicks;
    }

    public bool Refused(in CudnnSdpaShape shape) => _refused.ContainsKey(shape);

    /// <summary>Marks the signature as having run; returns the failures it recovered from.</summary>
    public int RecordSuccess(in CudnnSdpaSignature signature)
    {
        _working.TryAdd(signature, 0);
        return _state.TryRemove(signature, out FailureState? recovered) ? recovered.ConsecutiveFailures : 0;
    }

    public FailureOutcome RecordFailure(in CudnnSdpaShape shape, bool permanent, long nowTicks)
    {
        CudnnSdpaSignature sig = shape.Signature;
        if (permanent)
        {
            bool added = _refused.TryAdd(shape, 0);
            int refused = _refused.Keys.Count(k => k.Signature == sig);
            bool disable = !_working.ContainsKey(sig) && refused >= RefusalsToDisable;
            if (disable) _state[sig] = new FailureState(0, 0, Permanent: true);
            return new(true, added, added && _refused.Count <= RefusalWarnings, disable, refused, 0, 0);
        }
        int failures = _state.TryGetValue(sig, out FailureState? previous)
            ? previous.ConsecutiveFailures == int.MaxValue ? int.MaxValue : previous.ConsecutiveFailures + 1
            : 1;
        long backoffMs = BackoffMs(failures);
        _state[sig] = new FailureState(failures, nowTicks + backoffMs, Permanent: false);
        return new(false, false, false, false, 0, failures, backoffMs);
    }

    /// <summary>Doubles from 2 s to a 5 minute cap; a transient failure is never permanent.</summary>
    internal static long BackoffMs(int consecutiveFailures)
    {
        const long baseMs = 2000, capMs = 5 * 60 * 1000;
        int shift = Math.Min(consecutiveFailures - 1, 8);
        return Math.Min(baseMs * (1L << shift), capMs);
    }
}

/// <summary>What the cuDNN rows declare and the call-only rules they share, testable without a device.</summary>
internal static class CudnnSdpaRows
{
    /// <summary>Head-shared, per-head, per-batch and key-only biases.</summary>
    internal const AttentionMaskSupport Masks = AttentionMaskSupport.HeadSharedBias | AttentionMaskSupport.PerHeadBias;

    /// <summary>Every layout: the rows read each operand through its strides, and Q, K, V and the output may differ.</summary>
    internal const AttentionLayouts AnyView = AttentionLayouts.HeadMajor | AttentionLayouts.TokenMajor | AttentionLayouts.PackedRows;

    /// <summary>Head dims 32–256, any bias, F16 (and BF16 when <paramref name="types"/> has it). <paramref name="gqa"/> only where the
    /// row's entry can carry grouped K/V: the dense and token-major entries take equal head counts. <paramref name="strided"/> rows
    /// read offset and strided views; the F32 row reads whole tensors only.</summary>
    internal static AttentionKernelCapability Capability(string id, AttentionRoute entry, AttentionDTypes types, AttentionLayouts layouts,
        bool gqa = false, bool strided = true) => new()
    {
        Id = id, Entry = entry, QueryTypes = types, KvTypes = types, Layouts = layouts, StridedViews = strided,
        MinHeadDim = CudnnSdpa.MinHeadDim, MaxHeadDim = CudnnSdpa.MaxHeadDim, Gqa = gqa,
        Masks = Masks,
        UsesQueryKey = AttentionFormats.F16 | ((types & AttentionDTypes.BF16) != 0 ? AttentionFormats.Bf16 : 0),
        UsesProbValue = AttentionFormats.F16 | ((types & AttentionDTypes.BF16) != 0 ? AttentionFormats.Bf16 : 0),
    };

    /// <summary>All fp16, or all bf16 with the call's grant (legacy entry points grant none): the graph runs them uncast.</summary>
    internal static bool HalfOperands(in AttentionCall c)
    {
        DType t = c.Query.DType;
        if (c.Key.DType != t || c.Value.DType != t || c.Output.DType != t) return false;
        return t == DType.F16 || (t == DType.BF16 && c.Numerics.Permits(AttentionFormats.Bf16, AttentionFormats.Bf16));
    }

    /// <summary>Operands, views and mask all run in place.</summary>
    internal static bool Covers(in AttentionCall c) => HalfOperands(in c) && ViewsFit(in c) && MaskFits(in c);

    /// <summary>The mask is none or an fp32 bias the graph broadcasts.</summary>
    internal static bool MaskFits(in AttentionCall c) =>
        CudnnBias.TryFrom(c.Mask.Bias, c.Batch, c.Heads, c.QueryTokens, c.KeyTokens, out _);

    /// <summary>A view cuDNN reads in place: contiguous head dim, strides and offset on 16-byte boundaries, inside its tensor.</summary>
    internal static bool Takes(in AttentionOperand x) =>
        CudnnSdpa.View.Of(in x).Takes(x.HeadDim) && x.Offset >= 0 && x.Offset % CudnnSdpa.AlignElements == 0 && x.Extent <= x.Storage.ElementCount;

    /// <summary>The four operands agree on batch, heads, rows, head dim and dtype and each is a view cuDNN takes with no copy. The output
    /// must cover its whole tensor (it is bound to a fresh buffer); F32 operands are cast whole, so they must be whole tensors.</summary>
    internal static bool ViewsFit(in AttentionCall c)
    {
        AttentionOperand q = c.Query, k = c.Key, v = c.Value, o = c.Output;
        if (q.Batch < 1 || k.Heads < 1 || q.Heads % k.Heads != 0 || k.Batch != q.Batch || v.Batch != q.Batch || o.Batch != q.Batch
            || v.Heads != k.Heads || o.Heads != q.Heads || v.Tokens != k.Tokens || o.Tokens != q.Tokens
            || k.HeadDim != q.HeadDim || v.HeadDim != q.HeadDim || o.HeadDim != q.HeadDim
            || k.DType != q.DType || v.DType != q.DType || o.DType != q.DType)
            return false;
        if (!Takes(in q) || !Takes(in k) || !Takes(in v) || !Takes(in o) || !o.CoversStorage) return false;
        return q.DType != DType.F32
            || (q.IsPlain && k.IsPlain && v.IsPlain && o.IsPlain && q.Storage.ElementCount <= int.MaxValue && k.Storage.ElementCount <= int.MaxValue
                && v.Storage.ElementCount <= int.MaxValue && o.Storage.ElementCount <= int.MaxValue);
    }
}
