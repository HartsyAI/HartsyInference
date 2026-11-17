using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>Default-off observation of actual execution; enabling reserves host metadata only and never steers dispatch.</summary>
public static class AttentionEvidence
{
    private static AttentionEvidenceSession? _session;
    [ThreadStatic] internal static string? Component;
    [ThreadStatic] internal static int Layer;

    /// <summary>Whether an explicitly started diagnostic session is collecting observations.</summary>
    public static bool Enabled => Volatile.Read(ref _session) is not null;

    /// <summary>Starts one process-wide capture, before scheduling work; concurrent sessions are rejected.</summary>
    public static AttentionEvidenceSession Begin(int capacity = 65536)
    {
        AttentionEvidenceSession session = new(capacity);
        if (Interlocked.CompareExchange(ref _session, session, null) is not null)
            throw new InvalidOperationException("An attention evidence session is already active.");
        return session;
    }

    /// <summary>Attributes synchronous work; scheduling code must enter this on the execution thread.</summary>
    public static AttentionEvidenceContext Context(string component, int layer = -1) =>
        Enabled ? new AttentionEvidenceContext(component, layer) : default;

    /// <summary>Records a completed row without retaining tensors or asking admission to run again.</summary>
    public static void Executed(string backend, string implementation, string? branch, in AttentionCall call,
        AttentionLowering lowering = AttentionLowering.None)
    {
        AttentionEvidenceSession? session = Volatile.Read(ref _session);
        if (session is null) return;
        AttentionEvidenceEvent observation = new()
        {
            Kind = AttentionEvidenceKind.Execution, Site = backend, Component = Component, Layer = Component is null ? -1 : Layer,
            Implementation = implementation, Branch = branch, Route = call.Route, Batch = call.Batch,
            Heads = call.Heads, KvHeads = call.KvHeads, QueryTokens = call.QueryTokens, KeyCapacity = call.KeyTokens,
            ValidKeys = call.ValidKvLength, QkHeadDim = call.QkHeadDim, ValueHeadDim = call.ValueHeadDim,
            QueryType = call.Query.DType.Name, KvType = call.Key.DType.Name, QueryLayout = call.Query.Layout,
            KvLayout = call.Key.Layout, QueryOffset = call.Query.Offset, QueryHeadStride = call.Query.HeadStride,
            QueryTokenStride = call.Query.TokenStride, KeyOffset = call.Key.Offset, KeyHeadStride = call.Key.HeadStride,
            KeyTokenStride = call.Key.TokenStride, Numerics = call.Numerics, DeclaredLowering = lowering, Count = 1,
        };
        session.Record(in observation);
    }

    /// <summary>Records a completed named action; bytes are logical payload bytes, not measured bandwidth.</summary>
    public static void Observed(AttentionEvidenceKind kind, string site, long bytes = 0, long count = 1, int layer = -1)
    {
        AttentionEvidenceSession? session = Volatile.Read(ref _session);
        if (session is null) return;
        AttentionEvidenceEvent observation = new()
        {
            Kind = kind, Site = site, Component = Component, Layer = layer >= 0 ? layer : Component is null ? -1 : Layer,
            Bytes = bytes, Count = count,
        };
        session.Record(in observation);
    }

    /// <summary>Records completed preparation without touching storage when observation is disabled.</summary>
    public static void Payload(AttentionEvidenceKind kind, string site, Tensor output)
    {
        if (!Enabled) return;
        Observed(kind, site, output.DType.ComputeByteCount(output.ElementCount));
    }

    internal static void End(AttentionEvidenceSession session) => Interlocked.CompareExchange(ref _session, null, session);
}
