using System.Collections.Concurrent;
using System.Numerics;
using HartsyInference.Core.Logging;

namespace HartsyInference.Core.Backends.Attention;

/// <summary>The row each attention shape class ran on, logged once per class at debug level.</summary>
public static class AttentionPlanLog
{
    private static readonly ConcurrentDictionary<string, byte> Seen = new(StringComparer.Ordinal);

    /// <summary>Logs <paramref name="row"/> for <paramref name="call"/>'s shape class the first time that class runs on
    /// <paramref name="backend"/>. Sequence lengths are bucketed to powers of two so decode does not log every token.</summary>
    public static void Ran(string backend, AttentionKernelRow row, in AttentionCall call)
    {
        if (Logs.MinLevel > LogLevel.Debug) return;
        static uint Bucket(int n) => BitOperations.RoundUpToPowerOf2((uint)Math.Max(1, n));
        string mask = (call.Mask.Causal ? " causal" : "") + (call.Mask.WindowLeft is { } w ? $" window{w}" : "")
            + (call.Mask.Bias is { } b ? $" bias{b.Shape}" : "");
        string key = $"{backend} {call.Route} {call.Query.DType}/{call.Key.DType} h{call.Heads}/{call.KvHeads} d{call.QkHeadDim} "
            + $"sq<={Bucket(call.QueryTokens)} skv<={Bucket(call.KeyTokens)}{mask}";
        if (Seen.TryAdd(key, 0))
            Logs.Debug($"Attention {key}: {new AttentionPlan(row, row.Variant(in call), "ran", null, row.Preparation(in call)).Describe()}");
    }

    /// <summary>Warns once per backend and gap that a declared host-reference row served a call no device kernel covers.</summary>
    public static void HostReference(string backend, string gap)
    {
        if (Seen.TryAdd("host-reference " + backend + " " + gap, 0))
            Logs.Warning($"{backend} attention: {gap} has no device kernel and runs on the declared host-reference row.");
    }
}
