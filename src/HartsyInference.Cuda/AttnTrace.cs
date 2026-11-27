using System;
using System.Threading;

namespace HartsyInference.Cuda;

/// <summary>Per-decision attention trace for the same-seed determinism diagnosis. Off unless <c>HARTSY_ATTN_TRACE=1</c>. Writes one
/// line per budget-gate decision and per flash-attention tile height to stderr, numbered in call order, so two runs can be diffed to
/// the first decision that differs. Logging only: it never changes a path or a value.</summary>
internal static class AttnTrace
{
    private static readonly bool On = Environment.GetEnvironmentVariable("HARTSY_ATTN_TRACE") == "1";
    private static long _seq;

    internal static void Gate(string gate, bool over, ulong budget, ulong free, ulong total)
    {
        if (!On) return;
        Write($"gate {gate} over={over} budgetMiB={budget >> 20} freeMiB={free >> 20} totalMiB={total >> 20}");
    }

    internal static void Tile(long br, long sq, long skv, long heads, ulong free, ulong total, long capBytes, bool fromDevice)
    {
        if (!On) return;
        Write($"tile Br={br} sq={sq} skv={skv} heads={heads} freeMiB={free >> 20} totalMiB={total >> 20} capMiB={capBytes >> 20} fromDevice={fromDevice}");
    }

    private static void Write(string body)
    {
        long n = Interlocked.Increment(ref _seq);
        Console.Error.WriteLine($"attn-trace {n:D7} {body}");
    }
}
