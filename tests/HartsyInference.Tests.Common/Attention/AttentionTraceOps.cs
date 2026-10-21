using System.Reflection;

using HartsyInference.Core.Backends;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Runs a site on the recording stub and spells its ops so the legacy SDPA record and the layer's <c>Attention</c> record read the same.</summary>
public static class AttentionTraceOps
{
    /// <summary>The ops with each dense attention call spelled out: scale, F16 assertion, operand shapes, output dtype and mask.</summary>
    public static string[] Spell(AttentionTraceBackend b)
    {
        Queue<AttentionCall> dense = new(b.Calls.Where(c => c.Route is not (AttentionRoute.Flash or AttentionRoute.FlashDevice)));
        int sdpa = 0;
        List<string> ops = new();
        foreach (string op in b.Ops)
        {
            if (op.StartsWith("SDPA ", StringComparison.Ordinal))
            {
                SdpaCall c = b.SdpaCalls[sdpa++];
                ops.Add($"Attn Dense scale={c.Scale:0.######} f16={c.AllowF16} {c.Operands}");
            }
            else if (op.StartsWith("Attention ", StringComparison.Ordinal))
            {
                AttentionCall c = dense.Dequeue();
                ops.Add($"Attn {c.Route} scale={c.Scale:0.######} f16={c.Numerics.LegacyAssertedBounded} "
                    + AttentionTraceBackend.Operands(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Mask.Bias));
            }
            else
            {
                ops.Add(op);
            }
        }
        return [.. ops];
    }

    /// <summary>Runs <paramref name="site"/> against a fresh stub and returns its spelled ops and the live tensor count at each attention call; a
    /// <see cref="NotSupportedException"/> from an op the stub does not serve ends the trace there.</summary>
    public static AttentionTrace Record(Action<AttentionTraceBackend> site)
    {
        AttentionTraceBackend trace = new();
        try { site(trace); }
        catch (NotSupportedException) { /* ops past the attention that the stub does not serve */ }
        catch (TargetInvocationException e) when (e.InnerException is NotSupportedException) { /* the same, through reflection */ }
        return new AttentionTrace(Spell(trace), [.. trace.LiveAtAttention]);
    }
}

/// <summary>The recorded ops of a site and the live tensor count at each of its attention calls.</summary>
public sealed record AttentionTrace(string[] Ops, int[] LiveAtAttention);
