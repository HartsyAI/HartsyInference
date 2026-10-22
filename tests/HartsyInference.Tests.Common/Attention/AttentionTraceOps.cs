using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Runs a site on the recording stub and spells its ops: every op in order, each dense attention call spelled out (route, scale, F16
/// assertion, operands, mask, live tensors) and, after the run, which of its operands the site freed. The legacy SDPA record and the layer's
/// Dense <c>Attention</c> record read the same, so pins recorded before a site moved onto the layer hold after it.</summary>
public static class AttentionTraceOps
{
    private static readonly FieldInfo Disposed = typeof(Tensor).GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>Runs <paramref name="site"/> on a fresh stub; an op the stub does not serve ends the run, as the ops past the attention are not the subject.</summary>
    public static string[] Record(Action<AttentionTraceBackend> site, bool serveConvs = false)
    {
        AttentionTraceBackend trace = new() { ServeConvs = serveConvs };
        try { site(trace); }
        catch (NotSupportedException) { /* ops past the attention that the stub does not serve */ }
        catch (TargetInvocationException e) when (e.InnerException is NotSupportedException) { /* the same, through reflection */ }
        return Spell(trace);
    }

    public static string[] Spell(AttentionTraceBackend b)
    {
        Queue<AttentionCall> layer = new(b.Calls.Where(c => c.Route is not (AttentionRoute.Flash or AttentionRoute.FlashDevice)));
        List<SdpaCall> dense = [];
        List<string> lines = [];
        int sdpa = 0;
        foreach (string op in b.Ops)
        {
            string route = "Dense";
            SdpaCall c;
            if (op.StartsWith("SDPA ", StringComparison.Ordinal)) c = b.SdpaCalls[sdpa++];
            else if (op.StartsWith("Attention ", StringComparison.Ordinal))
            {
                AttentionCall a = layer.Dequeue();
                route = a.Route.ToString();
                c = new(a.Scale, a.Numerics.LegacyAssertedBounded, a.Output.Storage, a.Query.Storage, a.Key.Storage, a.Value.Storage, a.Mask.Bias);
            }
            else
            {
                lines.Add(op);
                continue;
            }
            lines.Add($"Attn {route} scale={c.Scale:0.######} f16={c.AllowF16} out={Shape(c.Output)} q={Shape(c.Query)} k={Shape(c.Key)} v={Shape(c.Value)} "
                + $"mask={(c.Mask is null ? "none" : Shape(c.Mask))} live={b.LiveAtAttention[dense.Count]}");
            dense.Add(c);
        }
        for (int i = 0; i < dense.Count; i++)
        {
            SdpaCall c = dense[i];
            lines.Add($"attention #{i} after run: out={State(c.Output)} q={State(c.Query)} k={State(c.Key)} v={State(c.Value)} mask={(c.Mask is null ? "none" : State(c.Mask))}");
        }
        return [.. lines];
    }

    /// <summary>With <c>ATTN_TRACE_RECORD</c> set to a file path, appends <paramref name="actual"/> to it as the pin of <paramref name="site"/> and
    /// returns true: how the pins are recorded from the code before a site moved. Otherwise false, and the caller compares.</summary>
    public static bool TryRecord(string site, string[] actual)
    {
        string? record = Environment.GetEnvironmentVariable("ATTN_TRACE_RECORD");
        if (string.IsNullOrEmpty(record)) return false;
        File.AppendAllLines(record, actual.Select(l => $"{site}\t{l}"));
        return true;
    }

    public static bool IsDisposed(Tensor t) => (bool)Disposed.GetValue(t)!;

    private static string Shape(Tensor t) => $"{t.Shape}:{t.DType}";

    private static string State(Tensor t) => IsDisposed(t) ? "freed" : "live";
}
