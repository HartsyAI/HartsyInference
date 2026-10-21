using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Runs a site on the recording stub and reduces what it did to comparable lines: every backend op in order, each attention
/// call spelled out (route, scale, F16 assertion, operand and mask shapes) and, after the run, which of its operands the site freed.
/// The legacy SDPA record and the layer's Dense <c>Attention</c> record read the same, so pins recorded before a site moved onto the
/// layer hold after it.</summary>
public static class AttentionOpTrace
{
    private static readonly FieldInfo Disposed = typeof(Tensor).GetField("_disposed", BindingFlags.Instance | BindingFlags.NonPublic)!;

    /// <summary>Runs <paramref name="site"/>; an op the stub does not serve ends the run, as the ops past the attention are not the subject.</summary>
    public static string[] Run(Action<AttentionTraceBackend> site, bool serveConvs = false)
    {
        AttentionTraceBackend trace = new() { ServeConvs = serveConvs };
        try { site(trace); }
        catch (NotSupportedException) { /* ops past the attention that the stub does not serve */ }
        catch (TargetInvocationException e) when (e.InnerException is NotSupportedException) { /* the same, through reflection */ }
        return Lines(trace);
    }

    public static string[] Lines(AttentionTraceBackend b)
    {
        Queue<AttentionCall> dense = new(b.Calls.Where(c => c.Route is not (AttentionRoute.Flash or AttentionRoute.FlashDevice)));
        List<string> ops = new();
        List<string> freed = new();
        int sdpa = 0;
        foreach (string op in b.Ops)
        {
            if (op.StartsWith("SDPA ", StringComparison.Ordinal))
            {
                SdpaCall c = b.SdpaCalls[sdpa++];
                ops.Add(Attn("Dense", c.Scale, c.AllowF16, c.Query!, c.Key!, c.Value!, c.Mask));
                freed.Add(Freed(c.Query!, c.Key!, c.Value!, c.Mask));
            }
            else if (op.StartsWith("Attention ", StringComparison.Ordinal))
            {
                AttentionCall c = dense.Dequeue();
                ops.Add(Attn(c.Route.ToString(), c.Scale, c.Numerics.LegacyAssertedBounded, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias));
                freed.Add(Freed(c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias));
            }
            else
            {
                ops.Add(op);
            }
        }
        for (int i = 0; i < freed.Count; i++) ops.Add($"after run, attention #{i}: {freed[i]}");
        return [.. ops];
    }

    private static string Attn(string route, float scale, bool f16, Tensor q, Tensor k, Tensor v, Tensor? mask) =>
        $"Attn {route} scale={scale:0.######} f16={f16} q={Shape(q)} k={Shape(k)} v={Shape(v)} mask={(mask is null ? "none" : Shape(mask))}";

    private static string Shape(Tensor t) => $"{t.DType}{t.Shape}";

    private static string Freed(Tensor q, Tensor k, Tensor v, Tensor? mask) =>
        $"q={State(q)} k={State(k)} v={State(v)} mask={(mask is null ? "none" : State(mask))}";

    private static string State(Tensor t) => (bool)Disposed.GetValue(t)! ? "freed" : "live";

    /// <summary>With <c>WAVE_E2_RECORD</c> set to a file path, appends <paramref name="actual"/> to it as the pin of <paramref name="site"/> and
    /// returns true: how the pins are recorded from the code before a site moved. Otherwise false, and the caller compares.</summary>
    public static bool TryRecord(string site, string[] actual)
    {
        string? record = Environment.GetEnvironmentVariable("WAVE_E2_RECORD");
        if (string.IsNullOrEmpty(record)) return false;
        File.AppendAllLines(record, actual.Select(l => $"{site}\t{l}"));
        return true;
    }
}
