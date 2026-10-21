using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Tests.Common.Attention;

/// <summary>Reads a site's run on <see cref="AttentionTraceBackend"/> as one list that does not change when a site moves from a direct
/// <c>ScaledDotProductAttention</c> call onto the shared layer: the ops in order, each attention call's scale, F16 assertion and mask, then the
/// shapes of its tensors and which of them are freed once the site has returned.</summary>
public static class AttentionTraceReport
{
    /// <summary>Whether <paramref name="t"/> has been disposed.</summary>
    public static unsafe bool IsDisposed(Tensor t)
    {
        try
        {
            _ = (nint)t.DataPointer;
            return false;
        }
        catch (ObjectDisposedException)
        {
            return true;
        }
    }

    /// <summary>The report of what <paramref name="b"/> recorded.</summary>
    public static string[] Report(AttentionTraceBackend b)
    {
        Queue<AttentionCall> dense = new(b.Calls.Where(c => c.Route is not (AttentionRoute.Flash or AttentionRoute.FlashDevice)));
        List<SdpaOperands> calls = [];
        int sdpa = 0;
        List<string> ops = [];
        foreach (string op in b.Ops)
        {
            if (op.StartsWith("SDPA ", StringComparison.Ordinal))
            {
                SdpaCall c = b.SdpaCalls[sdpa];
                calls.Add(b.SdpaTensors[sdpa++]);
                ops.Add($"Attn Dense scale={c.Scale:0.######} f16={c.AllowF16} mask={c.Masked}");
            }
            else if (op.StartsWith("Attention ", StringComparison.Ordinal))
            {
                AttentionCall c = dense.Dequeue();
                calls.Add(new SdpaOperands(c.Output.Storage, c.Query.Storage, c.Key.Storage, c.Value.Storage, c.Mask.Bias));
                ops.Add($"Attn {c.Route} scale={c.Scale:0.######} f16={c.Numerics.LegacyAssertedBounded} mask={c.Mask.Bias is not null}");
            }
            else
            {
                ops.Add(op);
            }
        }
        foreach (SdpaOperands c in calls)
        {
            ops.Add($"out {Shape(c.Output)} q {Shape(c.Query)} k {Shape(c.Key)} v {Shape(c.Value)} mask {(c.Mask is null ? "none" : Shape(c.Mask))}");
            ops.Add($"freed out={IsDisposed(c.Output)} q={IsDisposed(c.Query)} k={IsDisposed(c.Key)} v={IsDisposed(c.Value)} mask={c.Mask is not null && IsDisposed(c.Mask)}");
        }
        return [.. ops];
    }

    /// <summary>Runs <paramref name="site"/> on a fresh recording stub and reports it; ops past the attention that the stub does not serve end the run.</summary>
    public static string[] Run(Action<AttentionTraceBackend> site)
    {
        AttentionTraceBackend trace = new();
        try { site(trace); }
        catch (NotSupportedException) { /* ops past the attention that the stub does not serve */ }
        catch (TargetInvocationException e) when (e.InnerException is NotSupportedException) { /* the same, through reflection */ }
        return Report(trace);
    }

    private static string Shape(Tensor t) => $"{t.Shape}{t.DType}";
}
