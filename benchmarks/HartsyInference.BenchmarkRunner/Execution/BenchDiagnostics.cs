using System.Diagnostics;
using HartsyInference.Core.Backends;
using HartsyInference.Engine.Diagnostics;
using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Token timestamps from native generation, independent of text chunk boundaries.</summary>
public sealed class BenchDiagnostics(string selector) : IInferenceDiagnostics
{
    private readonly long[] _timestamps = new long[4096];
    public long[] TokenTimestamps => _timestamps.AsSpan(0, Tokens).ToArray();
    public long Prefill { get; private set; }
    public long Start { get; private set; }
    public long First { get; private set; }
    public long Last { get; private set; }
    public int Tokens { get; private set; }
    public int Prompt { get; private set; }
    public DeviceRecord? Device { get; private set; }

    public void Reset()
    {
        Start = First = Last = Prefill = 0;
        Tokens = Prompt = 0;
    }

    public void OnBackendReady(long requestId, IBackend backend) => Device ??= Hardware.Describe(selector, backend);
    public void OnEvent(in InferenceDiagnosticEvent diagnostic)
    {
        switch (diagnostic.Kind)
        {
            case InferenceDiagnosticKind.RequestStarted:
                Start = diagnostic.Timestamp;
                break;
            case InferenceDiagnosticKind.PrefillCompleted:
                Prompt = diagnostic.Count;
                Prefill = diagnostic.Timestamp;
                break;
            case InferenceDiagnosticKind.TokenGenerated:
                if (First == 0)
                    First = diagnostic.Timestamp;
                Last = diagnostic.Timestamp;
                if (diagnostic.Count > _timestamps.Length)
                    throw new InvalidOperationException("Diagnostic token limit exceeded.");
                _timestamps[diagnostic.Count - 1] = diagnostic.Timestamp;
                Tokens = diagnostic.Count;
                break;
        }
    }

    public double? FirstMs => First > Start && Start > 0 ? Stopwatch.GetElapsedTime(Start, First).TotalMilliseconds : null;
    public double? DecodeRate => Tokens > 1 && Last > First ? (Tokens - 1) / Stopwatch.GetElapsedTime(First, Last).TotalSeconds : null;
}
