using HartsyInference.Core.Backends;
using HartsyInference.Engine;
using HartsyInference.Engine.Diagnostics;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;
/// <summary>Observer failures must not replace engine errors or poison later requests.</summary>
public sealed class DiagnosticsTests
{
    [Fact]
    public async Task ThrowingObserverIsDisabledAcrossRequests()
    {
        ThrowingObserver observer = new();
        using InferenceEngine engine = new("cpu", new EngineOptions { Diagnostics = observer });
        ModelSpec spec = new()
        {
            Requested = "missing",
            LocalPath = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".gguf"),
            Modality = Modality.Text
        };
        TextRequest request = new()
        {
            Messages = [new TextMessage
            {
                Role = TextRole.User,
                Content = "test"
            }

            ],
            Device = "cpu"
        };
        Exception first = await Assert.ThrowsAnyAsync<Exception>(() => engine.Text.GenerateAsync(spec, request));
        Exception second = await Assert.ThrowsAnyAsync<Exception>(() => engine.Text.GenerateAsync(spec, request));
        Assert.IsNotType<ObserverException>(first);
        Assert.IsNotType<ObserverException>(second);
        Assert.Equal(1, observer.Calls);
    }

    private sealed class ObserverException : Exception;
    private sealed class ThrowingObserver : IInferenceDiagnostics
    {
        public int Calls { get; private set; }

        public void OnEvent(in InferenceDiagnosticEvent diagnostic)
        {
            Calls++;
            throw new ObserverException();
        }

        public void OnBackendReady(long requestId, IBackend backend) => throw new ObserverException();
    }
}
