using System.Reflection;
using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using Xunit;

namespace HartsyInference.Audio.Tests;

/// <summary>The Whisper weight preload degrades instead of failing: a card that cannot hold the weights keeps the
/// backend's per-use upload, so <see cref="WhisperOps.PreloadOrStream"/> must swallow exactly the capacity failure,
/// warn once, and let every other failure through. A regression here is silent until a small card refuses to
/// transcribe at all.</summary>
public sealed class WhisperPreloadFallbackTests
{
    [Fact]
    public void OutOfVram_WarnsOnce_AndTheCallCarriesOn()
    {
        IBackend backend = PreloadFailingBackend.Create(new OutOfVramException(1L << 30, 1L << 20));
        int warned = 0;
        WhisperOps.PreloadOrStream(backend, [], "encoder", ref warned);
        WhisperOps.PreloadOrStream(backend, [], "encoder", ref warned);
        Assert.Equal(1, warned);
    }

    [Fact]
    public void OtherFailures_Propagate()
    {
        IBackend backend = PreloadFailingBackend.Create(new InvalidOperationException("driver failure"));
        int warned = 0;
        Assert.Throws<InvalidOperationException>(() => WhisperOps.PreloadOrStream(backend, [], "decoder", ref warned));
        Assert.Equal(0, warned);
    }

    /// <summary>An <see cref="IBackend"/> whose <see cref="IBackend.PreloadWeights"/> throws the given failure; every
    /// other member is unreachable here and throws.</summary>
    public class PreloadFailingBackend : DispatchProxy
    {
        private Exception? _failure;

        public static IBackend Create(Exception failure)
        {
            IBackend backend = Create<IBackend, PreloadFailingBackend>();
            ((PreloadFailingBackend)(object)backend)._failure = failure;
            return backend;
        }

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) =>
            targetMethod?.Name == nameof(IBackend.PreloadWeights)
                ? throw _failure!
                : throw new NotSupportedException($"{targetMethod?.Name} is not part of this test.");
    }
}
