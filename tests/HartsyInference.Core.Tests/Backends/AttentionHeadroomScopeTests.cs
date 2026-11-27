using System.Reflection;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Memory.Vram.Streaming;
using Xunit;

namespace HartsyInference.Core.Tests.Backends;

/// <summary>A streaming scope notes its headroom on the backend while it is open and clears the note when it is disposed, so the attention budget
/// (<c>numerics.tileFromDevice</c>) follows the open scope only.</summary>
public sealed class AttentionHeadroomScopeTests
{
    [Fact]
    public void OpeningAScopeNotesItsHeadroomAndDisposingClearsIt()
    {
        IBackend iBackend = NoteRecorder.Make();
        NoteRecorder backend = (NoteRecorder)(object)iBackend;
        BlockStreamingOptions options = new()
        {
            Backend = iBackend,
            Denoiser = DispatchProxy.Create<IStreamableDenoiser, NullDenoiser>(),
            ModelName = "headroom-test",
            HeadroomBytes = 3L << 30,
        };

        BlockStreamingScope scope = new(options, residentPrefixBlocks: 0, streamer: null);
        Assert.Equal(new long[] { 3L << 30 }, backend.Notes);

        scope.Dispose();
        Assert.Equal(new long[] { 3L << 30, 0 }, backend.Notes);
    }

    [Fact]
    public void TheNoteDoesNotDependOnThePrefixCount()
    {
        // The headroom is what the scope was given; the resident prefix count does not change what is noted.
        IBackend ia = NoteRecorder.Make(), ib = NoteRecorder.Make();
        NoteRecorder a = (NoteRecorder)(object)ia, b = (NoteRecorder)(object)ib;
        BlockStreamingOptions Options(IBackend backend) => new()
        {
            Backend = backend,
            Denoiser = DispatchProxy.Create<IStreamableDenoiser, NullDenoiser>(),
            ModelName = "headroom-test",
            HeadroomBytes = 3L << 30,
        };
        using BlockStreamingScope scopeA = new(Options(ia), residentPrefixBlocks: 0, streamer: null);
        using BlockStreamingScope scopeB = new(Options(ib), residentPrefixBlocks: 24, streamer: null);
        Assert.Equal(a.Notes, b.Notes);
    }

    /// <summary>An <see cref="IBackend"/> that records <c>NoteAttentionHeadroom</c> and does nothing else.</summary>
    public class NoteRecorder : DispatchProxy
    {
        public List<long> Notes { get; } = new();

        public static IBackend Make() => Create<IBackend, NoteRecorder>();

        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args)
        {
            if (targetMethod?.Name == nameof(IBackend.NoteAttentionHeadroom)) Notes.Add((long)args![0]!);
            return null;
        }
    }

    /// <summary>An <see cref="IStreamableDenoiser"/> whose members do nothing; the scope only unhooks its block callback on dispose.</summary>
    public class NullDenoiser : DispatchProxy
    {
        protected override object? Invoke(MethodInfo? targetMethod, object?[]? args) => null;
    }
}
