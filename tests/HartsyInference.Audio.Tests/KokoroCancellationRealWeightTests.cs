using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Audio.Tests;

/// <summary>Kokoro's stage boundaries on real weights and the CPU backend. A call with a live token that is never
/// cancelled is byte-identical to one without a token; cancelling at each boundary in turn stops the call there with no
/// later stage run; and the call after all of those is still byte-identical, so a stopped synthesis leaves nothing
/// behind that changes the next one. Skips through <see cref="RealWeightGate"/> when the files are not on the box.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class KokoroCancellationRealWeightTests
{
    private const string Sentence = "Please hold while I check.";

    /// <summary>Every boundary of an iSTFTNet synthesis, in the order a call passes them.</summary>
    private static readonly string[] Boundaries =
        ["start", "plbert", "textenc", "durations", "regulate", "f0n", "encode", "decode", "source", "stage0", "stage1", "post"];

    private readonly ITestOutputHelper _output;

    public KokoroCancellationRealWeightTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task CancellingAtEachBoundary_StopsThere_AndTheNextCallIsByteIdentical()
    {
        if (!RealWeightGate.Require(_output.WriteLine, [.. GpuBenchSupport.KokoroFiles(), .. GpuBenchSupport.KokoroG2PFiles()])) return;

        using CpuBackend backend = new();
        using KokoroPipeline kokoro = await KokoroPipeline.LoadAsync();
        string ipa = GpuBenchSupport.KokoroG2P().ToIpa(Sentence);
        List<string> seen = [];
        kokoro.TestStageObserver = seen.Add;

        string reference = PcmDigest.Of(kokoro.Synthesize(backend, ipa, GpuBenchSupport.KokoroVoice));
        Assert.Equal(Boundaries, seen);
        using (CancellationTokenSource live = new())
        {
            Assert.Equal(reference, PcmDigest.Of(kokoro.Synthesize(backend, ipa, GpuBenchSupport.KokoroVoice, cancel: live.Token)));
        }
        _output.WriteLine($"uncancelled digest {reference}");

        for (int stop = 0; stop < Boundaries.Length; stop++)
        {
            using CancellationTokenSource cancel = new();
            string boundary = Boundaries[stop];
            seen.Clear();
            kokoro.TestStageObserver = stage =>
            {
                seen.Add(stage);
                if (stage == boundary)
                {
                    cancel.Cancel();
                }
            };
            Assert.Throws<OperationCanceledException>(() =>
                kokoro.Synthesize(backend, ipa, GpuBenchSupport.KokoroVoice, cancel: cancel.Token));
            Assert.Equal(Boundaries[..(stop + 1)], seen);
        }

        kokoro.TestStageObserver = null;
        Assert.Equal(reference, PcmDigest.Of(kokoro.Synthesize(backend, ipa, GpuBenchSupport.KokoroVoice)));
    }
}
