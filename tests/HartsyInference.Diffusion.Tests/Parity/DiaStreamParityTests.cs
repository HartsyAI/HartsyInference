using Xunit;
using Xunit.Abstractions;
using HartsyInference.Audio.Io;
using HartsyInference.Audio.Streaming;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Real end-to-end verification for Dia's streaming path (<c>DiaPipeline.DecodeSettledFramesTail</c>):
/// full re-decode of the utterance-so-far through DAC on every emission, keeping only the new trailing samples —
/// the same workaround nari-labs' own streaming attempt used for this architecture.
///
/// <para>DAC's decoder uses symmetric (non-causal) padding throughout, so a partial-prefix decode's samples near
/// the current right edge are computed against zero-padding where the monolithic decode sees real future frames —
/// each of this test's 69 chunk boundaries carries that contamination. A receptive-field-margin fix (hold back the
/// contaminated tail, only emit once a chunk stops being the rightmost) would remove it, but the margin for
/// DAC-44.1k's cumulative 512x upsample is tens of milliseconds of decode lookahead on a model whose 1720-frame AR
/// loop already takes 10+ minutes for 20s of audio — not worth building. Dia's streaming is registered on the
/// engine (<see cref="HartsyInference.Engine.Audio.DiaTtsModel"/>) but intentionally NOT wired to the extension's
/// <c>tts_streaming</c> flag; it stays text-chunked in the live UI.</para>
///
/// <para>This test is the regression guard: an index bug (verified fixed here — an earlier version of the
/// settlement loop was missing the BOS-prefill-row offset and produced relL2=1.42, i.e. two decorrelated signals)
/// would blow well past the 0.25 ceiling below and get caught.</para></summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class DiaStreamParityTests
{
    private readonly ITestOutputHelper _output;
    public DiaStreamParityTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task DiaStream_RealWeights_MatchesNonStreamingWithinBoundaryContaminationTolerance()
    {
        ModelSpec spec = ModelResolver.Resolve("dia", modelPathArg: null, Modality.Speech);
        SpeechRequest request = new SpeechRequest
        {
            Text = "[S1] The quick brown fox jumps over the lazy dog near the river bank. [S2] And then it kept running for quite a while longer, all the way down to the water.",
            Seed = 7,
        };

        using InferenceEngine engine = new InferenceEngine("cuda");

        List<float> streamed = [];
        int chunkCount = 0;
        await foreach (AudioChunk chunk in engine.Speech.SynthesizeStreamAsync(spec, request))
        {
            chunkCount++;
            if (chunk.Samples.Length == 0) continue;
            streamed.AddRange(chunk.Samples);
        }
        _output.WriteLine($"streamed: chunks={chunkCount} totalSamples={streamed.Count}");
        Assert.True(chunkCount > 0, "expected at least one chunk.");

        AudioResult monoResult = await engine.Speech.SynthesizeAsync(spec, request);
        using MemoryStream monoStream = new(monoResult.Data);
        WavFile.DecodedAudio decoded = WavFile.Read(monoStream);
        float[] mono = decoded.ToMono();
        _output.WriteLine($"monolithic: totalSamples={mono.Length}");

        // Sample-count parity is a hard requirement regardless of waveform closeness: it's what proves the
        // settlement/chunking logic never drops or duplicates a frame.
        Assert.Equal(mono.Length, streamed.Count);

        double maxAbs = 0, sumSq = 0, refSumSq = 0;
        int maxAbsIndex = -1;
        for (int i = 0; i < mono.Length; i++)
        {
            double diff = streamed[i] - mono[i];
            if (Math.Abs(diff) > maxAbs) { maxAbs = Math.Abs(diff); maxAbsIndex = i; }
            sumSq += diff * diff;
            refSumSq += mono[i] * (double)mono[i];
        }
        double relL2 = Math.Sqrt(sumSq / Math.Max(refSumSq, 1e-12));
        _output.WriteLine($"streamed vs monolithic: maxAbs={maxAbs:E4} relL2={relL2:E4} maxAbsIndex={maxAbsIndex}");

        // Loose tolerance by design: this measures DAC symmetric-padding boundary contamination across 69
        // chunk edges, not an approximation-free reconstruction. 0.25 sits above the measured 0.152 with margin
        // for seed/prompt variance while still catching an index regression (which reads as ~1.4, not ~0.15-0.25).
        Assert.True(relL2 < 0.25, $"streamed output diverges from monolithic non-streaming synthesis beyond the "
            + $"known DAC boundary-contamination band: maxAbs={maxAbs:E4} relL2={relL2:E4} at sample {maxAbsIndex}");
    }
}
