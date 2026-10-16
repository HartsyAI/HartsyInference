using HartsyInference.Audio.Streaming;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A streamed take hands back its turns as a one-shot take does, so it can be revised: the service passes the caller's
/// callback down to the runner, which calls it before the stream ends. The service's own stream path runs on a fake runner seeded in
/// the runner cache under Drama 3's key, so no weights load.</summary>
public sealed class SpeechStreamTurnsTests
{
    /// <summary>Drama 3's default build on one device: its repo, with no precision tag and no shard suffix.</summary>
    private const string Drama3Key = "fishaudio/drama-3";

    private static readonly ModelSpec Drama3 = new() { Requested = "fishaudio-drama3", Modality = Modality.Speech };

    private static readonly SpeechTurn[] Produced = [new() { Text = "Come closer.", Codes = [[5, 6], [1, 2]] }];

    [Fact]
    public async Task StreamedTake_HandsItsTurnsToTheCaller_BeforeTheStreamEnds()
    {
        using InferenceEngine engine = new("cpu");
        await SeedRunner(engine);
        IReadOnlyList<SpeechTurn>? received = null;
        int chunks = 0;
        await foreach (AudioChunk _ in engine.Speech.SynthesizeStreamAsync(Drama3, new SpeechRequest { Text = "Come closer." }, turns => received = turns))
        {
            chunks++;
        }
        Assert.Equal(2, chunks);
        Assert.Same(Produced, received);
    }

    [Fact]
    public async Task StreamWithoutACallback_StreamsTheSameChunks()
    {
        using InferenceEngine engine = new("cpu");
        await SeedRunner(engine);
        List<long> offsets = [];
        await foreach (AudioChunk chunk in engine.Speech.SynthesizeStreamAsync(Drama3, new SpeechRequest { Text = "Come closer." }))
        {
            offsets.Add(chunk.StartSampleOffset);
        }
        Assert.Equal([0L, 2L], offsets);
    }

    /// <summary>Seeds a runner that streams two buffers and then reports its turns, as Drama 3's does.</summary>
    private static Task<ITtsRunner> SeedRunner(InferenceEngine engine)
    {
        StreamingTtsRunner runner = new(24_000,
            (_, job) =>
            {
                job.TurnSink?.Invoke(Produced);
                return [0.1f];
            },
            (_, job, cancel) => AudioStreamer.FromProducer(emit =>
            {
                emit([0.1f, 0.2f]);
                emit([0.3f]);
                job.TurnSink?.Invoke(Produced);
            }, 24_000, cancel));
        return engine.AudioRuntime.Tts.GetOrLoadAsync(Drama3Key, _ => Task.FromResult<ITtsRunner>(runner), CancellationToken.None);
    }
}
