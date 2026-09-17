using System.Runtime.InteropServices;

using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.Mert2;
using HartsyInference.Audio.Models.SheetSage2;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Audio.Pipelines;

/// <summary>SheetSage2 end to end: a recording in, a two-voice lead sheet out.
///
/// <para>The encoder attends over a fixed five-minute window, so a longer song is read in overlapping passes and
/// the results are stitched onto one timeline. Each pass replays the tail of what came before as a prompt, which
/// is what keeps bar numbering and key continuous across a seam instead of restarting at every window.</para>
///
/// <para>The score is written out twice, with chord symbols and without. Both come from one decode: the decode is
/// the whole cost and serializing events already in hand is free, and the two are not a substitution apart, so
/// neither can be derived from the other afterwards.</para></summary>
public sealed class SheetSage2Pipeline : IDisposable
{
    /// <summary>Rate the model consumes; the service decodes the clip straight to it.</summary>
    public const int SampleRate = 24_000;

    /// <summary>Headroom the released decode demands between an overlap prompt and the token budget. A prompt
    /// that leaves less than this has no room to transcribe anything, so the pass is refused rather than run.</summary>
    private const int MinimumDecodeHeadroom = 128;

    private readonly Mert2AudioEncoder _encoder;
    private readonly SheetSage2Decoder _decoder;
    private readonly ScoreTokenizer _tokenizer;
    private readonly ScoreEventCodec _codec;
    private readonly WindowStitcher _stitcher;
    private readonly SafeTensorsLoader _loader;
    private bool _disposed;

    private SheetSage2Pipeline(Mert2AudioEncoder encoder, SheetSage2Decoder decoder, ScoreTokenizer tokenizer,
        SafeTensorsLoader loader)
    {
        _encoder = encoder;
        _decoder = decoder;
        _tokenizer = tokenizer;
        _codec = new ScoreEventCodec(tokenizer);
        _stitcher = new WindowStitcher(tokenizer);
        _loader = loader;
    }

    /// <summary>Files the repo contributes. The checkpoint sits under the YuE2 repo because SheetSage2 ships with
    /// it, and the category is <c>"music"</c> for the same reason — sharing that directory is what lets a machine
    /// that already has YuE2 avoid a second copy.</summary>
    public static IReadOnlyList<AudioModelFile> ModelFiles { get; } =
    [
        new("audio_encoders/sheetsage2_bf16.safetensors"),
    ];

    /// <summary>Loads the checkpoint, downloading it on first use.</summary>
    /// <param name="hfRepoId">Repo id; the released weights live in <c>"Comfy-Org/YuE2"</c>.</param>
    public static async Task<SheetSage2Pipeline> LoadAsync(string hfRepoId, CancellationToken ct = default)
    {
        IReadOnlyDictionary<string, string> fetched = await AudioModelCache
            .FetchAllAsync(hfRepoId, ModelFiles, category: "music", ct: ct).ConfigureAwait(false);

        SafeTensorsLoader loader = new();
        loader.Load(fetched["audio_encoders/sheetsage2_bf16.safetensors"]);
        Dictionary<string, Tensor> weights = loader.GetAllTensors();

        ScoreTokenizer tokenizer = new();
        Mert2AudioEncoder encoder = new(new Mert2Config());
        SheetSage2Decoder decoder = new(SheetSage2Config.Released, tokenizer);
        encoder.LoadWeights(weights);
        decoder.LoadWeights(weights);
        return new SheetSage2Pipeline(encoder, decoder, tokenizer, loader);
    }

    /// <summary>Transcribes a mono clip already at <see cref="SampleRate"/> into both renderings of its score.</summary>
    public ScoreTranscription Transcribe(IBackend backend, float[] audioMono)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(audioMono);
        double duration = (double)audioMono.Length / SampleRate;
        IReadOnlyList<SheetSage2Window> plan = SlidingWindowPlan.For(duration);
        List<StitchedScoreEvent> stitched = [];
        bool truncated = false;

        foreach (SheetSage2Window window in plan)
        {
            (List<int>? prefix, int subbeatBase) = _stitcher.OverlapPrefix(stitched, window);
            // The prompt and the transcription share one budget, so a long dense overlap can leave no room at
            // all. The released decode refuses that rather than returning a score that silently stops early.
            if (prefix is not null && prefix.Count >= _decoder.Config.MaxTokens - MinimumDecodeHeadroom)
            {
                throw new InvalidOperationException(
                    $"The overlap between windows fills SheetSage2's {_decoder.Config.MaxTokens}-token context "
                    + $"({prefix.Count} tokens at {window.Start:0.#}s). Transcribe shorter sections of this audio.");
            }

            int from = (int)Math.Round(window.Start * SampleRate);
            int to = Math.Min(audioMono.Length, (int)Math.Round(window.End * SampleRate));
            Tensor memory = _encoder.Encode(backend, audioMono.AsSpan(from, to - from));
            double stop = window.GenerationStop ?? Math.Min(duration - window.Start, SlidingWindowPlan.WindowSeconds);
            List<int> tokens = _decoder.GenerateTokens(backend, memory, stop, prefix is null ? default : CollectionsMarshal.AsSpan(prefix));
            truncated |= tokens.Count >= _decoder.Config.MaxTokens;

            IReadOnlyList<ScoreEvent> decoded = _codec.DecodeSequence(tokens);
            SubbeatTimeMap times = new(decoded);
            stitched.AddRange(_stitcher.WindowEvents(decoded, times, window, duration, subbeatBase));
        }

        // A window's time map is not guaranteed monotonic, so without this the finished stream can walk
        // backwards where two windows meet -- a scrambled score rather than an error.
        stitched.Sort(static (a, b) => a.Seconds != b.Seconds
            ? a.Seconds.CompareTo(b.Seconds) : a.GlobalSubbeat.CompareTo(b.GlobalSubbeat));

        List<TimedScoreEvent> timed = _stitcher.ToTimedEvents(stitched);
        Logs.Verbose($"[Audio][SheetSage2] {duration:0.0}s in {plan.Count} window(s) -> {timed.Count} events.");
        return new ScoreTranscription(
            AbcSerializer.EventsToAbc(timed, duration, melodyOnly: false),
            AbcSerializer.EventsToAbc(timed, duration),
            duration, plan.Count, truncated);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _encoder.Dispose();
        _decoder.Dispose();
        _loader.Dispose();
    }
}

/// <summary>A finished transcription: the score with chord symbols and without, plus how it was read.</summary>
/// <param name="FullAbc">The score carrying its chord symbols.</param>
/// <param name="MelodyAbc">The same score with no chord symbols, for melody-only rendering.</param>
/// <param name="Duration">Seconds of audio covered.</param>
/// <param name="WindowCount">Windows the clip was read in; more than one means the result was stitched.</param>
/// <param name="Truncated">True when a window hit the token budget, so the score stops short of the audio.</param>
public readonly record struct ScoreTranscription(
    string FullAbc, string MelodyAbc, double Duration, int WindowCount, bool Truncated);
