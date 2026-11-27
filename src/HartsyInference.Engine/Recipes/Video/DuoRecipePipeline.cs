using HartsyInference.Core.Backends;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>A constructed Duo pipeline driven against the native <see cref="VideoRequest"/>. Each generation reads one run folder
/// (shared and per-block Intent rows, reference latent, noise), runs a fresh <see cref="DuoSession"/> with a sliding clean-block history,
/// and returns the 24 fps frames with their 32 kHz stereo audio. The prompt and the seed are not used: the folder holds the condition and
/// the noise. Takes ownership of the transformer and every disposable it is given.</summary>
public sealed class DuoRecipePipeline : IVideoRecipePipeline
{
    /// <summary><see cref="VideoRequest.Extra"/> key for the run folder (see <see cref="DuoRunFiles"/>). Required.</summary>
    public const string RunFolderKey = "DuoRunFolder";

    /// <summary><see cref="VideoRequest.Extra"/> key for the history window in frames, counting the current block; -1 keeps all history.
    /// The history is held in F32 on the backend, so this bounds its memory.</summary>
    public const string WindowFramesKey = "DuoWindowFrames";

    /// <summary>Upstream's sliding-window option default: seven clean 17-frame blocks of history behind the current one. Its demo runtime
    /// passes 238.</summary>
    public const int DefaultWindowFrames = 144;

    private readonly IBackend _backend;
    private readonly MiniMaxH3Transformer _transformer;
    private readonly MiniMaxH3Config _config;
    private readonly DuoCodecs _codecs;
    private readonly long _ditResidentBytes;
    private readonly bool _distilled;
    private readonly IReadOnlyList<IDisposable> _owned;
    private readonly ResidentWeights _dit;
    private readonly ResidentWeights _codecWeights;
    private int _disposed;

    /// <param name="ditResidentBytes">What the resident DiT needs beside the history: its weights and the activation reserve.</param>
    internal DuoRecipePipeline(IBackend backend, MiniMaxH3Transformer transformer, MiniMaxH3Config config, DuoCodecs codecs,
        long ditResidentBytes, IReadOnlyList<IDisposable> owned, bool distilled = false)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _transformer = transformer ?? throw new ArgumentNullException(nameof(transformer));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _codecs = codecs ?? throw new ArgumentNullException(nameof(codecs));
        _ditResidentBytes = ditResidentBytes;
        _distilled = distilled;
        _owned = owned ?? throw new ArgumentNullException(nameof(owned));
        _dit = new ResidentWeights(_backend, _transformer.EnumerateWeights);
        _codecWeights = new ResidentWeights(_backend, _codecs.Weights);
    }

    /// <inheritdoc/>
    public VideoGenerationResult Generate(VideoRequest request, VideoEffectiveSettings settings, IProgress<StepPreview>? progress,
        CancellationToken cancel)
    {
        ArgumentNullException.ThrowIfNull(request);
        ArgumentNullException.ThrowIfNull(settings);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        cancel.ThrowIfCancellationRequested();
        DuoRunInput input = DuoRunInput.Read(request, settings, _config, _distilled);
        input.RequireHistoryFits(_config, _backend.GetDeviceMemoryUsage());
        int totalSteps = input.Sample.Blocks.Count * input.Run.Steps;
        Logs.Info($"[Duo] {input.Sample.Blocks.Count} blocks, {input.Run.Steps} steps each, {input.CacheRows} history rows; "
            + "the prompt and seed are unused (Intent rows and noise come from the run folder).");

        try
        {
            TryPreload(_codecWeights, "codec");
            // A DiT kept from an earlier run already takes its share of the free memory, so the fit test would turn it down.
            if (_dit.IsResident || DitFits(input.HistoryBytes(_config)))
            {
                TryPreload(_dit, "DiT");
            }
            using DuoStreamCache cache = new(_backend, _config.NumLayers,
                _config.NumAttentionHeads * _config.AttentionHeadDim, input.CacheRows);
            using DuoVideoStreamDecoder video = _codecs.Video(_backend);
            using DuoAudioStreamDecoder audio = _codecs.Audio(_backend);
            DuoSession session = new(_transformer, _backend, _config, input.Sample, DuoRopePolicy.New, cache, video, audio);
            Action<int>? onStep = progress is null
                ? null
                : done => progress.Report(new StepPreview { Step = done, TotalSteps = totalSteps });
            DuoOutput output = session.Run(input.Run, onStep, cancel);
            return Result(output, request, settings);
        }
        catch (OutOfVramException)
        {
            // The engine frees all device memory after this, so the DiT is no longer held and the next run runs the fit test again.
            _dit.Release(force: true);
            throw;
        }
        finally
        {
            EndRun(_codecWeights, _codecs.Weights);
            EndRun(_dit, _transformer.EnumerateWeights);
        }
    }

    /// <summary>Releases a preloaded model as the VRAM profile says. One that was not preloaded can still have weights on the device, because
    /// CUDA makes a weight resident the second time an op uploads it; those are freed here, or they would stay while the pipeline is
    /// cached.</summary>
    private void EndRun(ResidentWeights model, Func<IEnumerable<Tensor>> weights)
    {
        if (model.IsResident)
        {
            model.Release();
        }
        else
        {
            // A run that threw or was cancelled can leave work queued that still reads them.
            _backend.Sync();
            _backend.FreeWeights(weights());
        }
    }

    /// <summary>The first sample of <paramref name="block"/> on the absolute clock, <c>ceil(block · 17 · 32000 / 24)</c>.</summary>
    internal static long FirstSample(int block)
    {
        long numerator = (long)block * DuoCodecSpec.Get("duo_5_17").Frames * DuoCodecSpec.SampleRate;
        return (numerator + DuoCodecSpec.Fps - 1) / DuoCodecSpec.Fps;
    }

    /// <summary>Joins the decoded blocks, each <c>[channel][sample]</c>, keeping each block's own span of the clock from its front, as
    /// upstream does: a block decodes to 29 latent steps (23200 samples) but lasts 22666 or 22667.</summary>
    internal static AudioBuffer Soundtrack(IReadOnlyList<float[]> blocks)
    {
        int total = checked((int)FirstSample(blocks.Count));
        float[][] channels = [new float[total], new float[total]];
        for (int block = 0; block < blocks.Count; block++)
        {
            int start = (int)FirstSample(block), keep = (int)FirstSample(block + 1) - start;
            int decoded = blocks[block].Length / 2;
            if (decoded < keep)
            {
                throw new InvalidOperationException($"Audio block {block} decoded {decoded} samples per channel; its span needs {keep}.");
            }
            blocks[block].AsSpan(0, keep).CopyTo(channels[0].AsSpan(start));
            blocks[block].AsSpan(decoded, keep).CopyTo(channels[1].AsSpan(start));
        }
        return new AudioBuffer { Channels = channels, SampleRate = DuoCodecSpec.SampleRate };
    }

    private static VideoGenerationResult Result(DuoOutput output, VideoRequest request, VideoEffectiveSettings settings)
    {
        int frameBytes = checked(settings.Width * settings.Height * 3);
        List<byte[]> frames = new(settings.Frames);
        foreach (DuoDecodedBlock block in output.Video)
        {
            if (block.Width != settings.Width || block.Height != settings.Height)
            {
                throw new InvalidOperationException(
                    $"The codec returned {block.Width}x{block.Height} frames for a {settings.Width}x{settings.Height} request.");
            }
            for (int frame = 0; frame < block.Frames; frame++)
            {
                frames.Add(block.Rgb.AsSpan(frame * frameBytes, frameBytes).ToArray());
            }
        }
        return VideoRecipeUtils.ToResult([.. frames], settings.Width, settings.Height, request, Soundtrack(output.Audio),
            DuoCodecSpec.Fps);
    }

    /// <summary>Whether the DiT can stay resident beside this run's F32 history. A backend that reports no free figure is taken to fit.</summary>
    private bool DitFits(long historyBytes)
    {
        (long freeBytes, _) = _backend.GetVramInfo();
        bool fits = freeBytes <= 0 || freeBytes >= _ditResidentBytes + historyBytes;
        Logs.Info($"[Duo] DiT {_ditResidentBytes / 1e9:F2} GB with reserve + history {historyBytes / 1e9:F2} GB vs "
            + $"{freeBytes / 1e9:F2} GB free: " + (fits ? "resident" : "streaming"));
        return fits;
    }

    private static void TryPreload(ResidentWeights weights, string label)
    {
        try
        {
            weights.Preload();
        }
        catch (OutOfVramException ex)
        {
            // Residency is an optimization: without it each operation uploads its own weights.
            Logs.Warning($"[Duo] {label} preload did not fit ({ex.Message}); streaming per call.");
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _transformer.Dispose();
        for (int i = _owned.Count - 1; i >= 0; i--)
        {
            _owned[i].Dispose();
        }
    }
}
