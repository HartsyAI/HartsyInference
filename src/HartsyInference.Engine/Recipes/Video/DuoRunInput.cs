using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Requests;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>One Duo generation's input, checked against the request before anything is computed: the packed sample, the run read from its
/// folder, and the history rows the cache must hold.</summary>
internal sealed record DuoRunInput(DuoSampleInput Sample, DuoRun Run, int CacheRows)
{
    private const int PixelsPerLatent = 16;
    private const int SizeMultiple = 32;

    /// <summary>Reads the run folder named by the request. The geometry comes from the settings and the Intent row counts from the folder's
    /// file lengths; <see cref="DuoRunFiles.Read"/> then checks every file.</summary>
    /// <exception cref="ArgumentException">The request asks for something Duo does not do, or disagrees with the run folder.</exception>
    /// <exception cref="InvalidDataException">The run folder is malformed.</exception>
    public static DuoRunInput Read(VideoRequest request, VideoEffectiveSettings settings, MiniMaxH3Config config)
    {
        RefuseUnsupported(request, settings);
        string folder = Option(request, DuoRecipePipeline.RunFolderKey)
            ?? throw new ArgumentException(
                $"Duo needs a run folder: set Extra[\"{DuoRecipePipeline.RunFolderKey}\"] to a folder with manifest.json.", nameof(request));
        int window = WindowFrames(request);
        DuoCodecSpec codec = DuoCodecSpec.Get("duo_5_17");
        int blocks = settings.Frames / codec.Frames;

        DuoRunShape shape = DuoRunFiles.ReadShape(folder, config.TextDim);
        if (shape.BodyRows.Count != blocks)
        {
            int held = shape.BodyRows.Count;
            throw new ArgumentException(
                $"The run folder holds {held} blocks ({held * codec.Frames} frames); the request asks for {settings.Frames} frames.",
                nameof(settings));
        }
        if (shape.SharedRows == 0)
        {
            throw new InvalidDataException("sharedText holds no rows; a Duo run needs the shared Intent rows.");
        }
        DuoGrid reference = new(1, settings.Height / PixelsPerLatent, settings.Width / PixelsPerLatent);
        DuoBlockInput[] inputs = new DuoBlockInput[blocks];
        for (int block = 0; block < blocks; block++)
        {
            inputs[block] = Block(codec, block, shape.BodyRows[block], reference with { Frames = codec.Latents });
        }
        // The two shared spans take the same positions and see the same rows as one, so the whole span is passed as the description.
        DuoSampleInput sample = new()
        {
            SampleId = 0, SharedDescriptionRows = shape.SharedRows, SharedCueRows = 0, Reference = reference, Blocks = inputs,
            CleanBlocks = 0, NoisyBlocks = [],
        };

        int cacheRows = HistoryRows(sample, window == -1 ? blocks : Math.Min((window / codec.Frames) - 1, blocks));
        DuoRun run = DuoRunFiles.Read(folder, sample, config.TextDim, config.VideoPatchDim, config.AudioLatentsDim);
        float shift = settings.FlowShift ?? (float)DuoSchedule.CausalShift;
        if (run.Steps != settings.Steps || (float)run.Shift != shift)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"The run folder was made for {run.Steps} steps at shift {run.Shift}; the request asks for {settings.Steps} at {shift}."),
                nameof(settings));
        }
        return new DuoRunInput(sample, run, cacheRows);
    }

    /// <summary>Bytes of the history this run keeps on the backend: F32 keys and values for every row of every block.</summary>
    public long HistoryBytes(MiniMaxH3Config config) =>
        (long)CacheRows * config.NumLayers * 2 * config.NumAttentionHeads * config.AttentionHeadDim * sizeof(float);

    /// <summary>Refuses a history that cannot fit the memory the backend can still hand out (free plus its pool's unused reserve). A
    /// backend that reports no memory figure is not checked.</summary>
    /// <exception cref="ArgumentException">The history is larger than that.</exception>
    public void RequireHistoryFits(MiniMaxH3Config config, DeviceMemoryUsage memory)
    {
        if (memory.TotalBytes <= 0)
        {
            return;
        }
        long available = memory.FreeBytes + Math.Max(0, memory.PoolReservedBytes - memory.PoolUsedBytes);
        long history = HistoryBytes(config);
        if (history > available)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"The Duo history needs {history / 1e9:F1} GB on the device and {available / 1e9:F1} GB is available. Lower "
                + $"Extra[\"{DuoRecipePipeline.WindowFramesKey}\"] or the resolution."));
        }
    }

    /// <summary>Block <paramref name="index"/> on the 24 fps clock: each latent frame starts where the frames before it end.</summary>
    private static DuoBlockInput Block(DuoCodecSpec codec, int index, int bodyRows, DuoGrid video)
    {
        int first = index * codec.Frames;
        double[] starts = new double[codec.Latents];
        int frame = first;
        for (int latent = 0; latent < starts.Length; latent++)
        {
            starts[latent] = frame * 1000.0 / DuoCodecSpec.Fps;
            frame += codec.FrameSpans[latent];
        }
        return new DuoBlockInput
        {
            StartMs = first * 1000.0 / DuoCodecSpec.Fps, BodyTextRows = bodyRows, Video = video, AudioSteps = codec.AudioLatents,
            FrameStartsMs = starts,
        };
    }

    /// <summary>The reference, the shared rows and <paramref name="historyBlocks"/> clean blocks. Refuses a run whose history plus current
    /// block passes the cached forward's row limit, before any work.</summary>
    private static int HistoryRows(DuoSampleInput sample, int historyBlocks)
    {
        DuoBlockInput first = sample.Blocks[0];
        int blockRows = first.Video.VideoRows + (2 * first.AudioSteps);
        long rows = sample.Reference.VideoRows + sample.SharedDescriptionRows + ((long)historyBlocks * blockRows);
        int bodyRows = sample.Blocks.Max(b => b.BodyTextRows);
        int multiple = DuoLayoutBuilder.Multiple;
        long current = ((long)bodyRows + blockRows + multiple - 1) / multiple * multiple;
        if (rows + current > MiniMaxH3Transformer.MaxDuoDenseBiasRows)
        {
            throw new ArgumentException(
                $"The history ({rows} rows) and one block ({current} rows) pass the cached forward's {MiniMaxH3Transformer.MaxDuoDenseBiasRows} "
                + $"rows. Lower Extra[\"{DuoRecipePipeline.WindowFramesKey}\"] or the resolution.", nameof(sample));
        }
        return (int)rows;
    }

    private static void RefuseUnsupported(VideoRequest request, VideoEffectiveSettings settings)
    {
        if (!string.IsNullOrWhiteSpace(request.Prompt) || !string.IsNullOrWhiteSpace(request.NegativePrompt))
        {
            throw new ArgumentException(
                "Duo reads no prompt or negative prompt: the Intent rows come from the run folder. Leave both empty.", nameof(request));
        }
        if (settings.CfgScale != 1f)
        {
            throw new ArgumentException("Duo samples without guidance; the guidance scale must be 1.", nameof(settings));
        }
        if ((settings.Sampler is not null && !settings.Sampler.Equals("euler", StringComparison.OrdinalIgnoreCase))
            || settings.Scheduler is not null || settings.AudioFlowShift is not null)
        {
            throw new ArgumentException(
                "Duo runs its released Euler grid with one flow shift; leave the sampler, scheduler and audio flow shift unset.",
                nameof(settings));
        }
        if (settings.Fps != DuoCodecSpec.Fps || (request.Fps is int fps && fps != DuoCodecSpec.Fps))
        {
            throw new ArgumentException($"Duo runs at {DuoCodecSpec.Fps} fps; its audio is cut to that clock.", nameof(settings));
        }
        if (request.TrimVideoStartFrames != 0 || request.TrimVideoEndFrames != 0 || request.VideoBoomerang)
        {
            throw new ArgumentException("Duo returns audio on the video clock; trim and boomerang would break the sync.", nameof(request));
        }
        if (settings.Width <= 0 || settings.Height <= 0 || settings.Width % SizeMultiple != 0 || settings.Height % SizeMultiple != 0)
        {
            throw new ArgumentException(
                $"Duo sizes are multiples of {SizeMultiple}; got {settings.Width}x{settings.Height}.", nameof(settings));
        }
        if (settings.Frames <= 0 || settings.Frames % DuoCodecSpec.Get("duo_5_17").Frames != 0)
        {
            throw new ArgumentException($"Duo generates whole 17-frame blocks; got {settings.Frames} frames.", nameof(settings));
        }
    }

    private static int WindowFrames(VideoRequest request)
    {
        string? text = Option(request, DuoRecipePipeline.WindowFramesKey);
        if (text is null)
        {
            return DuoRecipePipeline.DefaultWindowFrames;
        }
        // A window of one block keeps no room to commit the block, so two is the smallest.
        int smallest = 2 * DuoCodecSpec.Get("duo_5_17").Frames;
        return int.TryParse(text, NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out int frames)
            && (frames == -1 || frames >= smallest)
            ? frames
            : throw new ArgumentException(
                $"Extra[\"{DuoRecipePipeline.WindowFramesKey}\"] must be -1 (all history) or at least {smallest} frames; got '{text}'.",
                nameof(request));
    }

    private static string? Option(VideoRequest request, string key)
    {
        string? text = request.Extra.TryGetValue(key, out object? value) ? Convert.ToString(value, CultureInfo.InvariantCulture) : null;
        return string.IsNullOrWhiteSpace(text) ? null : text;
    }
}
