namespace HartsyInference.Diffusion.Models.Duo;

/// <summary>Packs conversations into the frame-v5 sequence in compact order: the reference, the shared Intent spans, each clean block's media, then per block its Intent rows and, when denoised, its media. Video rows come before audio rows within a group, and audio is channel-major.
/// Positions follow upstream <c>duo/data/sequence.py</c> and <c>text_positions.py</c> (@068e357). The 25 ms clock is 0.04 units per millisecond.</summary>
public static class DuoLayoutBuilder
{
    /// <summary>Packed length granularity; upstream pads to it.</summary>
    public const int Multiple = 128;

    /// <summary>Clock units per millisecond (one unit is 25 ms).</summary>
    public const double ClockPerMs = 0.04;

    /// <summary>Packs <paramref name="samples"/> in order. <paramref name="rope"/> is required because upstream's default is not the serving policy.</summary>
    /// <exception cref="ArgumentException">A conversation is malformed (ranges, frame counts, even spatial sizes).</exception>
    public static DuoLayout Build(IReadOnlyList<DuoSampleInput> samples, DuoRopePolicy rope)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Count == 0)
        {
            throw new ArgumentException("At least one conversation is needed.", nameof(samples));
        }

        Writer writer = new();
        int sourceBase = 0;
        for (int s = 0; s < samples.Count; s++)
        {
            DuoSampleInput sample = samples[s];
            Validate(sample);
            AddReference(writer, s, sample);
            AddSharedSpans(writer, s, sample, sourceBase, rope);

            int sharedTotal = sample.SharedDescriptionRows + sample.SharedCueRows;
            int bodySource = sourceBase + sharedTotal;
            for (int b = 0; b < sample.CleanBlocks; b++)
            {
                AddCleanBlock(writer, s, b, sample.Blocks[b]);
            }
            for (int b = 0; b < sample.Blocks.Count; b++)
            {
                DuoBlockInput block = sample.Blocks[b];
                AddBlockBody(writer, s, b, block, bodySource, rope);
                bodySource += block.BodyTextRows;
                if (sample.NoisyBlocks.Contains(b))
                {
                    AddNoisyBlock(writer, s, b, block);
                }
            }

            sourceBase = bodySource;
        }

        return writer.Finish();
    }

    /// <summary>The portrait reference: one video frame, timeline 0.</summary>
    private static void AddReference(Writer writer, int sample, DuoSampleInput input) =>
        writer.AddMedia(sample, -1, DuoGroupKind.Reference, input.Reference, [0.0], audioSteps: 0, startMs: 0.0, timeId: 0);

    /// <summary>The two shared Intent spans (description, then cue), read from the text source starting at <paramref name="sourceBase"/>.</summary>
    private static void AddSharedSpans(Writer writer, int sample, DuoSampleInput input, int sourceBase, DuoRopePolicy rope)
    {
        int descriptionRows = input.SharedDescriptionRows, cueRows = input.SharedCueRows, sharedTotal = descriptionRows + cueRows;
        if (descriptionRows > 0)
        {
            writer.AddText(sample, -1, DuoGroupKind.Shared, SharedPositions(rope, -sharedTotal, descriptionRows), sourceBase);
        }
        if (cueRows > 0)
        {
            writer.AddText(sample, -1, DuoGroupKind.Shared, SharedPositions(rope, -cueRows, cueRows), sourceBase + descriptionRows);
        }
    }

    /// <summary>A committed clean block: its media only, timeline 0.</summary>
    private static void AddCleanBlock(Writer writer, int sample, int block, DuoBlockInput input) =>
        writer.AddMedia(sample, block, DuoGroupKind.Clean, input.Video, input.FrameStartsMs, input.AudioSteps, input.StartMs, timeId: 0);

    /// <summary>A block's body Intent rows, read from the text source starting at <paramref name="source"/>, when it has any.</summary>
    private static void AddBlockBody(Writer writer, int sample, int block, DuoBlockInput input, int source, DuoRopePolicy rope)
    {
        if (input.BodyTextRows > 0)
        {
            writer.AddText(sample, block, DuoGroupKind.Text, BodyPositions(rope, input), source);
        }
    }

    /// <summary>A noisy block to denoise: its media with the next noisy timeline id.</summary>
    private static void AddNoisyBlock(Writer writer, int sample, int block, DuoBlockInput input) =>
        writer.AddMedia(sample, block, DuoGroupKind.Noisy, input.Video, input.FrameStartsMs, input.AudioSteps, input.StartMs,
            timeId: writer.NextNoisyTimeId());

    /// <summary>Intent positions of the shared spans. Under <see cref="DuoRopePolicy.Old"/> every shared row is -1; under <see cref="DuoRopePolicy.New"/> the two spans end at -1 with no gap.</summary>
    private static double[] SharedPositions(DuoRopePolicy rope, int start, int rows)
    {
        double[] t = new double[rows];
        for (int j = 0; j < rows; j++)
        {
            t[j] = rope == DuoRopePolicy.Old ? -1.0 : start + j;
        }
        return t;
    }

    private static double[] BodyPositions(DuoRopePolicy rope, DuoBlockInput block)
    {
        double start = block.StartMs * ClockPerMs;
        double[] t = new double[block.BodyTextRows];
        for (int j = 0; j < t.Length; j++)
        {
            t[j] = rope == DuoRopePolicy.Old ? start : start + j;
        }
        return t;
    }

    /// <summary>The cache prefill for one conversation: the portrait reference and the two shared Intent spans, the rows
    /// <see cref="Build"/> packs before any block. Positions and tags match <see cref="Build"/>, so a cached forward over this
    /// layout commits the same keys and values the full layout does.</summary>
    internal static DuoLayout BuildPrefix(DuoSampleInput sample, DuoRopePolicy rope)
    {
        Validate(sample);
        Writer writer = new();
        AddReference(writer, 0, sample);
        AddSharedSpans(writer, 0, sample, 0, rope);
        return writer.Finish();
    }

    /// <summary>One block as the cached path sees it: its body Intent rows and either its noisy media (<paramref name="clean"/>
    /// false, a denoise pass) or its clean media (a commit, timeline id 0). The text source rows index the chunk's own text
    /// tensor, which is the shared rows followed by this block's body rows.</summary>
    internal static DuoLayout BuildChunk(DuoSampleInput sample, int block, DuoRopePolicy rope, bool clean)
    {
        Validate(sample);
        if ((uint)block >= (uint)sample.Blocks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(block), block, "The block is not in the sample.");
        }
        DuoBlockInput input = sample.Blocks[block];
        Writer writer = new();
        AddBlockBody(writer, 0, block, input, sample.SharedDescriptionRows + sample.SharedCueRows, rope);
        if (clean)
        {
            AddCleanBlock(writer, 0, block, input);
        }
        else
        {
            AddNoisyBlock(writer, 0, block, input);
        }
        return writer.Finish();
    }

    /// <summary>Spatial axes of a latent grid: <c>((1 - d/area)/2 + i*2/area) * 32</c> for <c>d/2</c> patches per axis, with <c>area = sqrt(H*W)</c>.</summary>
    internal static (double[] Height, double[] Width) SpatialAxes(int height, int width)
    {
        double area = Math.Sqrt((double)height * width);
        return (Axis(height, area), Axis(width, area));
    }

    private static double[] Axis(int dim, double area)
    {
        double[] axis = new double[dim / 2];
        for (int i = 0; i < axis.Length; i++)
        {
            axis[i] = ((1.0 - dim / area) / 2.0 + i * 2.0 / area) * 32.0;
        }
        return axis;
    }

    private static void Validate(DuoSampleInput sample)
    {
        if (sample.Blocks.Count == 0)
        {
            throw new ArgumentException($"Conversation {sample.SampleId} has no blocks.", nameof(sample));
        }
        if (sample.SharedDescriptionRows < 0 || sample.SharedCueRows < 0)
        {
            throw new ArgumentException($"Conversation {sample.SampleId} has a negative shared row count.", nameof(sample));
        }
        ValidateGrid(sample.Reference, $"Conversation {sample.SampleId} reference");
        if (sample.Reference.Frames != 1)
        {
            throw new ArgumentException($"Conversation {sample.SampleId} reference must be one latent frame.", nameof(sample));
        }
        if (sample.CleanBlocks < 0 || sample.CleanBlocks > sample.Blocks.Count)
        {
            throw new ArgumentOutOfRangeException(nameof(sample), sample.CleanBlocks, "CleanBlocks must be within the block count.");
        }
        HashSet<int> noisy = [];
        foreach (int b in sample.NoisyBlocks)
        {
            if (b < 0 || b >= sample.Blocks.Count || !noisy.Add(b))
            {
                throw new ArgumentException($"Conversation {sample.SampleId} names an invalid or repeated noisy block {b}.", nameof(sample));
            }
        }
        for (int b = 0; b < sample.Blocks.Count; b++)
        {
            DuoBlockInput block = sample.Blocks[b];
            ValidateGrid(block.Video, $"Conversation {sample.SampleId} block {b} video");
            if (block.FrameStartsMs.Length != block.Video.Frames)
            {
                throw new ArgumentException(
                    $"Conversation {sample.SampleId} block {b} has {block.FrameStartsMs.Length} frame starts for {block.Video.Frames} frames.", nameof(sample));
            }
            if (block.BodyTextRows < 0 || block.AudioSteps < 0)
            {
                throw new ArgumentException($"Conversation {sample.SampleId} block {b} has a negative row count.", nameof(sample));
            }
        }
    }

    private static void ValidateGrid(DuoGrid grid, string what)
    {
        if (grid.Frames <= 0 || grid.Height <= 0 || grid.Width <= 0 || grid.Height % 2 != 0 || grid.Width % 2 != 0)
        {
            throw new ArgumentException($"{what} must be [T, H, W] with positive frames and even positive H and W, got {grid}.");
        }
    }

    private sealed class Writer
    {
        private readonly List<DuoGroup> _groups = [];
        private readonly List<double> _positions = [];
        private readonly List<int> _tags = [];
        private readonly List<int> _timeIds = [];
        private readonly List<int> _videoRows = [];
        private readonly List<int> _audioRows = [];
        private readonly List<int> _textRows = [];
        private readonly List<int> _textSource = [];
        private readonly List<int> _noisyGroups = [];
        private int _noisyCount;

        private int Cursor => _tags.Count;

        /// <summary>The next 1-based time id for a noisy block, in layout order across the pack.</summary>
        public int NextNoisyTimeId() => ++_noisyCount;

        public void AddMedia(int sample, int block, DuoGroupKind kind, DuoGrid grid, IReadOnlyList<double> frameStartsMs,
            int audioSteps, double startMs, int timeId)
        {
            (double[] hAxis, double[] wAxis) = DuoLayoutBuilder.SpatialAxes(grid.Height, grid.Width);
            AddVideo(sample, block, kind, grid.Frames, hAxis, wAxis, frameStartsMs, timeId);
            AddAudio(sample, block, kind, audioSteps, wAxis, startMs, timeId);
        }

        public void AddText(int sample, int block, DuoGroupKind kind, double[] tPositions, int sourceStart)
        {
            int start = Cursor;
            for (int j = 0; j < tPositions.Length; j++)
            {
                _positions.Add(tPositions[j]);
                _positions.Add(0.0);
                _positions.Add(0.0);
                _tags.Add(1);
                _timeIds.Add(0);
                _textRows.Add(start + j);
                _textSource.Add(sourceStart + j);
            }
            _groups.Add(new DuoGroup(sample, block, kind, DuoModality.None, start, Cursor));
        }

        private void AddVideo(int sample, int block, DuoGroupKind kind, int frames, double[] hAxis, double[] wAxis,
            IReadOnlyList<double> frameStartsMs, int timeId)
        {
            int start = Cursor;
            for (int f = 0; f < frames; f++)
            {
                // The frame time is rounded to F32 before storage, as upstream's torch.tensor of the frame spans is.
                double t = (float)(frameStartsMs[f] * DuoLayoutBuilder.ClockPerMs);
                for (int i = 0; i < hAxis.Length; i++)
                {
                    for (int j = 0; j < wAxis.Length; j++)
                    {
                        _videoRows.Add(Cursor);
                        _positions.Add(t);
                        _positions.Add(hAxis[i]);
                        _positions.Add(wAxis[j]);
                        _tags.Add(0);
                        _timeIds.Add(timeId);
                    }
                }
            }
            AddGroup(sample, block, kind, DuoModality.Video, start);
        }

        private void AddAudio(int sample, int block, DuoGroupKind kind, int steps, double[] wAxis, double startMs, int timeId)
        {
            if (steps == 0)
            {
                return;
            }
            int start = Cursor;
            double baseT = startMs * DuoLayoutBuilder.ClockPerMs;
            for (int channel = 0; channel < 2; channel++)
            {
                double w = channel == 0 ? wAxis[0] : wAxis[^1];
                for (int k = 0; k < steps; k++)
                {
                    _audioRows.Add(Cursor);
                    _positions.Add(baseT + k);
                    _positions.Add(0.0);
                    _positions.Add(w);
                    _tags.Add(2);
                    _timeIds.Add(timeId);
                }
            }
            AddGroup(sample, block, kind, DuoModality.Audio, start);
        }

        private void AddGroup(int sample, int block, DuoGroupKind kind, DuoModality modality, int start)
        {
            _groups.Add(new DuoGroup(sample, block, kind, modality, start, Cursor));
            if (kind == DuoGroupKind.Noisy)
            {
                _noisyGroups.Add(_groups.Count - 1);
            }
        }

        public DuoLayout Finish()
        {
            int live = Cursor;
            if (live == 0)
            {
                throw new ArgumentException("The packed sequence has no rows.");
            }
            int padded = (live + DuoLayoutBuilder.Multiple - 1) / DuoLayoutBuilder.Multiple * DuoLayoutBuilder.Multiple;
            if (padded > live)
            {
                int start = live;
                for (int row = live; row < padded; row++)
                {
                    _positions.Add(0.0);
                    _positions.Add(0.0);
                    _positions.Add(0.0);
                    _tags.Add(-1);
                    _timeIds.Add(0);
                }
                _groups.Add(new DuoGroup(-1, -1, DuoGroupKind.Padding, DuoModality.None, start, padded));
            }
            return new DuoLayout(_groups, live, padded, _positions.ToArray(), _tags.ToArray(), _timeIds.ToArray(),
                _videoRows.ToArray(), _audioRows.ToArray(), _textRows.ToArray(), _textSource.ToArray(), _noisyGroups);
        }
    }
}
