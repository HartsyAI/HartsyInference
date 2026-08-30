namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The static packed-sequence structure for one shape + conditioning signature: segment table, per-row
/// (t, h, w) position ids, and the masks marking which video/audio rows are the denoised targets rather than
/// conditioning. Positions are built in double precision and only narrowed when RoPE angles are formed.</summary>
public sealed class MiniMaxH3PackedLayout
{
    /// <summary>Latent frames each video token spans, cycling every 5 tokens.</summary>
    private static readonly int[] FramePerToken = { 1, 4, 4, 4, 4 };

    private const double FrameRescale = 5.0 / 3.0;

    public int SequenceLength { get; }

    /// <summary>Per-row (t, h, w) coordinates, row-major <c>[S, 3]</c>.</summary>
    public double[] PositionIds { get; }

    public IReadOnlyList<MiniMaxH3Segment> Segments { get; }

    /// <summary>True where a video row is a denoise target rather than a conditioning row.</summary>
    public bool[] ImageUpdate { get; }

    /// <summary>True where an audio row is a denoise target rather than a conditioning row.</summary>
    public bool[] AudioUpdate { get; }

    public (int TextLen, int LatentT, int LatentH, int LatentW, int AudioT) Signature { get; }

    /// <summary>Builds the segment and position tables for one complete packed sequence.</summary>
    public MiniMaxH3PackedLayout(int textLen, int latentT, int latentH, int latentW, int audioT,
        IReadOnlyList<MiniMaxH3Keyframe>? keyframes = null, IReadOnlyList<MiniMaxH3RefBlock>? refs = null,
        int? frameCount = null)
    {
        (double[] frame, double[] wGrid) = FrameGrid(latentH, latentW);
        int frameRows = frame.Length / 2;

        List<MiniMaxH3Segment> segments = new List<MiniMaxH3Segment>();
        List<double[]> positions = new List<double[]>();
        List<bool> imageUpdate = new List<bool>();
        List<bool> audioUpdate = new List<bool>();

        double[] textPos = new double[textLen * 3];
        for (int i = 0; i < textLen; i++)
        {
            textPos[i * 3] = i;
        }
        int row = 0;
        Append(segments, positions, ref row, MiniMaxH3SegmentKind.Text, textLen, textPos);

        double targetCursor = textLen;
        if (refs is { Count: > 0 })
        {
            foreach (MiniMaxH3RefBlock blk in refs)
            {
                targetCursor += ReferenceTimeSpan(blk);
            }
        }

        if (keyframes is { Count: > 0 })
        {
            foreach (MiniMaxH3Keyframe kf in keyframes)
            {
                if (kf.ResolvedFrameIndex < 0 || frameCount is int count && kf.ResolvedFrameIndex >= count)
                {
                    throw new ArgumentOutOfRangeException(nameof(keyframes), kf.ResolvedFrameIndex,
                        frameCount is int total
                            ? $"MiniMax-H3 guide frame must be in [0, {total}); got {kf.ResolvedFrameIndex}."
                            : $"MiniMax-H3 guide frame must be non-negative; got {kf.ResolvedFrameIndex}.");
                }
                if (kf.VideoLatentFrames < 0 || kf.AudioLatentFrames < 0
                    || kf.VideoLatentFrames == 0 && kf.AudioLatentFrames == 0)
                {
                    throw new ArgumentException(
                        "A MiniMax-H3 guide must carry at least one visual or audio latent row.", nameof(keyframes));
                }

                double condT = targetCursor + FrameRescale * kf.ResolvedFrameIndex;
                if (kf.VideoLatentFrames > 0)
                {
                    int videoCondRows = kf.VideoLatentFrames * frameRows;
                    Append(segments, positions, ref row, MiniMaxH3SegmentKind.Cond, videoCondRows,
                        VideoGrid(kf.VideoLatentFrames, frame, condT));
                    imageUpdate.AddRange(Enumerable.Repeat(false, videoCondRows));
                }
                if (kf.AudioLatentFrames > 0)
                {
                    int audioCondRows = kf.AudioLatentFrames * 2;
                    Append(segments, positions, ref row, MiniMaxH3SegmentKind.CondAudio, audioCondRows,
                        AudioGrid(condT, kf.AudioLatentFrames, wGrid[0], wGrid[^1]));
                    audioUpdate.AddRange(Enumerable.Repeat(false, audioCondRows));
                }
            }
        }

        double targetWLow = wGrid[0], targetWHigh = wGrid[^1];
        double cursor = textLen;
        if (refs is { Count: > 0 })
        {
            foreach (MiniMaxH3RefBlock blk in refs)
            {
                if (blk.Kind == "image")
                {
                    (double[] rFrame, _) = FrameGrid(PadToPatch(blk.LatentH, 2), PadToPatch(blk.LatentW, 2));
                    int n = rFrame.Length / 2;
                    double[] g = new double[n * 3];
                    for (int i = 0; i < n; i++)
                    {
                        g[i * 3] = cursor;
                        g[i * 3 + 1] = rFrame[i * 2];
                        g[i * 3 + 2] = rFrame[i * 2 + 1];
                    }
                    Append(segments, positions, ref row, MiniMaxH3SegmentKind.RefImage, n, g);
                    imageUpdate.AddRange(Enumerable.Repeat(false, n));
                    cursor += 1.0;
                }
                else if (blk.Kind == "audio")
                {
                    int rt = blk.RefAudioT;
                    if (rt > 0)
                    {
                        Append(segments, positions, ref row, MiniMaxH3SegmentKind.RefAudio, rt * 2,
                            AudioGrid(cursor, rt, targetWLow, targetWHigh));
                        audioUpdate.AddRange(Enumerable.Repeat(false, rt * 2));
                    }
                    cursor += rt;
                }
                else if (blk.Kind is "video" or "video_audio")
                {
                    int rt = blk.RefAudioT, vt = blk.LatentT;
                    (double[] rFrame, double[] rWGrid) =
                        FrameGrid(PadToPatch(blk.LatentH, 2), PadToPatch(blk.LatentW, 2));
                    if (rt > 0)
                    {
                        Append(segments, positions, ref row, MiniMaxH3SegmentKind.RefAudio, rt * 2,
                            AudioGrid(cursor, rt, rWGrid[0], rWGrid[^1]));
                        audioUpdate.AddRange(Enumerable.Repeat(false, rt * 2));
                    }
                    int n = vt * (rFrame.Length / 2);
                    Append(segments, positions, ref row, MiniMaxH3SegmentKind.RefImage, n, VideoGrid(vt, rFrame, cursor));
                    imageUpdate.AddRange(Enumerable.Repeat(false, n));
                    cursor += Math.Max(rt, SumVideoSpans(vt));
                }
                else
                {
                    throw new ArgumentException($"Unknown MiniMax-H3 reference block kind '{blk.Kind}'.");
                }
            }
        }

        Append(segments, positions, ref row, MiniMaxH3SegmentKind.Audio, audioT * 2,
            AudioGrid(targetCursor, audioT, targetWLow, targetWHigh));
        audioUpdate.AddRange(Enumerable.Repeat(true, audioT * 2));

        int videoRows = latentT * frameRows;
        Append(segments, positions, ref row, MiniMaxH3SegmentKind.Video, videoRows,
            VideoGrid(latentT, frame, targetCursor));
        imageUpdate.AddRange(Enumerable.Repeat(true, videoRows));

        SequenceLength = row;
        Segments = segments;
        ImageUpdate = imageUpdate.ToArray();
        AudioUpdate = audioUpdate.ToArray();
        Signature = (textLen, latentT, latentH, latentW, audioT);

        PositionIds = new double[SequenceLength * 3];
        int offset = 0;
        foreach (double[] block in positions)
        {
            Array.Copy(block, 0, PositionIds, offset, block.Length);
            offset += block.Length;
        }
    }

    private static void Append(List<MiniMaxH3Segment> segments, List<double[]> positions, ref int row,
        MiniMaxH3SegmentKind kind, int count, double[] grid)
    {
        segments.Add(new MiniMaxH3Segment(row, row + count, kind));
        positions.Add(grid);
        row += count;
    }

    /// <summary>Area-normalized axis coordinates: <c>linspace((1-r)/2, (1+r)/2, dim/patch, endpoint=False) * 32</c>.</summary>
    private static double[] AxisFromSqrtArea(int dim, int patch, double sqrtArea)
    {
        double ratio = dim / sqrtArea;
        int n = dim / patch;
        double[] axis = new double[n];
        for (int i = 0; i < n; i++)
        {
            axis[i] = (i * (ratio / n) + (1.0 - ratio) / 2.0) * 32.0;
        }
        return axis;
    }

    /// <summary>Row-major <c>[rows, 2]</c> (h, w) coordinates of one latent frame's 2x2-patch rows, plus the w axis.</summary>
    private static (double[] Frame, double[] WGrid) FrameGrid(int h, int w)
    {
        double area = Math.Sqrt((double)h * w);
        double[] hAxis = AxisFromSqrtArea(h, 2, area);
        double[] wAxis = AxisFromSqrtArea(w, 2, area);
        double[] frame = new double[hAxis.Length * wAxis.Length * 2];
        int k = 0;
        foreach (double hv in hAxis)
        {
            foreach (double wv in wAxis)
            {
                frame[k++] = hv;
                frame[k++] = wv;
            }
        }
        return (frame, wAxis);
    }

    private static double SumVideoSpans(int n)
    {
        double sum = 0;
        for (int k = 0; k < n; k++)
        {
            sum += FrameRescale * FramePerToken[k % 5];
        }
        return sum;
    }

    private static double ReferenceTimeSpan(MiniMaxH3RefBlock block) => block.Kind switch
    {
        "image" => 1.0,
        "audio" => block.RefAudioT,
        "video" or "video_audio" => Math.Max(block.RefAudioT, SumVideoSpans(block.LatentT)),
        _ => 0.0,
    };

    private static int PadToPatch(int size, int patch)
    {
        if (size <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(size), size, "MiniMax-H3 latent axes must be positive.");
        }
        return size + (patch - size % patch) % patch;
    }

    /// <summary>Origin plus the exclusive cumulative sum of per-token frame spans.</summary>
    private static double[] VideoTGrid(int n, double origin)
    {
        double[] t = new double[n];
        double acc = 0;
        for (int k = 0; k < n; k++)
        {
            t[k] = origin + acc;
            acc += FrameRescale * FramePerToken[k % 5];
        }
        return t;
    }

    /// <summary>Channel-major stereo rows: t advances per latent frame, w pins to a grid extreme per channel, h stays 0.</summary>
    private static double[] AudioGrid(double cursor, int t, double wLow, double wHigh)
    {
        double[] g = new double[t * 2 * 3];
        for (int i = 0; i < t; i++)
        {
            g[i * 3] = cursor + i;
            g[i * 3 + 2] = wLow;
            int j = t + i;
            g[j * 3] = cursor + i;
            g[j * 3 + 2] = wHigh;
        }
        return g;
    }

    private static double[] VideoGrid(int vt, double[] frame, double cursor)
    {
        int frameRows = frame.Length / 2;
        double[] tGrid = VideoTGrid(vt, cursor);
        double[] g = new double[vt * frameRows * 3];
        int k = 0;
        for (int t = 0; t < vt; t++)
        {
            for (int i = 0; i < frameRows; i++)
            {
                g[k++] = tGrid[t];
                g[k++] = frame[i * 2];
                g[k++] = frame[i * 2 + 1];
            }
        }
        return g;
    }
}
