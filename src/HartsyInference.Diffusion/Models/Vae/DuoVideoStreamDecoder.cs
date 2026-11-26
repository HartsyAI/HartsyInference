using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>One published block of the <c>duo_5_17</c> stream: its frames as RGB8, <c>[frame][row][col][channel]</c>, flattened.</summary>
public sealed record DuoDecodedBlock(int Block, int Frames, int Height, int Width, byte[] Rgb);

/// <summary>Streams normalized Duo latent chunks through a video decoder with the <c>duo_5_17</c> schedule of
/// <see cref="DuoVideoDecodePlan"/>. Each chunk is five latent steps of 24 channels. The chunks, the windows and the cropped frames stay on
/// the backend's device; the only host read is the published RGB, one block at a time. The decoder is a delegate, so tests drive the window
/// and crop logic with a fake; <see cref="ForH3"/> wires the released H3 decoder.
/// <para>The decoder returns <c>[1, 3, F, H, W]</c> pixels in [-1, 1], as <see cref="MiniMaxH3VideoVaeDecoder.Decode"/> does. Upstream's
/// ImageNet denormalize then clamp to [0, 1] is the same map as <c>(x + 1) / 2</c> clamped, so RGB is <c>round(255 · clamp((x + 1) / 2))</c>,
/// with round-half-to-even as torch does. This equivalence is derived; parity against upstream is not run.</para></summary>
public sealed class DuoVideoStreamDecoder : IDisposable
{
    private const int Channels = 24;
    private const int ChunkLatents = 5;
    private const int LatentAxis = 2;

    private readonly IBackend _backend;
    private readonly Func<Tensor, Tensor> _decodeWindow;
    private readonly List<Chunk> _recent = [];
    private int _received;
    private int _base;
    private int _height;
    private int _width;
    private bool _ended;
    private bool _disposed;

    /// <summary>Chunks held between calls: at most two. A call briefly holds the new chunk as well, and removes it if the decode throws.</summary>
    internal int RetainedChunks => _recent.Count;

    /// <param name="backend">Backend that owns the chunks and runs the window and crop copies.</param>
    /// <param name="decodeWindow">Decodes a window <c>[1, 24, T, H, W]</c>; the window is owned by this class. The returned pixels are taken
    /// and disposed here.</param>
    public DuoVideoStreamDecoder(IBackend backend, Func<Tensor, Tensor> decodeWindow)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _decodeWindow = decodeWindow ?? throw new ArgumentNullException(nameof(decodeWindow));
    }

    /// <summary>The H3 decoder, run on one window at a time. Refuses a decoder whose frame counts differ from the schedule's, at build time.</summary>
    public static DuoVideoStreamDecoder ForH3(MiniMaxH3VideoVaeDecoder decoder, IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(backend);
        if (decoder.OutputFrames(7) != 22 || decoder.OutputFrames(12) != 39)
        {
            throw new InvalidOperationException(
                $"The decoder returns {decoder.OutputFrames(7)} and {decoder.OutputFrames(12)} frames for 7 and 12 latents; the duo_5_17 schedule needs 22 and 39.");
        }
        return new DuoVideoStreamDecoder(backend, window => decoder.Decode(backend, window));
    }

    /// <summary>Receives the next chunk, <c>[1, 24, 5, H, W]</c> normalized latents. The stream keeps its own device copy. Returns the block it
    /// publishes, if any: none for the first chunk, otherwise the previous block.</summary>
    public IReadOnlyList<DuoDecodedBlock> OnChunk(Tensor chunk)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_ended)
        {
            throw new InvalidOperationException("The stream has ended; no chunk can follow End.");
        }
        Chunk z = Keep(chunk);
        int index = _received;
        _recent.Add(z);
        _received++;
        try
        {
            DuoDecodeStep? step = DuoVideoDecodePlan.OnChunk(index);
            if (step is null)
            {
                Trim();
                return [];
            }
            // Whole chunks from the window's first chunk up to the newest previous one, then the newest chunk's first two latents.
            List<Segment> segments = [];
            for (int c = step.Value.FirstChunk; c < index; c++)
            {
                segments.Add(new Segment(c, 0, ChunkLatents));
            }
            segments.Add(new Segment(index, 0, 2));
            DuoDecodedBlock block = Publish(step.Value, segments);
            Trim();
            return [block];
        }
        catch
        {
            // Undo this chunk, so a decode that throws leaves the stream exactly as it was before the call.
            Chunk undone = _recent[^1];
            _recent.RemoveAt(_recent.Count - 1);
            undone.Data.Dispose();
            _received--;
            throw;
        }
    }

    /// <summary>Ends the stream and publishes the last block from its own last two latents. Empty when no chunk arrived.</summary>
    public IReadOnlyList<DuoDecodedBlock> End()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_ended)
        {
            throw new InvalidOperationException("The stream has already ended.");
        }
        DuoDecodeStep? step = DuoVideoDecodePlan.AtEnd(_received);
        if (step is null)
        {
            _ended = true;
            return [];
        }
        int last = step.Value.LastChunk;
        List<Segment> segments = [];
        for (int c = step.Value.FirstChunk; c <= last; c++)
        {
            segments.Add(new Segment(c, 0, ChunkLatents));
        }
        segments.Add(new Segment(last, ChunkLatents - 2, 2));
        // Only a published End closes the stream: if the decode throws, End can be called again.
        DuoDecodedBlock block = Publish(step.Value, segments);
        _ended = true;
        return [block];
    }

    /// <summary>Frees every retained chunk. The stream takes no more chunks.</summary>
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        _disposed = true;
        foreach (Chunk chunk in _recent)
        {
            chunk.Data.Dispose();
        }
        _recent.Clear();
    }

    private DuoDecodedBlock Publish(DuoDecodeStep step, List<Segment> segments)
    {
        int latents = segments.Sum(s => s.Count);
        if (latents != step.WindowLatents)
        {
            throw new InvalidOperationException($"The window holds {latents} latents; the schedule needs {step.WindowLatents}.");
        }
        Tensor window = BuildWindow(segments, _height, _width);
        Tensor pixels;
        try
        {
            pixels = _decodeWindow(window);
        }
        finally
        {
            window.Dispose();
        }
        try
        {
            return Crop(step, pixels);
        }
        finally
        {
            pixels.Dispose();
        }
    }

    /// <summary>Concatenates the segments along the latent axis into a device <c>[1, 24, T, H, W]</c>: one fill of the whole output, then one
    /// strided scatter per segment, so the device never uploads a partly written window.</summary>
    private Tensor BuildWindow(List<Segment> segments, int height, int width)
    {
        int total = segments.Sum(s => s.Count);
        Tensor window = new(new TensorShape(new long[] { 1, Channels, total, height, width }), DType.F32);
        StridedCopies.FillFrom(_backend, window, At(segments[0].Chunk).Data);
        int at = 0;
        foreach (Segment segment in segments)
        {
            Tensor part = Slice(At(segment.Chunk).Data, segment.Start, segment.Count);
            try
            {
                StridedCopies.ScatterAxis(_backend, window, part, LatentAxis, at);
            }
            finally
            {
                part.Dispose();
            }
            at += segment.Count;
        }
        return window;
    }

    /// <summary>Copies the published frames of a decoded window on the device, then reads them once as RGB8.</summary>
    private DuoDecodedBlock Crop(DuoDecodeStep step, Tensor pixels)
    {
        if (pixels.Shape.Rank != 5 || pixels.Shape[0] != 1 || pixels.Shape[1] != 3)
        {
            throw new InvalidOperationException($"The decoder returned {pixels.Shape}; expected [1, 3, F, H, W].");
        }
        int frames = (int)pixels.Shape[2], height = (int)pixels.Shape[3], width = (int)pixels.Shape[4];
        if (frames != step.RawFrames)
        {
            throw new InvalidOperationException($"The decoder returned {frames} frames; the schedule needs {step.RawFrames}.");
        }
        if (step.CropEnd > frames || step.CropStart < 0 || step.CropEnd <= step.CropStart)
        {
            throw new InvalidOperationException($"The crop [{step.CropStart}, {step.CropEnd}) does not fit {frames} frames.");
        }
        int published = step.CropEnd - step.CropStart;
        int plane = height * width;
        Tensor cropped = Slice(pixels, step.CropStart, published);
        try
        {
            ReadOnlySpan<float> source = cropped.AsSpan<float>();
            byte[] rgb = new byte[published * plane * 3];
            for (int f = 0; f < published; f++)
            {
                for (int p = 0; p < plane; p++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        float value = source[((c * published) + f) * plane + p];
                        rgb[((f * plane) + p) * 3 + c] = ToByte(value);
                    }
                }
            }
            return new DuoDecodedBlock(step.Block, published, height, width, rgb);
        }
        finally
        {
            cropped.Dispose();
        }
    }

    /// <summary>The RGB byte of a pixel in [-1, 1]: <c>round(255 · clamp((x + 1) / 2))</c>, round-half-to-even.</summary>
    public static byte ToByte(float x)
    {
        float unit = Math.Clamp((x + 1f) * 0.5f, 0f, 1f);
        return (byte)Math.Round(unit * 255f, MidpointRounding.ToEven);
    }

    /// <summary>Validates a chunk and returns the stream's own device copy of it.</summary>
    private Chunk Keep(Tensor chunk)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        if (chunk.DType != DType.F32)
        {
            throw new ArgumentException($"A chunk must be F32; got {chunk.DType}.", nameof(chunk));
        }
        if (chunk.Shape.Rank != 5 || chunk.Shape[0] != 1 || chunk.Shape[1] != Channels || chunk.Shape[2] != ChunkLatents)
        {
            throw new ArgumentException($"A chunk is [1, {Channels}, {ChunkLatents}, H, W]; got {chunk.Shape}.", nameof(chunk));
        }
        int height = (int)chunk.Shape[3], width = (int)chunk.Shape[4];
        if (_received == 0)
        {
            _height = height;
            _width = width;
        }
        else if (height != _height || width != _width)
        {
            throw new ArgumentException($"Every chunk must be {_height}x{_width} latent pixels, as the first; got {height}x{width}.", nameof(chunk));
        }
        return new Chunk(Slice(chunk, 0, ChunkLatents), height, width);
    }

    /// <summary>Latent steps <c>[start, start + count)</c> of a chunk or window, as a new device tensor the caller owns.</summary>
    private Tensor Slice(Tensor source, int start, int count)
    {
        Tensor part = new(WithLatents(source.Shape, count), DType.F32);
        try
        {
            StridedCopies.SliceAxis(_backend, part, source, LatentAxis, start, count);
        }
        catch
        {
            part.Dispose();
            throw;
        }
        return part;
    }

    private static TensorShape WithLatents(TensorShape shape, int count)
    {
        long[] dims = new long[shape.Rank];
        shape.CopyDimsTo(dims);
        dims[LatentAxis] = count;
        return new TensorShape(dims);
    }

    /// <summary>The retained chunk with absolute index <paramref name="chunk"/>.</summary>
    private Chunk At(int chunk) => _recent[chunk - _base];

    /// <summary>Drops every chunk but the newest two. The next window reads from two chunks back, and nothing older is ever needed again.</summary>
    private void Trim()
    {
        while (_recent.Count > 2)
        {
            _recent[0].Data.Dispose();
            _recent.RemoveAt(0);
            _base++;
        }
    }

    private readonly record struct Segment(int Chunk, int Start, int Count);

    /// <summary>A received chunk on the device: latents laid out <c>[1][24][5][H][W]</c>, owned by the stream.</summary>
    private sealed record Chunk(Tensor Data, int Height, int Width);
}
