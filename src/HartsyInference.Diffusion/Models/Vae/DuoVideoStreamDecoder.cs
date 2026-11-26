using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Duo;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>One published block of the <c>duo_5_17</c> stream: its frames as RGB8, <c>[frame][row][col][channel]</c>, flattened.</summary>
public sealed record DuoDecodedBlock(int Block, int Frames, int Height, int Width, byte[] Rgb);

/// <summary>Streams normalized Duo latent chunks through a video decoder with the <c>duo_5_17</c> schedule of
/// <see cref="DuoVideoDecodePlan"/>. Each chunk is five latent steps of 24 channels. The decoder is a delegate, so tests drive the
/// window and crop logic with a fake; <see cref="ForH3"/> wires the released H3 decoder.
/// <para>The decoder returns <c>[1, 3, F, H, W]</c> pixels in [-1, 1], as <see cref="MiniMaxH3VideoVaeDecoder.Decode"/> does. Upstream's
/// ImageNet denormalize then clamp to [0, 1] is the same map as <c>(x + 1) / 2</c> clamped, so RGB is <c>round(255 · clamp((x + 1) / 2))</c>,
/// with round-half-to-even as torch does. This equivalence is derived; parity against upstream is not run.</para></summary>
public sealed class DuoVideoStreamDecoder
{
    private const int Channels = 24;
    private const int ChunkLatents = 5;

    private readonly Func<Tensor, Tensor> _decodeWindow;
    private readonly List<Chunk> _recent = [];
    private int _received;
    private int _base;
    private int _height;
    private int _width;
    private bool _ended;

    /// <summary>Chunks still held. The schedule reads at most the last two, so this stays at two or three.</summary>
    internal int RetainedChunks => _recent.Count;

    public DuoVideoStreamDecoder(Func<Tensor, Tensor> decodeWindow)
    {
        _decodeWindow = decodeWindow ?? throw new ArgumentNullException(nameof(decodeWindow));
    }

    /// <summary>The H3 decoder, run on one window at a time.</summary>
    public static DuoVideoStreamDecoder ForH3(MiniMaxH3VideoVaeDecoder decoder, IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(decoder);
        ArgumentNullException.ThrowIfNull(backend);
        return new DuoVideoStreamDecoder(window => decoder.Decode(backend, window));
    }

    /// <summary>Receives the next chunk, <c>[1, 24, 5, H, W]</c> normalized latents. Returns the block it publishes, if any: none for
    /// the first chunk, otherwise the previous block.</summary>
    public IReadOnlyList<DuoDecodedBlock> OnChunk(Tensor chunk)
    {
        if (_ended)
        {
            throw new InvalidOperationException("The stream has ended; no chunk can follow End.");
        }
        Chunk z = Read(chunk);
        int index = _received;
        bool first = index == 0;
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
            _recent.RemoveAt(_recent.Count - 1);
            _received--;
            if (first)
            {
                _height = 0;
                _width = 0;
            }
            throw;
        }
    }

    /// <summary>Ends the stream and publishes the last block from its own last two latents. Empty when no chunk arrived.</summary>
    public IReadOnlyList<DuoDecodedBlock> End()
    {
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

    private DuoDecodedBlock Publish(DuoDecodeStep step, List<Segment> segments)
    {
        int latents = segments.Sum(s => s.Count);
        if (latents != step.WindowLatents)
        {
            throw new InvalidOperationException($"The window holds {latents} latents; the schedule needs {step.WindowLatents}.");
        }
        using Tensor window = BuildWindow(segments, _height, _width);
        using Tensor pixels = _decodeWindow(window);
        return Crop(step, pixels);
    }

    /// <summary>Concatenates the segments along the latent axis into <c>[1, 24, T, H, W]</c>.</summary>
    private Tensor BuildWindow(List<Segment> segments, int height, int width)
    {
        int total = segments.Sum(s => s.Count);
        int plane = height * width;
        Tensor window = new(new TensorShape(new long[] { 1, Channels, total, height, width }), DType.F32);
        Span<float> destination = window.AsSpan<float>();
        for (int c = 0; c < Channels; c++)
        {
            int t = 0;
            foreach (Segment segment in segments)
            {
                ReadOnlySpan<float> source = At(segment.Chunk).Data;
                for (int k = 0; k < segment.Count; k++, t++)
                {
                    source.Slice(((c * ChunkLatents) + segment.Start + k) * plane, plane)
                        .CopyTo(destination.Slice(((c * total) + t) * plane, plane));
                }
            }
        }
        return window;
    }

    private static DuoDecodedBlock Crop(DuoDecodeStep step, Tensor pixels)
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
        ReadOnlySpan<float> source = pixels.AsSpan<float>();
        byte[] rgb = new byte[published * plane * 3];
        for (int f = 0; f < published; f++)
        {
            int sourceFrame = step.CropStart + f;
            for (int p = 0; p < plane; p++)
            {
                for (int c = 0; c < 3; c++)
                {
                    float value = source[((c * frames) + sourceFrame) * plane + p];
                    rgb[((f * plane) + p) * 3 + c] = ToByte(value);
                }
            }
        }
        return new DuoDecodedBlock(step.Block, published, height, width, rgb);
    }

    /// <summary>The RGB byte of a pixel in [-1, 1]: <c>round(255 · clamp((x + 1) / 2))</c>, round-half-to-even.</summary>
    public static byte ToByte(float x)
    {
        float unit = Math.Clamp((x + 1f) * 0.5f, 0f, 1f);
        return (byte)Math.Round(unit * 255f, MidpointRounding.ToEven);
    }

    private Chunk Read(Tensor chunk)
    {
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
        return new Chunk(chunk.AsSpan<float>().ToArray(), height, width);
    }

    /// <summary>The retained chunk with absolute index <paramref name="chunk"/>.</summary>
    private Chunk At(int chunk) => _recent[chunk - _base];

    /// <summary>Drops every chunk but the newest two. The next window reads from two chunks back, and nothing older is ever needed again.</summary>
    private void Trim()
    {
        while (_recent.Count > 2)
        {
            _recent.RemoveAt(0);
            _base++;
        }
    }

    private readonly record struct Segment(int Chunk, int Start, int Count);

    /// <summary>A received chunk: latents laid out <c>[24][5][H][W]</c>.</summary>
    private sealed record Chunk(float[] Data, int Height, int Width);
}
