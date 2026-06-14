using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Kokoro;

/// <summary>A Kokoro "voice pack" is a length-keyed style table: a raw float32 binary
/// blob shaped as <c>[N, 1, 256]</c> where each row is a 256-dim style vector tuned for
/// a specific phoneme-sequence length. At inference time the pipeline selects
/// <c>style[len(tokens) - 1]</c> (clamped to <c>[0, N-1]</c>) and slices it into
/// <c>s_dec = style[:128]</c> + <c>s_pred = style[128:]</c>.
///
/// <para>Voice packs ship one-per-voice (<c>af_heart.bin</c>, <c>bm_lewis.bin</c>, etc.)
/// alongside the model weights under <c>hexgrad/Kokoro-82M/voices/</c>. They are tiny
/// (~510 KB) and load directly from disk — no decoding needed.</para>
///
/// <para>The exact bucket count varies by checkpoint: the original 1.0 release shipped
/// 511 rows; later voice packs ship 509 or 510. We infer <c>N</c> from the file size,
/// require it to divide evenly by <c>4 * 256</c>, and clamp out-of-range queries.</para></summary>
public sealed class KokoroVoicePack : IDisposable
{
    /// <summary>Width of each style row. The first <see cref="KokoroConfig.StyleDim"/>
    /// dims condition the decoder; the trailing dims condition the predictor.</summary>
    public const int StyleWidth = 256;

    private const int BytesPerFloat = 4;
    private const long BytesPerRow = BytesPerFloat * StyleWidth;

    private readonly Tensor _table;     // [N, StyleWidth] flat
    private int _disposed;

    /// <summary>Number of length buckets stored.</summary>
    public int NumBuckets { get; }

    /// <summary>Pack name (file basename without extension), e.g. <c>"af_heart"</c>.</summary>
    public string Name { get; }

    private KokoroVoicePack(string name, Tensor table, int n)
    {
        Name = name;
        _table = table;
        NumBuckets = n;
    }

    /// <summary>Loads a voice pack from a raw float32 binary file. The file's element
    /// count must be a multiple of 256.</summary>
    public static KokoroVoicePack LoadFromFile(string path)
    {
        long size = new FileInfo(path).Length;
        if (size % BytesPerRow != 0)
            throw new InvalidDataException(
                $"Voice pack '{path}' size {size} is not a multiple of {BytesPerRow} (one row of {StyleWidth} f32 floats). " +
                $"This file is likely the wrong format.");

        int n = (int)(size / BytesPerRow);
        Tensor table = new(new TensorShape(n, StyleWidth), DType.F32);
        using (FileStream fs = File.OpenRead(path))
        {
            unsafe
            {
                byte* dst = (byte*)table.DataPointer;
                Span<byte> dstSpan = new(dst, (int)size);
                int read = 0;
                while (read < size)
                {
                    int r = fs.Read(dstSpan[read..]);
                    if (r <= 0) throw new EndOfStreamException($"Voice pack '{path}' truncated: read {read} of {size}.");
                    read += r;
                }
            }
        }
        return new KokoroVoicePack(Path.GetFileNameWithoutExtension(path), table, n);
    }

    /// <summary>Returns a fresh <c>[1, 256]</c> tensor containing the style row for a
    /// sequence of <paramref name="tokenCount"/> phoneme tokens (including the leading
    /// + trailing BOS/EOS pads). The index is clamped into <c>[0, NumBuckets-1]</c>.
    ///
    /// <para>The returned tensor is independent of the underlying pack — disposing it
    /// is required and does not affect future calls.</para></summary>
    public Tensor GetStyle(int tokenCount)
    {
        ThrowIfDisposed();
        int idx = Math.Clamp(tokenCount - 1, 0, NumBuckets - 1);
        Tensor row = new(new TensorShape(1, StyleWidth), DType.F32);
        unsafe
        {
            float* src = (float*)_table.DataPointer + (long)idx * StyleWidth;
            float* dst = (float*)row.DataPointer;
            for (int i = 0; i < StyleWidth; i++) dst[i] = src[i];
        }
        return row;
    }

    /// <summary>Splits a <c>[1, 256]</c> style row into the decoder half <c>[1, 128]</c>
    /// (first 128 dims) and the predictor half <c>[1, 128]</c> (trailing 128 dims).
    /// Both returned tensors are independent of the input — the caller owns disposal.</summary>
    public static (Tensor Decoder, Tensor Predictor) SplitStyle(Tensor style, int styleDim = 128)
    {
        if (style.Shape.Rank != 2 || (int)style.Shape[1] != 2 * styleDim)
            throw new ArgumentException($"SplitStyle expects [B, {2 * styleDim}], got {style.Shape}.");

        int batch = (int)style.Shape[0];
        Tensor dec = new(new TensorShape(batch, styleDim), DType.F32);
        Tensor pred = new(new TensorShape(batch, styleDim), DType.F32);
        unsafe
        {
            float* src = (float*)style.DataPointer;
            float* d = (float*)dec.DataPointer;
            float* p = (float*)pred.DataPointer;
            for (int b = 0; b < batch; b++)
            {
                int row = b * 2 * styleDim;
                for (int j = 0; j < styleDim; j++) d[b * styleDim + j] = src[row + j];
                for (int j = 0; j < styleDim; j++) p[b * styleDim + j] = src[row + styleDim + j];
            }
        }
        return (dec, pred);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed != 0) throw new ObjectDisposedException(nameof(KokoroVoicePack));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _table.Dispose();
    }
}
