using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Checkpoints;

namespace HartsyInference.ModelAssets.Nf4;

/// <summary>Dequantizer for bitsandbytes <b>NF4</b> (4-bit NormalFloat) weights, as shipped by the <c>bitsandbytes</c> <c>Linear4bit</c> path (Flux.2 Klein nf4, Ideogram 4 nf4, many community quants).
///
/// <para>NF4 is a <b>non-linear</b> 4-bit code: each nibble indexes a fixed 16-entry codebook of normal quantiles (not the linear e2m1 grid used by MXFP4/NVFP4). Reconstruction is <c>x[i] = absmax[i / blockSize] · NF4_LUT[nibble]</c>, where <c>absmax</c> is one positive scale per block of <see cref="DefaultBlockSize"/> consecutive elements. Because the codebook is non-linear, NF4 has <b>no tensor-core dtype</b> — unlike FP4 it cannot feed a native FP4 GEMM, so the engine dequantizes it to F32 at load (the same strategy MXFP4/NVFP4 use). The native FP4 path (<c>BlockScaledGemmExecutor</c>) is e2m1 only.</para>
///
/// <para>Two layers exist on disk: plain quant (an explicit F32 <c>absmax</c>) and bitsandbytes "double quant" where <c>absmax</c> is itself 8-bit quantized against a 256-entry nested codebook plus a scalar offset. <see cref="ReconstructDoubleQuantAbsmax"/> rebuilds the F32 absmax for that case.</para></summary>
public static unsafe class Nf4Codec
{
    /// <summary>Default bitsandbytes NF4 block size (elements per absmax scale).</summary>
    public const int DefaultBlockSize = 64;

    /// <summary>The 16-entry NF4 codebook (bitsandbytes normal-float quantiles, symmetric, includes exact 0).</summary>
    public static readonly float[] Nf4Lut =
    [
        -1.0f,
        -0.6961928009986877f,
        -0.5250730514526367f,
        -0.39491748809814453f,
        -0.28444138169288635f,
        -0.18477343022823334f,
        -0.09105003625154495f,
         0.0f,
         0.07958029955625534f,
         0.16093020141124725f,
         0.24611230194568634f,
         0.33791524171829224f,
         0.44070982933044434f,
         0.5626170039176941f,
         0.7229568362236023f,
         1.0f,
    ];

    /// <summary>Dequantizes a plain (single-quant) NF4 tensor to F32. The packed input holds two nibbles per byte, first element in the high nibble (bitsandbytes order).</summary>
    /// <param name="packed">U8 tensor of <c>ceil(N/2)</c> bytes holding the 4-bit codes.</param>
    /// <param name="absmaxF32">F32 per-block scales, length <c>ceil(N / blockSize)</c>.</param>
    /// <param name="outputShape">Shape of the reconstructed tensor; its element count is <c>N</c>.</param>
    /// <param name="blockSize">Elements per absmax block (default <see cref="DefaultBlockSize"/>).</param>
    public static Tensor Dequantize(Tensor packed, Tensor absmaxF32, TensorShape outputShape, int blockSize = DefaultBlockSize)
    {
        if (packed.DType != DType.U8)
            throw new ArgumentException($"NF4 packed weight must be U8; got {packed.DType}.", nameof(packed));
        if (absmaxF32.DType != DType.F32)
            throw new ArgumentException($"NF4 absmax must be F32; got {absmaxF32.DType}. Use ReconstructDoubleQuantAbsmax for double-quant.", nameof(absmaxF32));
        if (blockSize <= 0)
            throw new ArgumentException("blockSize must be positive.", nameof(blockSize));

        long n = outputShape.ElementCount;
        long expectedBytes = (n + 1) / 2;
        if (packed.Shape.ElementCount < expectedBytes)
            throw new ArgumentException($"NF4 packed tensor has {packed.Shape.ElementCount} bytes; need at least {expectedBytes} for {n} elements.", nameof(packed));

        long expectedBlocks = (n + blockSize - 1) / blockSize;
        if (absmaxF32.Shape.ElementCount < expectedBlocks)
            throw new ArgumentException($"NF4 absmax has {absmaxF32.Shape.ElementCount} blocks; need {expectedBlocks} for {n} elements at blockSize {blockSize}.", nameof(absmaxF32));

        Tensor output = new Tensor(outputShape, DType.F32);
        byte* p = (byte*)packed.DataPointer;
        float* a = (float*)absmaxF32.DataPointer;
        float* o = (float*)output.DataPointer;

        for (long i = 0; i < n; i++)
        {
            byte b = p[i >> 1];
            int nibble = (i & 1) == 0 ? (b >> 4) & 0x0F : b & 0x0F; // high nibble = even element
            o[i] = Nf4Lut[nibble] * a[i / blockSize];
        }

        return output;
    }

    /// <summary>Quantizes F32 weights to plain NF4: one absmax per block, each element rounded to its nearest codebook entry.</summary>
    /// <returns>The packed nibbles (high nibble = even element) and one F32 absmax per block. Dequantize reproduces each element within <see cref="MaxBlockError"/> of its block absmax.</returns>
    /// <exception cref="ArgumentException">The block size is not positive, or a value is NaN or infinite.</exception>
    public static (byte[] Packed, float[] Absmax) Quantize(ReadOnlySpan<float> values, int blockSize = DefaultBlockSize)
    {
        if (blockSize <= 0)
            throw new ArgumentException("blockSize must be positive.", nameof(blockSize));

        int n = values.Length;
        float[] absmax = new float[((long)n + blockSize - 1) / blockSize];
        for (int b = 0; b < absmax.Length; b++)
        {
            int start = b * blockSize;
            int end = Math.Min(n, start + blockSize);
            float m = 0f;
            for (int i = start; i < end; i++)
            {
                if (!float.IsFinite(values[i]))
                    throw new ArgumentException("NF4 input must be finite; got NaN or infinity.", nameof(values));
                m = MathF.Max(m, MathF.Abs(values[i]));
            }
            absmax[b] = m;
        }

        byte[] codes = new byte[n];
        fixed (float* src = values)
        fixed (byte* dst = codes)
        {
            nint srcAddr = (nint)src, dstAddr = (nint)dst;
            CpuParallel.For(absmax.Length, n, b =>
            {
                float inv = absmax[b] > 0f ? 1f / absmax[b] : 0f;
                int start = b * blockSize;
                int end = Math.Min(n, start + blockSize);
                float* s = (float*)srcAddr;
                byte* c = (byte*)dstAddr;
                for (int i = start; i < end; i++)
                    c[i] = (byte)NearestCode(s[i] * inv);
            });
        }

        byte[] packed = new byte[(n + 1) / 2];
        for (int i = 0; i < n; i++)
            packed[i >> 1] |= (byte)((i & 1) == 0 ? codes[i] << 4 : codes[i]);

        return (packed, absmax);
    }

    /// <summary>Largest reconstruction error for one element, as a fraction of its block absmax: half the widest gap between adjacent codebook entries.</summary>
    public static float MaxBlockError
    {
        get
        {
            float widest = 0f;
            for (int i = 1; i < Nf4Lut.Length; i++)
                widest = MathF.Max(widest, Nf4Lut[i] - Nf4Lut[i - 1]);
            return widest / 2f;
        }
    }

    private static int NearestCode(float normalized)
    {
        int best = 0;
        float bestError = float.MaxValue;
        for (int c = 0; c < Nf4Lut.Length; c++)
        {
            float error = MathF.Abs(normalized - Nf4Lut[c]);
            if (error < bestError)
            {
                bestError = error;
                best = c;
            }
        }
        return best;
    }

    #region Companion tensors

    /// <summary>Quantizes a weight into the bitsandbytes companion tensors <see cref="Nf4CompanionFold"/> reads back: the
    /// packed U8 weight shaped <c>[n/2, 1]</c> as bitsandbytes stores it, F32 <c>.absmax</c>, the 16-entry F32 <c>.quant_map</c>,
    /// and the U8 JSON <c>.quant_state.bitsandbytes__nf4</c> blob that gives the shape, block size and compute dtype.</summary>
    /// <param name="weightKey">Key of the weight; the companions are named from it.</param>
    /// <param name="computeDtype">Torch dtype name the weight dequantizes back to: <c>float32</c>, <c>float16</c> or <c>bfloat16</c>.</param>
    /// <returns>Every tensor, weight first. The caller owns and disposes them.</returns>
    public static List<KeyValuePair<string, Tensor>> QuantizeToCompanions(string weightKey, ReadOnlySpan<float> values, long rows, long cols,
        string computeDtype, int blockSize = DefaultBlockSize)
    {
        (byte[] packed, float[] absmax) = Quantize(values, blockSize);

        Tensor weight = new(new TensorShape(packed.Length, 1), DType.U8);
        packed.AsSpan().CopyTo(weight.AsSpan<byte>());
        Tensor scales = new(new TensorShape(absmax.Length), DType.F32);
        absmax.AsSpan().CopyTo(scales.AsSpan<float>());
        Tensor codebook = new(new TensorShape(Nf4Lut.Length), DType.F32);
        Nf4Lut.AsSpan().CopyTo(codebook.AsSpan<float>());
        byte[] state = System.Text.Encoding.ASCII.GetBytes(StateText(rows, cols, blockSize, computeDtype));
        Tensor stateBlob = new(new TensorShape(state.Length), DType.U8);
        state.AsSpan().CopyTo(stateBlob.AsSpan<byte>());

        return
        [
            new(weightKey, weight),
            new(weightKey + ".absmax", scales),
            new(weightKey + ".quant_map", codebook),
            new(weightKey + ".quant_state.bitsandbytes__nf4", stateBlob),
        ];
    }

    #endregion

    #region Streaming FP8 source to NF4 file

    /// <summary>Writes a block-FP8 checkpoint as one NF4 safetensors file, one weight at a time: each FP8 weight is dequantized,
    /// quantized and written before the next is read, so memory holds one weight rather than the checkpoint. Plain tensors of
    /// any dtype are copied byte for byte. The F32 <c>weight_scale_inv</c> grids are consumed by the dequantization and not written.</summary>
    /// <remarks>A failed or cancelled write removes its <c>.partial</c> file and leaves any file already at
    /// <paramref name="outputPath"/> untouched. Cancellation is checked between tensors.</remarks>
    /// <returns>Lowercase hex SHA-256 of the file.</returns>
    /// <exception cref="ArgumentException"><paramref name="outputPath"/> or its <c>.partial</c> file is one of the source's files.</exception>
    public static string WriteFromFp8(HfQuantCheckpoint source, string outputPath, IReadOnlyDictionary<string, string> metadata,
        Action<string>? progress = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);
        ArgumentNullException.ThrowIfNull(metadata);
        // Stream into a sibling .partial file: the final path only ever holds a complete, length-checked checkpoint.
        string partial = outputPath + ".partial";
        if (source.IsSourceFile(outputPath) || source.IsSourceFile(partial))
            throw new ArgumentException($"The NF4 output '{outputPath}' must not overwrite its source checkpoint.", nameof(outputPath));
        List<HfQuantWeight> plain = [];
        List<HfQuantWeight> quantized = [];
        foreach (HfQuantWeight w in source.Weights.Values.OrderBy(static w => w.Name, StringComparer.Ordinal))
        {
            if (w.Storage == HfWeightStorage.Plain && w.StackedSource is null) plain.Add(w);
            else if (w.Storage == HfWeightStorage.Fp8Block && w.Shape.Rank == 2) quantized.Add(w);
            else throw new NotSupportedException($"'{w.Name}' is stored as {w.Storage} {w.Shape}; only FP8 2D weights and plain tensors are written.");
        }

        JsonObject header = new();
        if (metadata.Count > 0)
        {
            JsonObject meta = new();
            foreach (KeyValuePair<string, string> entry in metadata) meta[entry.Key] = entry.Value;
            header["__metadata__"] = meta;
        }
        long offset = 0;
        void Declare(string name, string dtype, long[] shape, long bytes)
        {
            JsonArray dims = new();
            foreach (long d in shape) dims.Add(d);
            JsonArray range = new() { offset, offset + bytes };
            header[name] = new JsonObject
            {
                ["dtype"] = dtype,
                ["shape"] = dims,
                ["data_offsets"] = range,
            };
            offset += bytes;
        }
        foreach (HfQuantWeight w in plain)
        {
            Declare(w.Name, w.StoredDType.Name, DimsOf(w.Shape), Tensor.ComputeByteSize(w.Shape, w.StoredDType));
        }
        foreach (HfQuantWeight w in quantized)
        {
            long rows = w.Shape[0], cols = w.Shape[1], elements = rows * cols;
            long blocks = (elements + DefaultBlockSize - 1) / DefaultBlockSize, packedBytes = (elements + 1) / 2;
            // bitsandbytes' own [n/2, 1] layout, so its loaders read the file as well as ours.
            Declare(w.Name, "U8", [packedBytes, 1], packedBytes);
            Declare(w.Name + ".absmax", "F32", [blocks], blocks * sizeof(float));
            Declare(w.Name + ".quant_map", "F32", [Nf4Lut.Length], Nf4Lut.Length * sizeof(float));
            long stateBytes = Encoding.ASCII.GetByteCount(StateText(rows, cols, DefaultBlockSize, "float32"));
            Declare(w.Name + Nf4CompanionFold.QuantStatePrefix + "nf4", "U8", [stateBytes], stateBytes);
        }

        byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
        byte[] headerBytes = new byte[json.Length + (8 - json.Length % 8) % 8];
        json.CopyTo(headerBytes, 0);
        Array.Fill(headerBytes, (byte)' ', json.Length, headerBytes.Length - json.Length);

        // Removed only if this call opened it: after a failed open the file may be another run's, even if empty.
        bool opened = false;
        try
        {
            string sha;
            using (FileStream file = new(partial, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 20))
            {
                opened = true;
                using IncrementalHash hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
                void Emit(ReadOnlySpan<byte> bytes)
                {
                    file.Write(bytes);
                    hash.AppendData(bytes);
                }

                Span<byte> length = stackalloc byte[sizeof(long)];
                System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(length, headerBytes.Length);
                Emit(length);
                Emit(headerBytes);

                foreach (HfQuantWeight w in plain)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    using Tensor view = source.Borrow(w.Name);
                    ReadOnlySpan<byte> raw = view.AsReadOnlySpan<byte>();
                    if (raw.Length != Tensor.ComputeByteSize(w.Shape, w.StoredDType))
                        throw new InvalidDataException($"'{w.Name}' holds {raw.Length} bytes; its shape {w.Shape} needs {Tensor.ComputeByteSize(w.Shape, w.StoredDType)}.");
                    Emit(raw);
                }

                int done = 0;
                foreach (HfQuantWeight w in quantized)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    long rows = w.Shape[0], cols = w.Shape[1];
                    using Tensor f32 = source.DequantizeF32(w.Name);
                    (byte[] packed, float[] absmax) = Quantize(f32.AsReadOnlySpan<float>());
                    Emit(packed);
                    Emit(MemoryMarshal.AsBytes(absmax.AsSpan()));
                    Emit(MemoryMarshal.AsBytes(Nf4Lut.AsSpan()));
                    Emit(Encoding.ASCII.GetBytes(StateText(rows, cols, DefaultBlockSize, "float32")));
                    if (++done % 500 == 0) progress?.Invoke($"{done}/{quantized.Count} FP8 weights written");
                }
                file.Flush();
                long expected = 8 + headerBytes.Length + offset;
                if (file.Length != expected)
                    throw new InvalidDataException($"Wrote {file.Length} bytes to '{partial}'; the header declares {expected}.");
                sha = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
            }

            File.Move(partial, outputPath, overwrite: true);
            return sha;
        }
        catch
        {
            // A failed cleanup is logged rather than thrown, so the error that stopped the write is the one the caller sees.
            try
            {
                if (opened) File.Delete(partial);
            }
            catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException)
            {
                Logs.Warning($"Could not remove '{partial}' after a failed write: {cleanup.Message}");
            }
            throw;
        }
    }

    /// <summary>The quant-state JSON bitsandbytes writes beside an NF4 weight: its type, logical shape, block size and compute
    /// dtype. bitsandbytes' own loader requires the type.</summary>
    private static string StateText(long rows, long cols, int blockSize, string computeDtype) =>
        $"{{\"quant_type\":\"nf4\",\"shape\":[{rows},{cols}],\"blocksize\":{blockSize},\"dtype\":\"{computeDtype}\"}}";

    private static long[] DimsOf(TensorShape shape) => Enumerable.Range(0, shape.Rank).Select(i => shape[i]).ToArray();

    #endregion

    /// <summary>Rebuilds the F32 per-block <c>absmax</c> for a bitsandbytes double-quantized NF4 tensor: <c>absmax[b] = nestedCodebook[ qAbsmax[b] ] · nestedAbsmax[b / nestedBlockSize] + offset</c>. Feed the result to <see cref="Dequantize"/>.</summary>
    /// <param name="qAbsmax">U8 quantized absmax codes, one per primary block.</param>
    /// <param name="nestedAbsmaxF32">F32 scales for the nested blocks.</param>
    /// <param name="nestedCodebookF32">256-entry F32 nested codebook (bitsandbytes <c>nested_quant_map</c>).</param>
    /// <param name="offset">Scalar offset (bitsandbytes <c>offset</c>).</param>
    /// <param name="nestedBlockSize">Primary blocks per nested scale (bitsandbytes default 256).</param>
    public static Tensor ReconstructDoubleQuantAbsmax(
        Tensor qAbsmax, Tensor nestedAbsmaxF32, Tensor nestedCodebookF32, float offset, int nestedBlockSize = 256)
    {
        if (qAbsmax.DType != DType.U8)
            throw new ArgumentException($"NF4 quantized absmax must be U8; got {qAbsmax.DType}.", nameof(qAbsmax));
        if (nestedAbsmaxF32.DType != DType.F32 || nestedCodebookF32.DType != DType.F32)
            throw new ArgumentException("NF4 nested absmax and codebook must be F32.");
        if (nestedCodebookF32.Shape.ElementCount < 256)
            throw new ArgumentException($"NF4 nested codebook must have 256 entries; got {nestedCodebookF32.Shape.ElementCount}.", nameof(nestedCodebookF32));

        long blocks = qAbsmax.Shape.ElementCount;
        Tensor absmax = new Tensor(new TensorShape(blocks), DType.F32);
        byte* q = (byte*)qAbsmax.DataPointer;
        float* nested = (float*)nestedAbsmaxF32.DataPointer;
        float* code = (float*)nestedCodebookF32.DataPointer;
        float* o = (float*)absmax.DataPointer;

        for (long b = 0; b < blocks; b++)
        {
            o[b] = code[q[b]] * nested[b / nestedBlockSize] + offset;
        }

        return absmax;
    }
}
