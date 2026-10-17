using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.LLM.Transformer;

/// <summary>Shared projection, dtype and normalization helpers for the LLM transformer family; <see cref="GenericTransformer"/> delegates here so other decoders reuse the exact same dispatch.</summary>
internal static class ProjectionOps
{
    /// <summary>Projection dispatch: float weights always take cuBLAS <see cref="IBackend.Linear"/>; quantized weights take the low-VRAM <see cref="IBackend.QuantizedMatMul"/> when <paramref name="lowVram"/> is set (weight stays compressed, transient dequant), else the faster <see cref="IBackend.Linear"/> path (dequants + caches an F16 weight).</summary>
    public static void Project(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool lowVram)
    {
        if (weight.DType.IsQuantized && lowVram) backend.QuantizedMatMul(output, input, weight, bias);
        else backend.Linear(output, input, weight, bias);
    }

    /// <summary><see cref="Project"/> with <see cref="IBackend.HighPrecisionGemm"/> forced on for this call, so F32 operands skip TF32; the previous setting is restored even when the projection throws.</summary>
    public static void ProjectPrecise(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool lowVram)
    {
        bool previous = backend.HighPrecisionGemm;
        backend.HighPrecisionGemm = true;
        try
        {
            Project(backend, output, input, weight, bias, lowVram);
        }
        finally
        {
            backend.HighPrecisionGemm = previous;
        }
    }

    /// <summary>F32 view-or-copy: dequantizes quantized tensors, casts 16-bit floats, returns the SAME reference when already F32 (callers check <c>ReferenceEquals</c> before disposing).</summary>
    /// <remarks>A ComfyUI <c>int8_tensorwise</c> weight is plain I8 whose scale — and, under ConvRot, whose basis —
    /// lives on <see cref="Tensor.QuantInfo"/>, so it is not <c>IsQuantized</c> and a cast would read a different
    /// weight rather than refuse. The embedding table of an int8 checkpoint reaches here, and it is host-gathered.</remarks>
    public static Tensor EnsureF32(Tensor t)
    {
        if (t.DType == DType.F32) return t;
        if (t.DType.IsQuantized) return HartsyInference.ModelAssets.Gguf.GgufDequantizer.Dequantize(t, DType.F32);
        if (t.DType == DType.I8 && t.QuantInfo is { RowScale: not null } int8)
        {
            using Tensor bf16 = Int8ConvRotCodec.DequantToBf16(t, int8.RowScale, int8.ConvRotGroupSize);
            return bf16.CastTo(DType.F32);
        }
        return t.CastTo(DType.F32);
    }

    /// <summary>Normalizes <paramref name="input"/> with <paramref name="weight"/> using LayerNorm (mean-centered, Cohere) or RMSNorm (everything else).</summary>
    public static void Normalize(IBackend backend, Tensor output, Tensor input, Tensor weight, Tensor? bias, bool layerNorm, float eps)
    {
        if (layerNorm) backend.LayerNorm(output, input, weight, bias!, eps);
        else backend.RmsNorm(output, input, weight, eps);
    }
    /// <summary>True when every part can be byte-concatenated without losing a quantization companion.</summary>
    /// <remarks>A ComfyUI <c>int8_tensorwise</c> weight keeps its dequant scale on <see cref="Tensor.QuantInfo"/>,
    /// indexed by exactly the rows a concatenation renumbers, so the fused copy would be raw int8 with no scale.
    /// Fusion is a decode-time dispatch saving; the separate projections are already the packed-weight fast path.</remarks>
    public static bool Fusable(params Tensor?[] parts)
    {
        foreach (Tensor? part in parts)
        {
            if (part?.QuantInfo is not null) return false;
        }
        return true;
    }

    /// <summary>Concatenates weight (<c>[N,K]</c>) or bias (<c>[N]</c>) tensors along dim 0 (output rows) via a plain byte-level copy — correct for any dtype, including block-quantized formats, since a GGUF/our quant block never spans two output rows. Load-time only, used to fuse separate Q/K/V or gate/up projections into a single larger GEMV dispatch. All parts must share dtype and (for 2D) K.</summary>
    public static unsafe Tensor ConcatRows(params Tensor[] parts)
    {
        DType dt = parts[0].DType;
        long elementsPerRow = parts[0].ElementCount / parts[0].Shape[0];
        long totalRows = 0;
        foreach (Tensor p in parts) totalRows += p.Shape[0];
        TensorShape outShape = parts[0].Shape.Rank == 1 ? new TensorShape(totalRows)
            : new TensorShape(totalRows, parts[0].Shape[1]);
        Tensor result = new(outShape, dt);
        long rowBytes = dt.ComputeByteCount(elementsPerRow);
        byte* dst = (byte*)result.DataPointer;
        long offset = 0;
        foreach (Tensor p in parts)
        {
            byte* src = (byte*)p.DataPointer;
            long bytes = p.Shape[0] * rowBytes;
            Buffer.MemoryCopy(src, dst + offset, bytes, bytes);
            offset += bytes;
        }
        return result;
    }

    /// <summary>Uploads <paramref name="source"/> once via a cheap, backend-agnostic <c>Scale(_, _, 1f)</c> identity op and returns the resident
    /// replacement, which gets the ordinary activation-cache binding every op output gets: no new backend API, no change to any shared op.
    /// Worth it for a tensor read on every iteration of a loop, not for one read once. Disposes the half-built copy and rethrows on failure;
    /// <paramref name="source"/>'s own disposal stays with the caller, which may only be borrowing it.</summary>
    public static Tensor MakeResident(IBackend backend, Tensor source)
    {
        Tensor resident = new(source.Shape, DType.F32);
        try
        {
            backend.Scale(resident, source, 1f);
        }
        catch
        {
            resident.Dispose();
            throw;
        }
        return resident;
    }

    /// <summary>Uploads a freshly built host RoPE table pair once and returns the resident replacements, disposing the host pair.</summary>
    public static (Tensor Cos, Tensor Sin) MakeRopeTableResident(IBackend backend, Tensor cos, Tensor sin)
    {
        Tensor residentCos = new(cos.Shape, DType.F32);
        Tensor residentSin = new(sin.Shape, DType.F32);
        try
        {
            backend.Scale(residentCos, cos, 1f);
            backend.Scale(residentSin, sin, 1f);
        }
        catch
        {
            residentCos.Dispose();
            residentSin.Dispose();
            throw;
        }
        finally
        {
            cos.Dispose();
            sin.Dispose();
        }
        return (residentCos, residentSin);
    }

    /// <summary>Projects <paramref name="hidden"/> <c>[1, t, hidden]</c> through the lm_head <paramref name="headW"/> into <paramref name="logits"/>: the fused
    /// quantized decode GEMV when the head's quant and <paramref name="hiddenSize"/> allow it, an F16-hidden quantized GEMM under low VRAM (no full-size F32
    /// staging for a large vocabulary), else the ordinary projection.</summary>
    /// <remarks>The divisibility gate mirrors the CUDA per-dtype dispatch: K-quants need <c>K % 256 == 0</c>, the 32-block quants <c>K % 32 == 0</c>; a blanket
    /// rule sent every head with <c>hidden ≡ 128 (mod 256)</c> down the dequant-then-cuBLAS route on every token.</remarks>
    public static void ProjectHead(IBackend backend, Tensor logits, Tensor hidden, Tensor headW, int hiddenSize, int t, bool lowVram)
    {
        bool fusedHead = t <= 8
            && (((headW.DType == DType.Q4_K || headW.DType == DType.Q5_K || headW.DType == DType.Q6_K) && hiddenSize % 256 == 0)
            || ((headW.DType == DType.Q8_0 || headW.DType == DType.Q4_0 || headW.DType == DType.Q5_0) && hiddenSize % 32 == 0));
        if (fusedHead)
        {
            Project(backend, logits, hidden, headW, bias: null, lowVram);
        }
        else if (lowVram && headW.DType.IsQuantized)
        {
            using Tensor hidF16 = new(hidden.Shape, DType.F16);
            backend.CastToF16(hidF16, hidden);
            backend.QuantizedMatMul(logits, hidF16, headW, bias: null);
        }
        else
        {
            Project(backend, logits, hidden, headW, bias: null, lowVram);
        }
    }
}
