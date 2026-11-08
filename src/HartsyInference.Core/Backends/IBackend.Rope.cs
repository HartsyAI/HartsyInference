using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

// Rotary position embedding in every pairing and layout the models use, plus the fused norm-then-rotate forms.
public partial interface IBackend
{
    /// <summary>Interleaved (2i,2i+1) partial rotary on <c>[b,heads,seq,headDim]</c>, in place, rotating the first rotDim dims.</summary>
    unsafe void OasisRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int batch, int heads, int seq, int headDim, int rotDim)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32) throw new NotSupportedException("OasisRopeInterleaved default fallback only supports F32.");
        int pairs = rotDim / 2;
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp2 = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < heads; h++)
                for (int s = 0; s < seq; s++)
                {
                    long xOff = (((long)b * heads + h) * seq + s) * headDim, cOff = (long)s * rotDim;
                    for (int p = 0; p < pairs; p++)
                    {
                        int i0 = 2 * p;
                        float re = xp[xOff + i0], im = xp[xOff + i0 + 1], c = cp[cOff + i0], sn = sp2[cOff + i0];
                        xp[xOff + i0] = re * c - im * sn;
                        xp[xOff + i0 + 1] = re * sn + im * c;
                    }
                }
    }

    /// <summary>In-place rotary on q/k <c>[B,L,numHeads,headDim]</c>: <c>out = x*cos + rotate_half(x)*sin</c> (Ideogram 4's ApplyRotary).
    /// Tolerates GQA: q and k may have different head counts — only batch, seqLen and headDim must agree. Was one
    /// loop nest sized from Q's <c>numHeads</c> indexing into K's buffer too: harmless for MHA callers (same shape
    /// either way), but whenever K had fewer heads than Q this read/wrote past K's actual element count — an
    /// out-of-bounds access, not just a wrong answer (same bug CUDA's and Vulkan's combined overloads had; caught
    /// via Dia's GQA decoder, 16 Q heads : 4 KV heads). Delegates to <see cref="ApplyRopeSingleReference"/> per
    /// tensor instead, which already derives its shape from whichever tensor it's given.</summary>
    unsafe void ApplyRope(Tensor q, Tensor k, Tensor cos, Tensor sin)
    {
        if (q.DType != DType.F32 || k.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRope default fallback only supports F32.");
        if (q.Shape.Rank != 4 || k.Shape.Rank != 4)
            throw new NotSupportedException($"ApplyRope expects token-major [B, L, heads, headDim] for q and k; got ranks {q.Shape.Rank}/{k.Shape.Rank}.");
        if (q.Shape[0] != k.Shape[0] || q.Shape[1] != k.Shape[1] || q.Shape[3] != k.Shape[3])
        {
            throw new NotSupportedException(
                $"ApplyRope requires q and k to share batch, seqLen and headDim (head COUNT may differ — GQA); got q={q.Shape}, k={k.Shape}.");
        }
        int headDim = (int)q.Shape[3];
        if (cos.Shape[cos.Shape.Rank - 1] != headDim || sin.Shape[sin.Shape.Rank - 1] != headDim)
        {
            throw new NotSupportedException(
                $"ApplyRope expects cos/sin's last dim to equal headDim ({headDim}); got cos={cos.Shape}, sin={sin.Shape}.");
        }
        ApplyRopeSingleReference(q, cos, sin);
        ApplyRopeSingleReference(k, cos, sin);
    }

    /// <summary>In-place <b>interleaved (GPT-J)</b> rotary on <c>x [B,L,numHeads,headDim]</c>; NOT same as <see cref="ApplyRopeSingle"/>.</summary>
    /// <param name="rotaryDim">0 (or &gt;=headDim) rotates every pair; otherwise only <c>[0, rotaryDim)</c> pairs rotate (HF partial-rotary).</param>
    unsafe void ApplyRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
        => ApplyRopeInterleavedReference(x, cos, sin, rotaryDim);

    /// <summary>The host implementation of <see cref="ApplyRopeInterleaved"/>, callable by a backend that
    /// overrides it and needs to bail on a dtype or rank its kernel does not cover.</summary>
    /// <remarks>Static because an override cannot reach its own interface default: <c>((IBackend)this).X(...)</c>
    /// binds back to the class and recurses until the stack ends. A lint fails the build on that spelling, and
    /// this is what it points callers at.</remarks>
    static unsafe void ApplyRopeInterleavedReference(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeInterleaved default fallback only supports F32.");
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        int half = headDim / 2;
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    float* vec = xPtr + vecOff;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        if (2 * i >= rdim) break;
                        float xe = vec[2 * i];
                        float xo = vec[2 * i + 1];
                        vec[2 * i] = xe * c[i] - xo * si[i];
                        vec[2 * i + 1] = xo * c[i] + xe * si[i];
                    }
                }
            }
    }

    /// <summary>Wan-Video interleaved in-place RoPE (shared path): <c>x [S,heads·headDim]</c>, cos/sin shared across heads.</summary>
    unsafe void WanRopeInterleaved(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < seqLen; s++)
            for (int h = 0; h < heads; h++)
            {
                long xoff = ((long)s * heads + h) * headDim;
                long coff = (long)s * headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[coff + i0], sn = sp[coff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
    }

    /// <summary>Matrix-Game 3.0 ActionModule: interleaved rope on <c>[sp,heads,tt,headDim]</c>, in place; grid row per broadcastSpatial.</summary>
    unsafe void Mg3RopeBatched(Tensor x, Tensor cos, Tensor sin, int sp, int heads, int tt, int headDim, int gh, int gw, bool broadcastSpatial)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, spn = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < sp; s++)
            for (int h = 0; h < heads; h++)
                for (int f = 0; f < tt; f++)
                {
                    long gridRow = broadcastSpatial ? f : (long)f * gh * gw + s;
                    long cOff = gridRow * headDim, xOff = (((long)s * heads + h) * tt + f) * headDim;
                    for (int i = 0; i < pairs; i++)
                    {
                        int i0 = 2 * i;
                        float re = xp[xOff + i0], im = xp[xOff + i0 + 1];
                        float c = cp[cOff + i0], sn = spn[cOff + i0];
                        xp[xOff + i0] = re * c - im * sn;
                        xp[xOff + i0 + 1] = re * sn + im * c;
                    }
                }
    }

    /// <summary>Per-head Wan interleaved RoPE (MG3 sigma_theta): like <see cref="WanRopeInterleaved"/>, cos/sin <c>[heads,S,headDim]</c>.</summary>
    unsafe void WanRopeInterleavedPerHead(Tensor x, Tensor cos, Tensor sin, int seqLen, int heads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int pairs = headDim / 2;
        for (int s = 0; s < seqLen; s++)
            for (int h = 0; h < heads; h++)
            {
                long xoff = ((long)s * heads + h) * headDim;
                long coff = ((long)h * seqLen + s) * headDim;
                for (int i = 0; i < pairs; i++)
                {
                    int i0 = 2 * i;
                    float re = xp[xoff + i0], im = xp[xoff + i0 + 1];
                    float c = cp[coff + i0], sn = sp[coff + i0];
                    xp[xoff + i0] = re * c - im * sn;
                    xp[xoff + i0 + 1] = re * sn + im * c;
                }
            }
    }

    /// <summary>LTX-2.5 NA-decoder rotary, in-place on <c>x [1,T,H,W,heads,headDim]</c>. Each head's channels split into
    /// three contiguous chunks — T at 0, H at <paramref name="splitT"/>, W at <c>splitT+splitH</c> — each rotated over
    /// interleaved <c>(2i, 2i+1)</c> pairs by that token's own t / h / w coordinate. The cos/sin tables are
    /// <c>[axisLength, pairs]</c> and are built by the caller so the F32 angle rounding stays identical everywhere.</summary>
    unsafe void Ltx25NaRope3d(Tensor x, Tensor cosT, Tensor sinT, Tensor cosH, Tensor sinH, Tensor cosW, Tensor sinW,
        int splitT, int splitH)
        => Ltx25NaRope3dReference(x, cosT, sinT, cosH, sinH, cosW, sinW, splitT, splitH);

    /// <summary>The managed <see cref="Ltx25NaRope3d"/> body — see <see cref="Na3dReference"/> for why it is separate.</summary>
    static unsafe void Ltx25NaRope3dReference(Tensor x, Tensor cosT, Tensor sinT, Tensor cosH, Tensor sinH,
        Tensor cosW, Tensor sinW, int splitT, int splitH)
    {
        Ltx25NaRope3dGeometry(x, splitT, splitH, out int dimT, out int dimH, out int dimW, out int heads,
            out int headDim, out int pairsT, out int pairsH, out int pairsW);
        int offsetH = splitT, offsetW = splitT + splitH;
        float* p = (float*)x.DataPointer;
        float* ct = (float*)cosT.DataPointer, st = (float*)sinT.DataPointer;
        float* ch = (float*)cosH.DataPointer, sh = (float*)sinH.DataPointer;
        float* cw = (float*)cosW.DataPointer, sw = (float*)sinW.DataPointer;

        for (int ti = 0; ti < dimT; ti++)
        for (int hi = 0; hi < dimH; hi++)
        for (int wi = 0; wi < dimW; wi++)
        {
            long tokenBase = (((long)ti * dimH + hi) * dimW + wi) * heads * headDim;
            for (int head = 0; head < heads; head++)
            {
                float* row = p + tokenBase + (long)head * headDim;
                RotateInterleaved(row, pairsT, ct, st, ti * pairsT);
                RotateInterleaved(row + offsetH, pairsH, ch, sh, hi * pairsH);
                RotateInterleaved(row + offsetW, pairsW, cw, sw, wi * pairsW);
            }
        }

        static void RotateInterleaved(float* row, int pairs, float* cos, float* sin, long tableOffset)
        {
            for (int i = 0; i < pairs; i++)
            {
                float even = row[2 * i], odd = row[2 * i + 1];
                float c = cos[tableOffset + i], s = sin[tableOffset + i];
                row[2 * i] = even * c - odd * s;
                row[2 * i + 1] = even * s + odd * c;
            }
        }
    }

    /// <summary>Validates an LTX-2.5 NA rope call and unpacks the grid and per-axis pair counts both the managed and
    /// the device path need.</summary>
    static void Ltx25NaRope3dGeometry(Tensor x, int splitT, int splitH, out int dimT, out int dimH, out int dimW,
        out int heads, out int headDim, out int pairsT, out int pairsH, out int pairsW)
    {
        if (x.DType != DType.F32)
            throw new NotSupportedException($"Ltx25NaRope3d supports F32 only — got {x.DType}.");
        if (x.Shape.Rank != 6 || x.Shape[0] != 1)
            throw new ArgumentException($"Ltx25NaRope3d expects [1, T, H, W, heads, headDim]; got {x.Shape}.", nameof(x));
        dimT = (int)x.Shape[1]; dimH = (int)x.Shape[2]; dimW = (int)x.Shape[3];
        heads = (int)x.Shape[4]; headDim = (int)x.Shape[5];
        int splitW = headDim - splitT - splitH;
        if (splitT <= 0 || splitH <= 0 || splitW <= 0 || splitT % 2 != 0 || splitH % 2 != 0 || splitW % 2 != 0)
            throw new ArgumentException(
                $"Ltx25NaRope3d needs a positive even 3-way split of head_dim {headDim}; got ({splitT}, {splitH}, {splitW}).",
                nameof(splitT));
        pairsT = splitT / 2; pairsH = splitH / 2; pairsW = splitW / 2;
    }

    /// <summary>LTX-2 "split" rotary (rotate-half, per-head cos), in-place on <c>x [seqLen,dim]</c>; matches <c>LtxVideo2Rope</c>.</summary>
    unsafe void Ltx2SplitRope(Tensor x, Tensor cos, Tensor sin, int seqLen, int numHeads, int headDim)
    {
        float* xp = (float*)x.DataPointer, cp = (float*)cos.DataPointer, sp = (float*)sin.DataPointer;
        int r = headDim / 2;
        int dim = numHeads * headDim;
        int cosWidth = dim / 2;
        for (int s = 0; s < seqLen; s++)
        {
            long xoff = (long)s * dim;
            long coff = (long)s * cosWidth;
            for (int h = 0; h < numHeads; h++)
            {
                long headBase = xoff + (long)h * headDim;
                long cosBase = coff + (long)h * r;
                for (int i = 0; i < r; i++)
                {
                    float a = xp[headBase + i];
                    float b = xp[headBase + i + r];
                    float c = cp[cosBase + i], sn = sp[cosBase + i];
                    xp[headBase + i] = a * c - b * sn;
                    xp[headBase + i + r] = b * c + a * sn;
                }
            }
        }
    }

    /// <summary>Precomputes the full RoPE cos/sin table <c>[maxPos,headDim]</c> device-resident, so a decode step only indexes a row.</summary>
    /// <param name="splitHalfPartial">Split-half vs. interleaved rotary pairing when rotaryDim &lt; headDim; ignored for full rotary.</param>
    (Tensor cos, Tensor sin) BuildRopeTableDevice(int maxPos, int headDim, int rotaryDim, float theta, HartsyInference.Core.Rope.RopeScaling scaling, bool splitHalfPartial = false)
        => throw new NotSupportedException("BuildRopeTableDevice requires GraphDecodeSupported.");

    /// <summary>In-place rotary on <c>x [B,L,numHeads,headDim]</c> — same math as <see cref="ApplyRope"/>, for GQA head-count mismatch.</summary>
    unsafe void ApplyRopeSingle(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
        => ApplyRopeSingleReference(x, cos, sin, rotaryDim);

    /// <summary>The managed <see cref="ApplyRopeSingle"/> body, callable directly. A backend override must call
    /// THIS to fall back — <c>((IBackend)this).ApplyRopeSingle(...)</c> re-enters the override through interface
    /// dispatch and recurses until the stack overflows.</summary>
    static unsafe void ApplyRopeSingleReference(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeSingle default fallback only supports F32.");
        int batch = (int)x.Shape[0];
        int seqLen = (int)x.Shape[1];
        int numHeads = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        // Partial rotary (Phi-4 / StableLM): rotate only the first rotaryDim dims of each head (NEOX pairing
        // (i, i+half), half = rotaryDim/2), leaving the rest unchanged. 0 / full = the whole head. cos/sin keep
        // headDim stride (only their first rotaryDim entries are read).
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int s = 0; s < seqLen; s++)
            {
                long freqBase = ((long)b * seqLen + s) * headDim;
                for (int h = 0; h < numHeads; h++)
                {
                    long vecOff = (((long)b * seqLen + s) * numHeads + h) * headDim;
                    float* vec = xPtr + vecOff;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        float lower = vec[i];
                        float upper = vec[i + half];
                        vec[i] = lower * c[i] - upper * si[i];
                        vec[i + half] = upper * c[i + half] + lower * si[i + half];
                    }
                }
            }
    }

    /// <summary>Head-major <see cref="ApplyRopeSingle"/>: in-place rotary on <c>x [B,heads,L,headDim]</c>, cos/sin still <c>[B,L,headDim]</c>.</summary>
    /// <remarks>Lets q/k from <see cref="QkvSplitNormHeadMajor"/> be roped in place and fed straight to attention.</remarks>
    unsafe void ApplyRopeSingleHeadMajor(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
        => ApplyRopeSingleHeadMajorReference(x, cos, sin, rotaryDim);

    /// <summary>The managed <see cref="ApplyRopeSingleHeadMajor"/> body, callable directly. A backend override must
    /// call THIS to fall back — <c>((IBackend)this).ApplyRopeSingleHeadMajor(...)</c> re-enters the override through
    /// interface dispatch and recurses until the stack overflows.</summary>
    static unsafe void ApplyRopeSingleHeadMajorReference(Tensor x, Tensor cos, Tensor sin, int rotaryDim = 0)
    {
        if (x.DType != DType.F32 || cos.DType != DType.F32 || sin.DType != DType.F32)
            throw new NotSupportedException("ApplyRopeSingleHeadMajor default fallback only supports F32.");
        if (x.Shape.Rank != 4)
            throw new ArgumentException($"ApplyRopeSingleHeadMajor needs x shaped [B,heads,seq,headDim], got {x.Shape}.", nameof(x));
        int batch = (int)x.Shape[0];
        int numHeads = (int)x.Shape[1];
        int seqLen = (int)x.Shape[2];
        int headDim = (int)x.Shape[3];
        // A token-major x is the same element count with heads/seq swapped; only the cos/sin row count catches it.
        long tableElements = (long)batch * seqLen * headDim;
        if (cos.ElementCount != tableElements || sin.ElementCount != tableElements)
            throw new ArgumentException(
                $"ApplyRopeSingleHeadMajor expects cos/sin [B,seq,headDim] for x {x.Shape}; got cos={cos.Shape}, sin={sin.Shape}.",
                cos.ElementCount != tableElements ? nameof(cos) : nameof(sin));
        int rdim = rotaryDim <= 0 || rotaryDim > headDim ? headDim : rotaryDim;
        int half = rdim / 2;
        float* xPtr = (float*)x.DataPointer;
        float* cosPtr = (float*)cos.DataPointer;
        float* sinPtr = (float*)sin.DataPointer;
        for (int b = 0; b < batch; b++)
            for (int h = 0; h < numHeads; h++)
                for (int s = 0; s < seqLen; s++)
                {
                    float* vec = xPtr + (((long)b * numHeads + h) * seqLen + s) * headDim;
                    long freqBase = ((long)b * seqLen + s) * headDim;
                    float* c = cosPtr + freqBase;
                    float* si = sinPtr + freqBase;
                    for (int i = 0; i < half; i++)
                    {
                        float lower = vec[i];
                        float upper = vec[i + half];
                        vec[i] = lower * c[i] - upper * si[i];
                        vec[i + half] = upper * c[i + half] + lower * si[i + half];
                    }
                }
    }

    /// <summary>LTX-2 attention QK path in one pass: full-row RMS norm over <c>heads*headDim</c>, then optional
    /// SPLIT RoPE, then a head-major emit — <c>input [seq, heads*headDim]</c> to <c>output [1, heads, seq, headDim]</c>.
    /// Pass <paramref name="cos"/>/<paramref name="sin"/> null to skip the rotation (text cross-attention).
    /// INTERLEAVED rope is not served here; that caller must keep the unfused sequence.
    /// Default: composes the three ops it replaces, which is exactly what the unfused path did.</summary>
    unsafe void Ltx2QkNormRopeHeadMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin,
        int seq, int heads, int headDim, float eps)
    {
        Tensor normed = new Tensor(input.Shape, input.DType);
        try
        {
            RmsNorm(normed, input, normWeight, eps);
            if (cos is not null && sin is not null) Ltx2SplitRope(normed, cos, sin, seq, heads, headDim);
            Permute0213(output, normed, seq, heads, headDim);
        }
        finally { normed.Dispose(); }
    }

    /// <summary>Token-major twin of <see cref="Ltx2QkNormRopeHeadMajor"/> — <c>output</c> keeps the input's
    /// <c>[seq, heads*headDim]</c> shape. Default: composes the two ops it replaces.</summary>
    unsafe void Ltx2QkNormRopeTokenMajor(Tensor output, Tensor input, Tensor normWeight, Tensor? cos, Tensor? sin,
        int seq, int heads, int headDim, float eps)
    {
        RmsNorm(output, input, normWeight, eps);
        if (cos is not null && sin is not null) Ltx2SplitRope(output, cos, sin, seq, heads, headDim);
    }
}
