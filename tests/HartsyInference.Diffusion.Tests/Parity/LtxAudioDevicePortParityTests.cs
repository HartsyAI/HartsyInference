using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Music;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Parity tests for the LTX-2 audio decode-tail GPU-residency port (VIDEO_GENPERF_PLAN Phase 4, audio
/// half): the vocoder's replicate pads / crops / zero pads / mel flatten and the audio VAE's causal top-pad,
/// upsampler row-drop and pixel norm now route through <see cref="IBackend"/> ops (<see cref="LtxAudioDeviceOps"/>
/// + WanRmsNormChannel). Each test compares against the pre-port host reference loops, copied here verbatim.</summary>
public unsafe class LtxAudioDevicePortParityTests
{
    [Theory]
    [InlineData(2, 37, 7, 7)]     // resampler geometry (stereo waveform)
    [InlineData(768, 129, 5, 5)]  // anti-alias up-pad geometry (wide resblock)
    [InlineData(96, 64, 6, 6)]    // anti-alias down-pad geometry
    [InlineData(16, 1, 3, 4)]     // degenerate single-sample time axis
    public void ReplicatePadTime_MatchesHostReference(int c, int t, int padL, int padR)
    {
        CpuBackend backend = new();
        Random rng = new(11);
        Tensor x = Rand(rng, [1, c, t]);

        Tensor actual = LtxAudioDeviceOps.ReplicatePadTime(backend, x, padL, padR);
        Tensor expected = RefReplicatePad(x, padL, padR);
        AssertExact(expected, actual);
        x.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    [Theory]
    [InlineData(2, 200, 42, 40)]  // resampler crop geometry (ratio 3, width 7)
    [InlineData(384, 61, 22, 23)] // anti-alias up crop
    public void CropTime_MatchesHostReference(int c, int t, int left, int right)
    {
        CpuBackend backend = new();
        Random rng = new(13);
        Tensor x = Rand(rng, [1, c, t]);

        Tensor actual = LtxAudioDeviceOps.CropTime(backend, x, left, right);
        Tensor expected = RefCrop(x, left, right);
        AssertExact(expected, actual);
        x.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    [Fact]
    public void PadRightZero_MatchesHostReference()
    {
        CpuBackend backend = new();
        Random rng = new(17);
        int c = 2, t = 123, pad = 37;
        Tensor x = Rand(rng, [1, c, t]);

        Tensor actual = LtxAudioDeviceOps.PadRightZero(backend, x, pad);
        Tensor expected = RefPadRight(x, pad);
        AssertExact(expected, actual);
        x.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    [Theory]
    [InlineData(1, 512, 6, 16, 2)]   // deep audio-VAE level, k=3
    [InlineData(1, 128, 25, 64, 2)]  // wide shallow level
    [InlineData(2, 8, 5, 16, 4)]     // batch>1 + larger kernel
    public void PadTimeTopZero_MatchesCausalPadReference(int b, int c, int h, int w, int padTop)
    {
        CpuBackend backend = new();
        Random rng = new(19);
        Tensor x = Rand(rng, [b, c, h, w]);

        Tensor actual = LtxAudioDeviceOps.PadTimeTopZero(backend, x, padTop);
        Tensor expected = RefCausalTopPad(x, padTop);
        AssertExact(expected, actual);
        x.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    [Theory]
    [InlineData(512, 12, 32)]
    [InlineData(256, 7, 64)]
    public void DropFirstTimeRow_MatchesHostReference(int c, int h, int w)
    {
        CpuBackend backend = new();
        Random rng = new(23);
        Tensor x = Rand(rng, [1, c, h, w]);

        Tensor actual = LtxAudioDeviceOps.DropFirstTimeRow(backend, x);
        Tensor expected = RefDropFirstRow(x);
        AssertExact(expected, actual);
        x.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    [Theory]
    [InlineData(2, 25, 64)]   // stage-1 / single-stage input (mel from the audio VAE)
    [InlineData(2, 40, 64)]
    public void FlattenMel_MatchesHostReference(int c, int frames, int melBins)
    {
        CpuBackend backend = new();
        Random rng = new(29);
        Tensor mel = Rand(rng, [1, c, frames, melBins]);

        Tensor actual = LtxAudioDeviceOps.FlattenMel(backend, mel);
        Tensor expected = RefFlattenMel(mel);
        AssertExact(expected, actual);
        mel.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    /// <summary>The BWE input build (<c>FlattenMel(TransposeLast(logMel))</c> in the pre-port code) is a pure
    /// identity relabel of the log-mel's <c>[1, C, mel, frames]</c> memory — the port therefore emits the
    /// generator-ready <c>[1, C·mel, frames]</c> tensor directly from MelSpectrogram.</summary>
    [Fact]
    public void BweInputBuild_IsIdentityRelabelOfLogMelMemory()
    {
        Random rng = new(31);
        int c = 2, mel = 64, frames = 53;
        Tensor logMel = Rand(rng, [1, c, mel, frames]);

        Tensor transposed = RefTransposeLast(logMel);
        Tensor flattened = RefFlattenMel(transposed);
        float* lp = (float*)logMel.DataPointer;
        float* fp = (float*)flattened.DataPointer;
        for (long i = 0; i < logMel.Shape.ElementCount; i++)
            Assert.True(lp[i] == fp[i], $"relabel mismatch at {i}: {lp[i]} vs {fp[i]}");
        logMel.Dispose();
        transposed.Dispose();
        flattened.Dispose();
    }

    /// <summary>PixelNorm now runs as WanRmsNormChannel with eps folded to <c>sqrt(C·eps)</c> — equal to the
    /// reference <c>x/sqrt(mean_C(x²)+eps)</c> within float rounding at signal magnitudes.</summary>
    [Theory]
    [InlineData(512, 6, 16)]
    [InlineData(128, 25, 64)]
    public void PixelNorm_FoldedEps_MatchesReference(int c, int h, int w)
    {
        IBackend backend = new CpuBackend();
        Random rng = new(37);
        Tensor x = Rand(rng, [1, c, h, w]);

        Tensor actual = new Tensor(x.Shape, DType.F32);
        backend.WanRmsNormChannel(actual, x, null, MathF.Sqrt(c * 1e-6f));
        Tensor expected = RefPixelNorm(x, 1e-6f);
        float* ap = (float*)actual.DataPointer;
        float* ep = (float*)expected.DataPointer;
        double sumSqDiff = 0, sumSqRef = 0;
        for (long i = 0; i < expected.Shape.ElementCount; i++)
        {
            double d = ap[i] - ep[i];
            sumSqDiff += d * d;
            sumSqRef += (double)ep[i] * ep[i];
        }
        double relL2 = Math.Sqrt(sumSqDiff / Math.Max(sumSqRef, 1e-30));
        Assert.True(relL2 <= 1e-5, $"PixelNorm relL2 {relL2:E3} > 1e-5 (c={c})");
        x.Dispose();
        actual.Dispose();
        expected.Dispose();
    }

    /// <summary>The vocoder tail <c>clamp(residual+skip,-1,1)</c>+crop now composes SliceLastDim/Add/Clamp.</summary>
    [Fact]
    public void AddClampCrop_OpComposition_MatchesHostReference()
    {
        CpuBackend backend = new();
        Random rng = new(41);
        int c = 2, rT = 150, kT = 144, outputSamples = 140;
        Tensor residual = Rand(rng, [1, c, rT]);
        Tensor skip = Rand(rng, [1, c, kT]);
        // Push some sums out of [-1,1] so the clamp is exercised.
        float* rp0 = (float*)residual.DataPointer;
        for (long i = 0; i < residual.Shape.ElementCount; i += 5) rp0[i] *= 3f;

        int outLen = Math.Min(Math.Min(rT, kT), outputSamples);
        Tensor rc = LtxAudioDeviceOps.CropTime(backend, residual, 0, rT - outLen);
        Tensor kc = LtxAudioDeviceOps.CropTime(backend, skip, 0, kT - outLen);
        Tensor actual = new Tensor(new TensorShape(1, c, outLen), DType.F32);
        backend.Add(actual, rc, kc);
        backend.Clamp(actual, actual, -1f, 1f);
        rc.Dispose();
        kc.Dispose();

        Tensor expected = RefAddClampCrop(residual, skip, outputSamples);
        AssertExact(expected, actual);
        residual.Dispose();
        skip.Dispose();
        expected.Dispose();
        actual.Dispose();
    }

    private static void AssertExact(Tensor expected, Tensor actual)
    {
        Assert.Equal(expected.Shape.ElementCount, actual.Shape.ElementCount);
        float* ep = (float*)expected.DataPointer;
        float* ap = (float*)actual.DataPointer;
        for (long i = 0; i < expected.Shape.ElementCount; i++)
            Assert.True(ep[i] == ap[i], $"mismatch at {i}: {ap[i]} vs {ep[i]}");
    }

    /// <summary>Pre-port host replicate-pad reference (verbatim from LtxBigVganGenerator.AntiAlias).</summary>
    private static Tensor RefReplicatePad(Tensor x, int padL, int padR)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        int t2 = t + padL + padR;
        Tensor o = new Tensor(new TensorShape(1, c, t2), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            float* src = xp + (long)ci * t;
            float* dst = op + (long)ci * t2;
            float left = src[0], right = src[t - 1];
            for (int i = 0; i < padL; i++) dst[i] = left;
            Buffer.MemoryCopy(src, dst + padL, (long)t * 4, (long)t * 4);
            for (int i = 0; i < padR; i++) dst[padL + t + i] = right;
        }
        return o;
    }

    /// <summary>Pre-port host crop reference (verbatim from LtxBigVganGenerator.AntiAlias).</summary>
    private static Tensor RefCrop(Tensor x, int left, int right)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2];
        int t2 = t - left - right;
        Tensor o = new Tensor(new TensorShape(1, c, t2), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            Buffer.MemoryCopy(xp + (long)ci * t + left, op + (long)ci * t2, (long)t2 * 4, (long)t2 * 4);
        return o;
    }

    /// <summary>Pre-port host right-pad reference (verbatim from LtxAudioVocoder).</summary>
    private static Tensor RefPadRight(Tensor x, int pad)
    {
        int c = (int)x.Shape[1], t = (int)x.Shape[2], t2 = t + pad;
        Tensor o = new Tensor(new TensorShape(1, c, t2), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
        {
            new Span<float>(op + (long)ci * t2, t2).Clear();
            Buffer.MemoryCopy(xp + (long)ci * t, op + (long)ci * t2, (long)t * 4, (long)t * 4);
        }
        return o;
    }

    /// <summary>Pre-port host causal top-pad reference (verbatim from LtxAudioCausalConv2d).</summary>
    private static Tensor RefCausalTopPad(Tensor x, int padTop)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], h = (int)x.Shape[2], w = (int)x.Shape[3];
        Tensor padded = new Tensor(new TensorShape(b, c, h + padTop, w), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* pp = (float*)padded.DataPointer;
        long inHW = (long)h * w, padHW = (long)(h + padTop) * w;
        for (int bc = 0; bc < b * c; bc++)
        {
            new Span<float>(pp + bc * padHW, padTop * w).Clear();
            Buffer.MemoryCopy(sp + bc * inHW, pp + bc * padHW + (long)padTop * w, inHW * sizeof(float), inHW * sizeof(float));
        }
        return padded;
    }

    /// <summary>Pre-port host first-row-drop reference (verbatim from LtxAudioUpsample).</summary>
    private static Tensor RefDropFirstRow(Tensor conv)
    {
        int b = (int)conv.Shape[0], ch = (int)conv.Shape[1], hh = (int)conv.Shape[2], ww = (int)conv.Shape[3];
        Tensor outT = new Tensor(new TensorShape(b, ch, hh - 1, ww), DType.F32);
        float* sp = (float*)conv.DataPointer;
        float* op = (float*)outT.DataPointer;
        long inHW = (long)hh * ww, outHW = (long)(hh - 1) * ww;
        for (int bc = 0; bc < b * ch; bc++)
            Buffer.MemoryCopy(sp + bc * inHW + ww, op + bc * outHW, outHW * sizeof(float), outHW * sizeof(float));
        return outT;
    }

    /// <summary>Pre-port host mel-flatten reference (verbatim from LtxAudioVocoder).</summary>
    private static Tensor RefFlattenMel(Tensor mel)
    {
        int c = (int)mel.Shape[1], frames = (int)mel.Shape[2], melBins = (int)mel.Shape[3];
        Tensor o = new Tensor(new TensorShape(1, c * melBins, frames), DType.F32);
        float* xp = (float*)mel.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int t = 0; t < frames; t++)
                for (int bin = 0; bin < melBins; bin++)
                    op[(long)(ci * melBins + bin) * frames + t] = xp[((long)ci * frames + t) * melBins + bin];
        return o;
    }

    /// <summary>Pre-port host last-two-axes transpose reference (verbatim from LtxAudioVocoder).</summary>
    private static Tensor RefTransposeLast(Tensor x)
    {
        int c = (int)x.Shape[1], a = (int)x.Shape[2], b = (int)x.Shape[3];
        Tensor o = new Tensor(new TensorShape(1, c, b, a), DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)o.DataPointer;
        for (int ci = 0; ci < c; ci++)
            for (int ai = 0; ai < a; ai++)
                for (int bi = 0; bi < b; bi++)
                    op[((long)ci * b + bi) * a + ai] = xp[((long)ci * a + ai) * b + bi];
        return o;
    }

    /// <summary>Pre-port host pixel-norm reference (verbatim from LtxAudioPixelNorm).</summary>
    private static Tensor RefPixelNorm(Tensor x, float eps)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1];
        long spatial = x.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                long basePos = (long)bi * c * spatial + s;
                double sum = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[basePos + (long)ci * spatial]; sum += (double)v * v; }
                float inv = 1f / MathF.Sqrt((float)(sum / c) + eps);
                for (int ci = 0; ci < c; ci++) { long off = basePos + (long)ci * spatial; op[off] = xp[off] * inv; }
            }
        return outT;
    }

    /// <summary>Pre-port host add+clamp+crop reference (verbatim from LtxAudioVocoder).</summary>
    private static Tensor RefAddClampCrop(Tensor residual, Tensor skip, int outputSamples)
    {
        int c = (int)residual.Shape[1];
        int len = Math.Min((int)residual.Shape[2], (int)skip.Shape[2]);
        int outLen = Math.Min(len, outputSamples);
        Tensor o = new Tensor(new TensorShape(1, c, outLen), DType.F32);
        float* rp = (float*)residual.DataPointer;
        float* kp = (float*)skip.DataPointer;
        float* op = (float*)o.DataPointer;
        int rT = (int)residual.Shape[2], kT = (int)skip.Shape[2];
        for (int ci = 0; ci < c; ci++)
            for (int t = 0; t < outLen; t++)
            {
                float v = rp[(long)ci * rT + t] + kp[(long)ci * kT + t];
                op[(long)ci * outLen + t] = Math.Clamp(v, -1f, 1f);
            }
        return o;
    }

    private static Tensor Rand(Random rng, long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
