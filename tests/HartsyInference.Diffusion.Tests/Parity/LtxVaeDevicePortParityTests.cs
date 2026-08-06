using Xunit;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Parity tests for the LTX VAE decode-tail GPU-residency port (VIDEO_GENPERF_PLAN Phase 4): the upsampler
/// pixel-shuffle + channel-repeat + residual add, the final pixel-unshuffle, and the channel-RMS norms now route
/// through <see cref="IBackend"/> ops (Permute0213 group-swap chain / SliceRows / Concat / UnpatchifyVae /
/// WanRmsNormChannel). Each test compares against the pre-port host reference loops, copied here verbatim.</summary>
public unsafe class LtxVaeDevicePortParityTests
{
    /// <summary>All 22B upsampler stride/upscale geometries: spatio-temporal (8,2)/(8,1), temporal (2,2), spatial (4,2).</summary>
    [Theory]
    [InlineData(2, 2, 2, 2, 16)]   // spatio-temporal, upscale 2 (stage 0)
    [InlineData(2, 2, 2, 1, 16)]   // spatio-temporal, upscale 1 (stage 1)
    [InlineData(2, 1, 1, 2, 8)]    // temporal, upscale 2 (stage 2; residual repeats == 1)
    [InlineData(1, 2, 2, 2, 8)]    // spatial, upscale 2 (stage 3)
    public void Upsampler_DevicePath_MatchesHostReference(int st0, int st1, int st2, int upscale, int inC)
    {
        CpuBackend backend = new();
        int stProd = st0 * st1 * st2;
        int convOutC = inC * stProd / upscale;
        int f = 3, h = 4, w = 5;
        Random rng = new(97);

        Dictionary<string, Tensor> weights = new()
        {
            ["up.conv.conv.weight"] = Rand(rng, [convOutC, inC, 3, 3, 3]),
            ["up.conv.conv.bias"] = Rand(rng, [convOutC]),
        };
        LtxVaeUpsampler3d upsampler = new(inC, (st0, st1, st2), upscaleFactor: upscale, residual: true, isCausal: false);
        upsampler.LoadWeights(weights, "up");

        Tensor x = Rand(rng, [1, inC, f, h, w]);
        Tensor actual = upsampler.Forward(backend, x);

        // Reference: identical conv, then the pre-port host shuffle + repeat + add loops.
        CausalConv3d conv = new(weights["up.conv.conv.weight"], weights["up.conv.conv.bias"],
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: false);
        Tensor convOut = conv.Forward(backend, x);
        int xOut = convOutC / stProd;
        Tensor expected = RefPixelShuffle(convOut, 1, convOutC, xOut, f, h, w, st0, st1, st2);
        convOut.Dispose();
        int y = inC / stProd;
        Tensor resShuf = RefPixelShuffle(x, 1, inC, y, f, h, w, st0, st1, st2);
        Tensor residual = RefRepeatChannels(resShuf, stProd / upscale);
        resShuf.Dispose();
        float* ep = (float*)expected.DataPointer;
        float* rp = (float*)residual.DataPointer;
        for (long i = 0; i < expected.Shape.ElementCount; i++) ep[i] += rp[i];
        residual.Dispose();

        Assert.Equal(expected.Shape.ElementCount, actual.Shape.ElementCount);
        Assert.Equal((long)xOut, actual.Shape[1]);
        Assert.Equal((long)(f * st0 - (st0 - 1)), actual.Shape[2]);
        Assert.Equal((long)(h * st1), actual.Shape[3]);
        Assert.Equal((long)(w * st2), actual.Shape[4]);
        float* ap = (float*)actual.DataPointer;
        for (long i = 0; i < expected.Shape.ElementCount; i++)
            Assert.True(ap[i] == ep[i], $"mismatch at {i}: {ap[i]} vs {ep[i]} (st=({st0},{st1},{st2}), up={upscale})");
        expected.Dispose();
        actual.Dispose();
        x.Dispose();
    }

    [Fact]
    public void UnpatchifyVae_MatchesLtxPixelUnshuffleLayout()
    {
        IBackend backend = new CpuBackend();
        int oc = 3, p = 4, f = 2, h = 3, w = 5;
        Random rng = new(31);
        Tensor packed = Rand(rng, [1, oc * p * p, f, h, w]);

        Tensor actual = new Tensor(new TensorShape([1L, oc, f, (long)h * p, (long)w * p]), DType.F32);
        backend.UnpatchifyVae(actual, packed, p);

        Tensor expected = RefPixelUnshuffle(packed, p);
        float* ap = (float*)actual.DataPointer;
        float* ep = (float*)expected.DataPointer;
        for (long i = 0; i < expected.Shape.ElementCount; i++)
            Assert.True(ap[i] == ep[i], $"mismatch at {i}: {ap[i]} vs {ep[i]}");
        packed.Dispose();
        actual.Dispose();
        expected.Dispose();
    }

    [Fact]
    public void WanRmsNormChannel_MatchesLtxChannelRms()
    {
        IBackend backend = new CpuBackend();
        int c = 64, f = 2, h = 5, w = 7;
        Random rng = new(53);
        Tensor x = Rand(rng, [1, c, f, h, w]);

        Tensor actual = new Tensor(x.Shape, DType.F32);
        backend.WanRmsNormChannel(actual, x, null, 1e-8f);

        Tensor expected = RefChannelRms(x, c);
        float* ap = (float*)actual.DataPointer;
        float* ep = (float*)expected.DataPointer;
        for (long i = 0; i < expected.Shape.ElementCount; i++)
        {
            float tol = 1e-5f * MathF.Max(1f, MathF.Abs(ep[i]));
            Assert.True(MathF.Abs(ap[i] - ep[i]) <= tol, $"mismatch at {i}: {ap[i]} vs {ep[i]}");
        }
        x.Dispose();
        actual.Dispose();
        expected.Dispose();
    }

    /// <summary>Pre-port host pixel-shuffle reference (verbatim from LtxVaeUpsampler3d).</summary>
    private static Tensor RefPixelShuffle(Tensor src, int b, int srcC, int xOut, int f, int h, int w, int st0, int st1, int st2)
    {
        int outF = f * st0 - (st0 - 1);
        int outH = h * st1, outW = w * st2;
        Tensor outT = new Tensor(new TensorShape([(long)b, xOut, outF, outH, outW]), DType.F32);
        float* sp = (float*)src.DataPointer;
        float* op = (float*)outT.DataPointer;
        long srcFrame = (long)h * w;
        for (int bi = 0; bi < b; bi++)
            for (int x = 0; x < xOut; x++)
                for (int fo = 0; fo < outF; fo++)
                {
                    int full = fo + (st0 - 1);
                    int fi = full / st0, s0 = full % st0;
                    for (int ho = 0; ho < outH; ho++)
                    {
                        int hi = ho / st1, s1 = ho % st1;
                        for (int wo = 0; wo < outW; wo++)
                        {
                            int wi = wo / st2, s2 = wo % st2;
                            int ch = ((x * st0 + s0) * st1 + s1) * st2 + s2;
                            long srcOff = (((long)bi * srcC + ch) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * xOut + x) * outF + fo) * ((long)outH * outW) + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
                        }
                    }
                }
        return outT;
    }

    /// <summary>Pre-port host channel-repeat reference (verbatim from LtxVaeUpsampler3d).</summary>
    private static Tensor RefRepeatChannels(Tensor x, int repeats)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        Tensor outT = new Tensor(new TensorShape([(long)b, c * repeats, f, h, w]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long chunk = (long)f * h * w;
        for (int bi = 0; bi < b; bi++)
            for (int r = 0; r < repeats; r++)
                for (int ci = 0; ci < c; ci++)
                {
                    long srcOff = ((long)bi * c + ci) * chunk;
                    long dstOff = ((long)bi * c * repeats + r * c + ci) * chunk;
                    Buffer.MemoryCopy(sp + srcOff, op + dstOff, chunk * 4, chunk * 4);
                }
        return outT;
    }

    /// <summary>Pre-port host pixel-unshuffle reference (verbatim from LtxVideo2VaeDecoder, patch p).</summary>
    private static Tensor RefPixelUnshuffle(Tensor x, int p)
    {
        int b = (int)x.Shape[0], srcC = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int oc = srcC / (p * p);
        int outH = h * p, outW = w * p;
        Tensor outT = new Tensor(new TensorShape([(long)b, oc, f, outH, outW]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long srcFrame = (long)h * w, dstFrame = (long)outH * outW;
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < oc; c++)
                for (int fi = 0; fi < f; fi++)
                    for (int ho = 0; ho < outH; ho++)
                    {
                        int hi = ho / p, pb = ho % p;
                        for (int wo = 0; wo < outW; wo++)
                        {
                            int wi = wo / p, pa = wo % p;
                            int ch = (c * p + pa) * p + pb;
                            long srcOff = (((long)bi * srcC + ch) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * oc + c) * f + fi) * dstFrame + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
                        }
                    }
        return outT;
    }

    /// <summary>Pre-port host channel-RMS reference (verbatim from the LTX decoders, eps 1e-8).</summary>
    private static Tensor RefChannelRms(Tensor x, int c)
    {
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                long basePos = (long)bi * c * spatial + s;
                double sum = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[basePos + (long)ci * spatial]; sum += (double)v * v; }
                float inv = 1f / MathF.Sqrt((float)(sum / c) + 1e-8f);
                for (int ci = 0; ci < c; ci++) { long off = basePos + (long)ci * spatial; op[off] = xp[off] * inv; }
            }
        return outT;
    }

    private static Tensor Rand(Random rng, long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }
}
