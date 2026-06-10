using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Vae;

namespace SharpInference.Diffusion.Tests;

/// <summary>Correctness tests for the reusable <see cref="CausalConv3d"/> video-VAE op (no GPU/checkpoint). Verifies the Conv2D-decomposition against the existing <see cref="SharpInference.Core.Backends.IBackend.Conv2D"/> and the causal temporal property that any streaming video decode relies on.</summary>
public unsafe class CausalConv3dTests
{
    [Fact]
    public void Kt1_EqualsConv2DPerFrame()
    {
        CpuBackend backend = new();
        int cOut = 2, cIn = 2, kh = 3, kw = 3, t = 3, h = 4, w = 4;
        Tensor weight = RandomTensor([cOut, cIn, 1, kh, kw], seed: 11);
        Tensor bias = RandomTensor([cOut], seed: 12);
        Tensor input = RandomTensor([1, cIn, t, h, w], seed: 13);

        CausalConv3d conv = new(weight, bias, strideT: 1, strideH: 1, strideW: 1, padT: 0, padH: 1, padW: 1);
        Tensor outT = conv.Forward(backend, input);
        Assert.Equal(t, (int)outT.Shape[2]);

        Tensor w2d = SliceTemporal(weight, 0, cOut, cIn, kh, kw);
        for (int ti = 0; ti < t; ti++)
        {
            Tensor frame = ExtractFrame(input, ti, cIn, h, w);
            Tensor expected = new Tensor(new TensorShape(1, cOut, h, w), DType.F32);
            backend.Conv2D(expected, frame, w2d, bias, 1, 1, 1, 1);
            AssertFrameClose(outT, ti, expected, cOut, h, w, 1e-4f);
            frame.Dispose(); expected.Dispose();
        }
    }

    [Fact]
    public void SingleFrame_Kt3_ReducesToLastTemporalSlice()
    {
        CpuBackend backend = new();
        int cOut = 2, cIn = 2, kh = 3, kw = 3, h = 4, w = 4;
        Tensor weight = RandomTensor([cOut, cIn, 3, kh, kw], seed: 21);
        Tensor bias = RandomTensor([cOut], seed: 22);
        Tensor input = RandomTensor([1, cIn, 1, h, w], seed: 23);  // T=1 image

        CausalConv3d conv = new(weight, bias, padT: 1, padH: 1, padW: 1);
        Tensor outT = conv.Forward(backend, input);
        Assert.Equal(1, (int)outT.Shape[2]);  // causal: 2 zero pad frames + 1 input, kt 3 → Tout 1

        // Taps 0,1 see zero frames → only the last temporal weight slice contributes.
        Tensor wLast = SliceTemporal(weight, 2, cOut, cIn, kh, kw);
        Tensor frame = ExtractFrame(input, 0, cIn, h, w);
        Tensor expected = new Tensor(new TensorShape(1, cOut, h, w), DType.F32);
        backend.Conv2D(expected, frame, wLast, bias, 1, 1, 1, 1);
        AssertFrameClose(outT, 0, expected, cOut, h, w, 1e-4f);
    }

    [Fact]
    public void Causality_OutputFrameDoesNotDependOnFutureInput()
    {
        CpuBackend backend = new();
        int cOut = 1, cIn = 1, h = 3, w = 3;
        Tensor weight = RandomTensor([cOut, cIn, 3, 3, 3], seed: 31);
        Tensor inputA = RandomTensor([1, cIn, 2, h, w], seed: 32);
        Tensor inputB = Clone(inputA);
        // Perturb ONLY frame 1 of inputB.
        float* b = (float*)inputB.DataPointer;
        long frame = (long)h * w;
        for (long i = 0; i < frame; i++) b[frame + i] += 5.0f;

        CausalConv3d conv = new(weight, null, padT: 1, padH: 1, padW: 1);
        Tensor outA = conv.Forward(backend, inputA);
        Tensor outB = conv.Forward(backend, inputB);
        Assert.Equal(2, (int)outA.Shape[2]);

        // Output frame 0 must be identical (depends only on input frame 0); frame 1 must differ.
        AssertFramesEqual(outA, outB, 0, cOut, h, w, equal: true);
        AssertFramesEqual(outA, outB, 1, cOut, h, w, equal: false);
    }

    // ── helpers ──
    private static Tensor RandomTensor(int[] dims, int seed)
    {
        TensorShape shape = dims.Length switch
        {
            1 => new TensorShape(dims[0]),
            4 => new TensorShape(dims[0], dims[1], dims[2], dims[3]),
            5 => new TensorShape([dims[0], dims[1], dims[2], dims[3], dims[4]]),
            _ => throw new ArgumentException("unsupported rank"),
        };
        Tensor t = new Tensor(shape, DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Clone(Tensor src)
    {
        Tensor t = new Tensor(src.Shape, DType.F32);
        long n = src.Shape.ElementCount;
        Buffer.MemoryCopy((float*)src.DataPointer, (float*)t.DataPointer, n * 4, n * 4);
        return t;
    }

    private static Tensor SliceTemporal(Tensor weight5d, int dt, int cOut, int cIn, int kh, int kw)
    {
        int kt = (int)weight5d.Shape[2];
        Tensor w = new Tensor(new TensorShape(cOut, cIn, kh, kw), DType.F32);
        float* sp = (float*)weight5d.DataPointer;
        float* dp = (float*)w.DataPointer;
        int khw = kh * kw;
        for (int co = 0; co < cOut; co++)
            for (int ci = 0; ci < cIn; ci++)
            {
                long s = (((long)co * cIn + ci) * kt + dt) * khw;
                long d = ((long)co * cIn + ci) * khw;
                Buffer.MemoryCopy(sp + s, dp + d, (long)khw * 4, (long)khw * 4);
            }
        return w;
    }

    private static Tensor ExtractFrame(Tensor input5d, int ti, int cIn, int h, int w)
    {
        int tin = (int)input5d.Shape[2];
        long frame = (long)h * w;
        Tensor f = new Tensor(new TensorShape(1, cIn, h, w), DType.F32);
        float* ip = (float*)input5d.DataPointer;
        float* dp = (float*)f.DataPointer;
        for (int ci = 0; ci < cIn; ci++)
        {
            long s = ((long)ci * tin + ti) * frame;
            Buffer.MemoryCopy(ip + s, dp + (long)ci * frame, frame * 4, frame * 4);
        }
        return f;
    }

    private static void AssertFrameClose(Tensor out5d, int ti, Tensor expected4d, int cOut, int h, int w, float tol)
    {
        int tout = (int)out5d.Shape[2];
        long frame = (long)h * w;
        float* op = (float*)out5d.DataPointer;
        float* ep = (float*)expected4d.DataPointer;
        for (int co = 0; co < cOut; co++)
            for (long i = 0; i < frame; i++)
            {
                float got = op[((long)co * tout + ti) * frame + i];
                float exp = ep[(long)co * frame + i];
                Assert.True(MathF.Abs(got - exp) <= tol, $"ch{co} t{ti} idx{i}: {got} vs {exp}");
            }
    }

    private static void AssertFramesEqual(Tensor a5d, Tensor b5d, int ti, int cOut, int h, int w, bool equal)
    {
        int tout = (int)a5d.Shape[2];
        long frame = (long)h * w;
        float* ap = (float*)a5d.DataPointer;
        float* bp = (float*)b5d.DataPointer;
        float maxDiff = 0f;
        for (int co = 0; co < cOut; co++)
            for (long i = 0; i < frame; i++)
            {
                long off = ((long)co * tout + ti) * frame + i;
                maxDiff = MathF.Max(maxDiff, MathF.Abs(ap[off] - bp[off]));
            }
        if (equal) Assert.True(maxDiff < 1e-6f, $"frame {ti} should be identical, maxDiff={maxDiff}");
        else Assert.True(maxDiff > 1e-3f, $"frame {ti} should differ, maxDiff={maxDiff}");
    }
}
