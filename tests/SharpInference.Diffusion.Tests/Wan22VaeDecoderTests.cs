using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Cpu;
using SharpInference.Diffusion.Models.Vae;
using SharpInference.Tests.Common;

namespace SharpInference.Diffusion.Tests;

/// <summary>End-to-end structural test for <see cref="Wan22VaeDecoder"/>: a tiny synthetic-weight decoder runs the full image decode path on the CPU backend and must produce the right RGB shape (16× spatial) with finite values. This verifies the entire wiring — key names, block composition through 4 up-stages + DupUp3D shortcuts + head + unpatchify. (Numeric correctness vs the real Wan2.2 checkpoint is a separate validation-pending item — no weights in this environment.)</summary>
public unsafe class Wan22VaeDecoderTests
{
    [Fact]
    public void Decode_ImagePath_Produces16xRgbAndIsFinite()
    {
        CpuBackend backend = new();
        int dim = 8, zDim = 48;
        int[] dimMult = [1, 2, 4, 4];
        int numResBlocks = 2;
        bool[] tUp = [false, true, true];

        Dictionary<string, Tensor> w = LanceSyntheticWeights.BuildVae(dim, zDim, dimMult, numResBlocks, tUp);
        Wan22VaeDecoder decoder = new(dim, zDim, dimMult, numResBlocks, tUp);
        decoder.LoadWeights(w);

        int latH = 2, latW = 2;
        Tensor latent = Rand([1, zDim, 1, latH, latW]);
        Tensor rgb = decoder.Decode(backend, latent);

        Assert.Equal(1, (int)rgb.Shape[0]);
        Assert.Equal(3, (int)rgb.Shape[1]);          // RGB
        Assert.Equal(1, (int)rgb.Shape[2]);          // single frame
        Assert.Equal(latH * 16, (int)rgb.Shape[3]);  // 8× (3 upsample stages) × 2× (unpatchify)
        Assert.Equal(latW * 16, (int)rgb.Shape[4]);

        float* p = (float*)rgb.DataPointer;
        for (long i = 0; i < rgb.Shape.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void Decode_VideoPath_ExpandsTemporallyAndIsFinite()
    {
        CpuBackend backend = new();
        int dim = 8, zDim = 48;
        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder decoder = new(dim, zDim, dimMult, 2, tUp);
        decoder.LoadWeights(LanceSyntheticWeights.BuildVae(dim, zDim, dimMult, 2, tUp));

        int tLat = 2;
        Tensor latent = Rand([1, zDim, tLat, 2, 2]);
        Tensor rgb = decoder.Decode(backend, latent);

        Assert.Equal(3, (int)rgb.Shape[1]);
        Assert.Equal((tLat - 1) * 4 + 1, (int)rgb.Shape[2]);  // 4× temporal upsample + 1 (= 5)
        Assert.Equal(2 * 16, (int)rgb.Shape[3]);
        Assert.Equal(2 * 16, (int)rgb.Shape[4]);
        float* p = (float*)rgb.DataPointer;
        for (long i = 0; i < rgb.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void Decode_VideoFirstFrame_EqualsImageDecode()
    {
        // The streaming first chunk (fresh all-None cache) must compute frame 0 identically to the
        // stateless image path — proving the feat_cache machinery is consistent.
        CpuBackend backend = new();
        int dim = 8, zDim = 48;
        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder decoder = new(dim, zDim, dimMult, 2, tUp);
        decoder.LoadWeights(LanceSyntheticWeights.BuildVae(dim, zDim, dimMult, 2, tUp));

        Tensor latent2 = Rand([1, zDim, 2, 2, 2]);
        Tensor latent1 = Vae3dLayout.SliceFrames(latent2, 0, 1);   // first latent frame only

        Tensor img = decoder.Decode(backend, latent1);    // [1,3,1,32,32]
        Tensor vid = decoder.Decode(backend, latent2);    // [1,3,5,32,32]

        // Compare output frame 0.
        int c = 3, h = 32, w = 32;
        long frame = (long)h * w;
        int vidT = (int)vid.Shape[2];
        float* ip = (float*)img.DataPointer;   // [1,3,1,32,32]
        float* vp = (float*)vid.DataPointer;   // [1,3,5,32,32]
        for (int ci = 0; ci < c; ci++)
            for (long i = 0; i < frame; i++)
            {
                float a = ip[(long)ci * 1 * frame + i];
                float b = vp[((long)ci * vidT + 0) * frame + i];
                Assert.True(MathF.Abs(a - b) < 1e-5f, $"ch{ci} idx{i}: image {a} vs video[0] {b}");
            }
    }

    [Fact]
    public void Decode_StreamingMatchesFullClip()
    {
        // The pull-based DecodeStreaming (one frame-group at a time) must produce byte-identical output to the
        // all-at-once Decode — a regression gate for the streaming-cache refactor.
        CpuBackend backend = new();
        int dim = 8, zDim = 48;
        int[] dimMult = [1, 2, 4, 4];
        bool[] tUp = [false, true, true];
        Wan22VaeDecoder decoder = new(dim, zDim, dimMult, 2, tUp);
        decoder.LoadWeights(LanceSyntheticWeights.BuildVae(dim, zDim, dimMult, 2, tUp));

        Tensor latent = Rand([1, zDim, 2, 2, 2]);
        Tensor full = decoder.Decode(backend, latent);

        List<Tensor> groups = new();
        foreach (Tensor g in decoder.DecodeStreaming(backend, latent)) groups.Add(g);
        Tensor streamed = Vae3dLayout.ConcatFrames(groups);
        foreach (Tensor g in groups) g.Dispose();

        Assert.Equal(full.Shape.ElementCount, streamed.Shape.ElementCount);
        float* a = (float*)full.DataPointer;
        float* b = (float*)streamed.DataPointer;
        for (long i = 0; i < full.Shape.ElementCount; i++)
            Assert.True(MathF.Abs(a[i] - b[i]) < 1e-6f, $"idx {i}: full {a[i]} vs streamed {b[i]}");
    }

    private static int _seed = 100;

    private static Tensor Rand(long[] dims)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        // Small magnitudes keep the deep stack numerically tame (this is a structural/finiteness test).
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return t;
    }
}
