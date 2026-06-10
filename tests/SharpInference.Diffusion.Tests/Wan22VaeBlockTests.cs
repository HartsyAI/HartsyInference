using Xunit;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Vae;

namespace SharpInference.Diffusion.Tests;

/// <summary>Correctness tests for the reusable Wan2.2 VAE blocks <see cref="DupUp3D"/> (temporal/spatial dup-upsample) and <see cref="WanRmsNorm"/> (channel RMS), no GPU/checkpoint.</summary>
public unsafe class Wan22VaeBlockTests
{
    [Fact]
    public void DupUp3D_ShapeAndFirstChunk()
    {
        Tensor x = Random([1, 2, 1, 2, 2], seed: 1);
        Tensor full = DupUp3D.Forward(x, outChannels: 2, factorT: 2, factorS: 2, firstChunk: false);
        Assert.Equal(2, (int)full.Shape[2]); // T·2
        Assert.Equal(4, (int)full.Shape[3]); // H·2
        Assert.Equal(4, (int)full.Shape[4]); // W·2

        Tensor first = DupUp3D.Forward(x, outChannels: 2, factorT: 2, factorS: 2, firstChunk: true);
        Assert.Equal(1, (int)first.Shape[2]); // drops factorT-1 lead frames → single frame
    }

    [Fact]
    public void DupUp3D_IsChannelPreservingNearestUpsample()
    {
        // inC=outC=2, factor=factorT·factorS²=8, repeats=8 → output channel oc draws entirely from input channel oc
        // (a nearest-style temporal+spatial duplication that preserves channels).
        Tensor x = new Tensor(new TensorShape([1L, 2, 1, 2, 2]), DType.F32);
        float* xp = (float*)x.DataPointer;
        for (int i = 0; i < 4; i++) xp[i] = 1f;        // channel 0 = 1
        for (int i = 4; i < 8; i++) xp[i] = 2f;        // channel 1 = 2
        Tensor outT = DupUp3D.Forward(x, 2, factorT: 2, factorS: 2, firstChunk: false); // [1,2,2,4,4]
        float* op = (float*)outT.DataPointer;

        long perChannel = 2L * 4 * 4; // keepT·H·W
        for (long i = 0; i < perChannel; i++) Assert.Equal(1f, op[i]);                 // channel 0 → all 1
        for (long i = 0; i < perChannel; i++) Assert.Equal(2f, op[perChannel + i]);    // channel 1 → all 2
    }

    [Fact]
    public void WanRmsNorm_NormalizesChannelVectorToUnitRms()
    {
        // C=2, gamma=1: a per-position channel vector [3,4] → rms-normalized so per-position rms == 1.
        WanRmsNorm norm = new(channels: 2);
        Tensor gamma = Ones(2);
        norm.LoadWeights(gamma);

        Tensor x = new Tensor(new TensorShape([1L, 2, 1, 1, 1]), DType.F32);
        float* xp = (float*)x.DataPointer; xp[0] = 3f; xp[1] = 4f;
        Tensor outT = norm.Forward(x);
        float* op = (float*)outT.DataPointer;

        float rms = MathF.Sqrt((op[0] * op[0] + op[1] * op[1]) / 2f);
        Assert.Equal(1.0f, rms, 4);
        // Direction preserved (3:4 ratio).
        Assert.Equal(3f / 4f, op[0] / op[1], 4);
    }

    [Fact]
    public void WanRmsNorm_AppliesPerChannelGamma()
    {
        WanRmsNorm norm = new(channels: 2);
        Tensor gamma = new Tensor(new TensorShape(2), DType.F32);
        float* gp = (float*)gamma.DataPointer; gp[0] = 2f; gp[1] = 0.5f;
        norm.LoadWeights(gamma);

        Tensor x = new Tensor(new TensorShape([1L, 2, 1, 1, 1]), DType.F32);
        float* xp = (float*)x.DataPointer; xp[0] = 1f; xp[1] = 1f; // equal → both normalize to sqrt(C-mean)=1 before gamma
        Tensor outT = norm.Forward(x);
        float* op = (float*)outT.DataPointer;
        // x=[1,1] → rms=1 → normalized=[1,1] → ·gamma=[2,0.5].
        Assert.Equal(2.0f, op[0], 4);
        Assert.Equal(0.5f, op[1], 4);
    }

    [Theory]
    [InlineData(2, 4)]   // in != out → conv shortcut
    [InlineData(4, 4)]   // in == out → identity shortcut
    public void ResidualBlock_PreservesSpatialShapeAndIsFinite(int inDim, int outDim)
    {
        SharpInference.Cpu.CpuBackend backend = new();
        int t = 1, h = 4, w = 4;
        Dictionary<string, Tensor> weights = new()
        {
            ["rb.residual.0.gamma"] = RandShape([inDim], 1),
            ["rb.residual.2.weight"] = RandShape([outDim, inDim, 3, 3, 3], 2),
            ["rb.residual.2.bias"] = RandShape([outDim], 3),
            ["rb.residual.3.gamma"] = RandShape([outDim], 4),
            ["rb.residual.6.weight"] = RandShape([outDim, outDim, 3, 3, 3], 5),
            ["rb.residual.6.bias"] = RandShape([outDim], 6),
        };
        if (inDim != outDim)
        {
            weights["rb.shortcut.weight"] = RandShape([outDim, inDim, 1, 1, 1], 7);
            weights["rb.shortcut.bias"] = RandShape([outDim], 8);
        }

        Wan22ResidualBlock block = new(inDim, outDim);
        block.LoadWeights(weights, "rb");
        Tensor x = RandShape([1, inDim, t, h, w], 9);
        Tensor outT = block.Forward(backend, x);

        Assert.Equal(outDim, (int)outT.Shape[1]);
        Assert.Equal(t, (int)outT.Shape[2]);
        Assert.Equal(h, (int)outT.Shape[3]);
        Assert.Equal(w, (int)outT.Shape[4]);
        float* p = (float*)outT.DataPointer;
        for (long i = 0; i < outT.Shape.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    [Fact]
    public void AttentionBlock_PreservesShapeAndIsFinite()
    {
        SharpInference.Cpu.CpuBackend backend = new();
        int dim = 4, h = 4, w = 4;
        Dictionary<string, Tensor> weights = new()
        {
            ["at.norm.gamma"] = RandShape([dim], 1),
            ["at.to_qkv.weight"] = RandShape([3 * dim, dim, 1, 1], 2),
            ["at.to_qkv.bias"] = RandShape([3 * dim], 3),
            ["at.proj.weight"] = RandShape([dim, dim, 1, 1], 4),
            ["at.proj.bias"] = RandShape([dim], 5),
        };
        Wan22AttentionBlock block = new(dim);
        block.LoadWeights(weights, "at");
        Tensor x = RandShape([1, dim, 1, h, w], 6);
        Tensor outT = block.Forward(backend, x);

        Assert.Equal(new long[] { 1, dim, 1, h, w }, ShapeOf(outT));
        AssertFinite(outT);
    }

    [Fact]
    public void Resample_DoublesSpatialResolution()
    {
        SharpInference.Cpu.CpuBackend backend = new();
        int dim = 4, h = 3, w = 5;
        Dictionary<string, Tensor> weights = new()
        {
            ["rs.resample.1.weight"] = RandShape([dim, dim, 3, 3], 1),
            ["rs.resample.1.bias"] = RandShape([dim], 2),
        };
        Wan22Resample rs = new(dim);
        rs.LoadWeights(weights, "rs");
        Tensor x = RandShape([1, dim, 1, h, w], 3);
        Tensor outT = rs.Forward(backend, x);

        Assert.Equal(new long[] { 1, dim, 1, h * 2, w * 2 }, ShapeOf(outT));
        AssertFinite(outT);
    }

    private static long[] ShapeOf(Tensor t)
    {
        long[] s = new long[t.Shape.Rank];
        for (int i = 0; i < s.Length; i++) s[i] = t.Shape[i];
        return s;
    }

    private static void AssertFinite(Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.Shape.ElementCount; i++)
            Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Tensor RandShape(long[] dims, int seed)
    {
        Tensor t = new Tensor(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Random(int[] dims, int seed)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (int i = 0; i < n; i++) p[i] = 1f;
        return t;
    }
}
