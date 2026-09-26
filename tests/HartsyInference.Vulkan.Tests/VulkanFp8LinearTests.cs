using HartsyInference.Core.Tensors;
using HartsyInference.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Vulkan.Tests;

/// <summary>The fp8 Linear: the activation quantizes to exactly the bytes CUDA's <c>fp8_quant</c> writes (on any device — the
/// conversion is integer math), and on a device with fp8 cooperative matrices the product equals the dequantized operands'
/// product to F32 accumulation error, stays within E4M3's activation error of the F16-cast path, and falls back unchanged on a
/// N or K off the fragment grid.</summary>
[Trait("Category", "GpuIntegration")]
public sealed class VulkanFp8LinearTests(ITestOutputHelper output)
{
    private readonly ITestOutputHelper _out = output;

    private static bool VulkanAvailable()
    {
        try { using VulkanInstance i = new(); return i.EnumeratePhysicalDevices().Length > 0; }
        catch { return false; }
    }

    /// <summary>block_scale.cuh's f32_to_e4m3, the reference both backends must reproduce.</summary>
    private static byte CudaE4M3(float f)
    {
        byte sign = (byte)((BitConverter.SingleToUInt32Bits(f) >> 24) & 0x80u);
        float a = MathF.Abs(f);
        if (float.IsNaN(a)) return (byte)(sign | 0x7F);
        if (a >= 464f) return (byte)(sign | 0x7E);
        if (a < 0.0009765625f) return sign;
        int e = MathF.ILogB(a) + 1;
        float m = MathF.ScaleB(a, -e);
        if (e - 1 >= -6)
        {
            int q = (int)MathF.Round(m * 16f, MidpointRounding.ToEven);
            if (q == 16) { q = 8; e += 1; }
            int expField = e - 1 + 7;
            if (expField >= 16) return (byte)(sign | 0x7E);
            return (byte)(sign | (expField << 3) | (q - 8));
        }
        int qs = (int)MathF.Round(a * 512f, MidpointRounding.ToEven);
        return qs >= 8 ? (byte)(sign | 0x08) : (byte)(sign | qs);
    }

    private static float DecodeE4M3(byte b)
    {
        int exp = (b >> 3) & 0xF, mant = b & 0x7;
        float v = exp == 0 ? mant * MathF.Pow(2, -9) : (1f + mant / 8f) * MathF.Pow(2, exp - 7);
        return (b & 0x80) != 0 ? -v : v;
    }

    private static float[] RandomValues(int count, int seed, float range)
    {
        Random rng = new(seed);
        float[] v = new float[count];
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rng.NextDouble() * 2 - 1) * range;
        return v;
    }

    private static Tensor F32(TensorShape shape, float[] values)
    {
        Tensor t = new(shape, DType.F32);
        values.CopyTo(t.AsSpan<float>());
        return t;
    }

    /// <summary>An E4M3 weight quantized at <paramref name="scale"/>, the fp8_scaled checkpoint form, plus its dequantized values.</summary>
    private static (Tensor Weight, float[] Dequant) Fp8Weight(int n, int k, int seed, float scale)
    {
        float[] w = RandomValues(n * k, seed, 1f);
        Tensor t = new(new TensorShape(n, k), DType.F8E4M3) { Fp8ScaleFactor = scale };
        Span<byte> bytes = t.AsSpan<byte>();
        float[] dq = new float[n * k];
        for (int i = 0; i < w.Length; i++) { bytes[i] = CudaE4M3(w[i] / scale); dq[i] = DecodeE4M3(bytes[i]) * scale; }
        return (t, dq);
    }

    private float MaxRelError(ReadOnlySpan<float> x, ReadOnlySpan<float> y)
    {
        float maxErr = 0f, maxAbs = 0f;
        for (int i = 0; i < x.Length; i++) { maxErr = MathF.Max(maxErr, MathF.Abs(x[i] - y[i])); maxAbs = MathF.Max(maxAbs, MathF.Abs(y[i])); }
        float rel = maxAbs > 0 ? maxErr / maxAbs : maxErr;
        _out.WriteLine($"maxErr={maxErr:E3} maxAbs={maxAbs:E3} rel={rel:E3}");
        return rel;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Quantize_WritesCudasBytes_AtAbsmaxOver448(bool f16Input)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = VulkanTestDevice.Create();
        // Spans subnormals, ties and the saturation edge once scaled.
        float[] v = RandomValues(64 * 1024, 7, 3f);
        v[5] = 0f; v[6] = -1e-9f; v[7] = 3f;
        using Tensor x = f16Input ? new Tensor(new TensorShape(v.Length), DType.F16) : F32(new TensorShape(v.Length), v);
        if (f16Input)
        {
            Span<Half> h = x.AsSpan<Half>();
            for (int i = 0; i < v.Length; i++) { h[i] = (Half)v[i]; v[i] = (float)h[i]; }
        }

        (byte[] bytes, float scale) = backend.QuantizeE4M3ForTest(x, staticScale: 0f);
        float amax = v.Max(MathF.Abs);
        Assert.Equal(amax / 448f, scale);
        float rs = 1f / scale;
        int mismatches = 0;
        for (int i = 0; i < v.Length; i++)
            if (bytes[i] != CudaE4M3(v[i] * rs)) mismatches++;
        _out.WriteLine($"{mismatches} of {v.Length} bytes differ from the CUDA conversion");
        Assert.Equal(0, mismatches);

        (byte[] staticBytes, _) = backend.QuantizeE4M3ForTest(x, staticScale: 0.004f);
        for (int i = 0; i < v.Length; i++) Assert.Equal(CudaE4M3(v[i] * (1f / 0.004f)), staticBytes[i]);
    }

    [Theory]
    [InlineData(false, false, 128)]
    [InlineData(true, true, 128)]
    [InlineData(false, true, 77)]   // a CLIP prompt's rows: the ragged last block goes through shared memory
    public void Fp8Linear_MatchesTheDequantizedProduct(bool f16Output, bool withBias, int M)
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.Vk.HasFloat8CooperativeMatrix) { _out.WriteLine("SKIPPED: no fp8 cooperative matrices on this device"); return; }
        const int K = 512, N = 192;
        float[] xv = RandomValues(M * K, 1, 2f);
        using Tensor x = F32(new TensorShape(M, K), xv);
        (Tensor w, float[] wDq) = Fp8Weight(N, K, 2, scale: 0.01f);
        using Tensor _w = w;
        using Tensor? bias = withBias ? F32(new TensorShape(N), RandomValues(N, 3, 1f)) : null;
        backend.PreloadWeights([w]);
        DType outType = f16Output ? DType.F16 : DType.F32;

        backend.EnableFp8Linear = true;
        using Tensor got = new(new TensorShape(M, N), outType);
        backend.Linear(got, x, w, bias);
        backend.Sync();

        // Reference: the activation at the bytes the kernel reads, times the dequantized weight, in double.
        float scale = xv.Max(MathF.Abs) / 448f;
        float[] xDq = xv.Select(f => DecodeE4M3(CudaE4M3(f * (1f / scale))) * scale).ToArray();
        float[] want = new float[M * N];
        ReadOnlySpan<float> b = bias is null ? default : bias.AsReadOnlySpan<float>();
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
            {
                double acc = 0;
                for (int t = 0; t < K; t++) acc += (double)xDq[i * K + t] * wDq[j * K + t];
                want[i * N + j] = (float)acc + (bias is null ? 0f : b[j]);
            }
        float[] gotF = f16Output ? got.AsReadOnlySpan<Half>().ToArray().Select(h => (float)h).ToArray() : got.AsReadOnlySpan<float>().ToArray();
        Assert.True(MaxRelError(gotF, want) < (f16Output ? 2e-3f : 1e-5f), "the fp8 product differs from the dequantized operands' product");

        // Against the F16-cast path, the gap is the activation's E4M3 rounding alone.
        backend.EnableFp8Linear = false;
        using Tensor f16Path = new(new TensorShape(M, N), outType);
        backend.Linear(f16Path, x, w, bias);
        backend.Sync();
        float[] refF = f16Output ? f16Path.AsReadOnlySpan<Half>().ToArray().Select(h => (float)h).ToArray() : f16Path.AsReadOnlySpan<float>().ToArray();
        Assert.True(MaxRelError(gotF, refF) < 4e-2f, "the fp8 product is outside E4M3's activation error of the F16-cast path");
    }

    [Fact]
    public void Fp8Linear_UsesTheCheckpointInputScale()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = VulkanTestDevice.Create();
        if (!backend.Vk.HasFloat8CooperativeMatrix) { _out.WriteLine("SKIPPED: no fp8 cooperative matrices on this device"); return; }
        const int M = 64, K = 256, N = 64;
        float[] xv = RandomValues(M * K, 4, 1f);
        using Tensor x = F32(new TensorShape(M, K), xv);
        (Tensor w, _) = Fp8Weight(N, K, 5, scale: 0.02f);
        using Tensor _w = w;
        backend.EnableFp8Linear = true;

        backend.EnableStaticFp8InputScale = false;
        using Tensor dynamic = new(new TensorShape(M, N), DType.F32);
        backend.Linear(dynamic, x, w, null);
        backend.Sync();

        // A static scale equal to the dynamic one must give the same answer bit for bit; a different one must not.
        backend.EnableStaticFp8InputScale = true;
        w.Fp8InputScaleFactor = xv.Max(MathF.Abs) / 448f;
        using Tensor same = new(new TensorShape(M, N), DType.F32);
        backend.Linear(same, x, w, null);
        backend.Sync();
        Assert.Equal(dynamic.AsReadOnlySpan<float>().ToArray(), same.AsReadOnlySpan<float>().ToArray());

        w.Fp8InputScaleFactor *= 4f;
        using Tensor coarser = new(new TensorShape(M, N), DType.F32);
        backend.Linear(coarser, x, w, null);
        backend.Sync();
        Assert.NotEqual(dynamic.AsReadOnlySpan<float>().ToArray(), coarser.AsReadOnlySpan<float>().ToArray());
    }

    [Fact]
    public void Fp8Linear_FallsBackOffTheFragmentGrid()
    {
        if (!VulkanAvailable()) return;
        using VulkanBackend backend = VulkanTestDevice.Create();
        const int M = 32, K = 256, N = 40;   // N off every fragment shape
        using Tensor x = F32(new TensorShape(M, K), RandomValues(M * K, 6, 1f));
        (Tensor w, _) = Fp8Weight(N, K, 7, scale: 0.01f);
        using Tensor _w = w;
        backend.EnableFp8Linear = false;
        using Tensor f16Path = new(new TensorShape(M, N), DType.F32);
        backend.Linear(f16Path, x, w, null);
        backend.Sync();
        backend.EnableFp8Linear = true;
        using Tensor knob = new(new TensorShape(M, N), DType.F32);
        backend.Linear(knob, x, w, null);
        backend.Sync();
        Assert.Equal(f16Path.AsReadOnlySpan<float>().ToArray(), knob.AsReadOnlySpan<float>().ToArray());
    }
}
