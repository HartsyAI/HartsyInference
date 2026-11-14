using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Cuda.Tests;

[Trait("Category", "GpuIntegration")]
[Collection("CudaSerial")]
public sealed class FlashDecodeGqaTests
{
    private readonly ITestOutputHelper _output;

    public FlashDecodeGqaTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void DecodeGqa_RandomizedCasesMatchHostReference()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        int cases = 0;
        float maxError = 0f;
        int[] groups = [1, 2, 3, 4, 8, 9, 16];
        int[] lengths = [64, 65, 127, 257];
        foreach (int d in new[] { 64, 128 })
        foreach (int group in groups)
        foreach (int length in lengths)
        foreach (int mode in new[] { 0, 1, 2 })
        foreach (bool f16 in new[] { false, true })
        foreach (bool device in new[] { false, true })
        {
            float error = RunRandomCase(backend, d, group, length, length + 19, mode, f16, device, batch: 1, seed: cases + 17);
            maxError = Math.Max(maxError, error);
            cases++;
        }

        float batchError = RunRandomCase(backend, 128, 3, 127, 146, mode: 0, f16: true, device: false, batch: 2, seed: 991);
        maxError = Math.Max(maxError, batchError);
        cases++;
        _output.WriteLine($"decode-gqa parity: {cases}/{cases} passed; max absolute error={maxError:E3}");
        Assert.True(maxError < 3e-3f, $"max absolute error {maxError:E3}");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DecodeGqa_FiniteMinimumScoreIsNotEmpty(bool device)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        foreach (bool q8 in new[] { false, true })
        {
            KnobStore.Set(EngineKnobs.QuantAtProducer, q8);
            KnobStore.Set(EngineKnobs.AttnCombineQ8, q8);
            try
            {
                using CudaBackend backend = new(0, PtxDir()) { EnableDp4aGemv = q8 };
                const int d = 64, heads = 8, length = 128;
                using Tensor q = new(new TensorShape(1, heads, 1, d), DType.F32);
                using Tensor k = new(new TensorShape(1, 1, length, d), DType.F32);
                using Tensor v = new(new TensorShape(1, 1, length, d), DType.F32);
                using Tensor actual = new(q.Shape, DType.F32);
                q.AsSpan<float>().Clear();
                k.AsSpan<float>().Clear();
                for (int h = 0; h < heads; h++) q.AsSpan<float>()[h * d] = 1f;
                for (int key = 0; key < length; key++)
                {
                    k.AsSpan<float>()[key * d] = -float.MaxValue;
                    v.AsSpan<float>().Slice(key * d, d).Fill(key % 2 == 0 ? 1f : 3f);
                }
                RunAttention(backend, actual, q, k, v, length, heads, device, scale: 1f, softcap: 0f, window: 0);
                backend.OffloadActivation(actual);
                foreach (float value in actual.AsReadOnlySpan<float>())
                    Assert.True(value == 2f, $"device={device}, q8={q8}: expected 2, got {value}");
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.QuantAtProducer);
                KnobStore.Clear(EngineKnobs.AttnCombineQ8);
            }
        }
    }

    [Fact]
    public void DecodeGqa_QueryAndKeyNanPropagate()
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        foreach (bool qNan in new[] { true, false })
        foreach (bool device in new[] { false, true })
        foreach (bool q8 in new[] { false, true })
        {
            KnobStore.Set(EngineKnobs.QuantAtProducer, q8);
            KnobStore.Set(EngineKnobs.AttnCombineQ8, q8);
            try
            {
                using CudaBackend backend = new(0, PtxDir()) { EnableDp4aGemv = q8 };
                const int d = 128, heads = 8, length = 129;
                using Tensor q = RandomTensor(new TensorShape(1, heads, 1, d), false, new Random(11));
                using Tensor k = RandomTensor(new TensorShape(1, 1, length, d), false, new Random(12));
                using Tensor v = RandomTensor(new TensorShape(1, 1, length, d), false, new Random(13));
                using Tensor actual = new(q.Shape, DType.F32);
                if (qNan)
                {
                    for (int h = 0; h < heads; h++) q.AsSpan<float>()[h * d] = float.NaN;
                }
                else k.AsSpan<float>()[0] = float.NaN;
                RunAttention(backend, actual, q, k, v, length, heads, device, 1f / MathF.Sqrt(d), 0f, 0);
                backend.OffloadActivation(actual);
                Assert.All(actual.AsReadOnlySpan<float>().ToArray(), value => Assert.True(float.IsNaN(value),
                    $"qNan={qNan}, device={device}, q8={q8}: got {value}"));
            }
            finally
            {
                KnobStore.Clear(EngineKnobs.QuantAtProducer);
                KnobStore.Clear(EngineKnobs.AttnCombineQ8);
            }
        }
    }

    [Theory]
    [InlineData(216, 215)]
    [InlineData(200, 200)]
    [InlineData(0, 0)]
    [InlineData(20, -1)]
    public void DecodeGqa_InvalidDevicePositionRaisesAtSynchronization(int kvLen, int qOffset)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        const int d = 64, heads = 8, capacity = 200;
        using Tensor q = RandomTensor(new TensorShape(1, heads, 1, d), false, new Random(21));
        using Tensor k = RandomTensor(new TensorShape(1, 1, capacity, d), false, new Random(22));
        using Tensor v = RandomTensor(new TensorShape(1, 1, capacity, d), false, new Random(23));
        using Tensor actual = new(q.Shape, DType.F32);
        ulong pos = backend.AllocDevicePos();
        try
        {
            backend.WriteDevicePos(pos, kvLen, qOffset);
            backend.FlashAttentionDev(actual, q, k, v, 0, heads, true, 0, 1f / MathF.Sqrt(d), pos, 0f, 16);
            InvalidOperationException error = Assert.Throws<InvalidOperationException>(() => backend.ThrowOnDevicePosError(pos));
            Assert.Contains($"kvLen={kvLen}", error.Message);
            Assert.Contains($"qOffset={qOffset}", error.Message);

            backend.WriteDevicePos(pos, 20, 19);
            backend.FlashAttentionDev(actual, q, k, v, 0, heads, true, 0, 1f / MathF.Sqrt(d), pos, 0f, 16);
            backend.ThrowOnDevicePosError(pos);
        }
        finally
        {
            backend.FreeDevicePos(pos);
        }
    }

    [Theory]
    [InlineData(false, "F.4")]
    [InlineData(true, "V.2")]
    public void DecodeGqa_SelectsGroupedDecodeRoute(bool device, string route)
    {
        if (!CudaContext.IsAvailable()) { _output.WriteLine("SKIPPED: CUDA unavailable"); return; }
        using CudaBackend backend = new(0, PtxDir());
        float error = RunRandomCase(backend, 128, 4, 129, 151, mode: 0, f16: true, device, batch: 1, seed: 44);
        Assert.True(error < 3e-3f);
        Assert.Equal(route, CudaBackend.LastAttentionRoute);
    }

    private static float RunRandomCase(CudaBackend backend, int d, int group, int length, int capacity, int mode,
        bool f16, bool device, int batch, int seed)
    {
        int kvHeads = 2, heads = kvHeads * group;
        float softcap = mode == 2 ? 20f : 0f;
        int window = mode == 1 ? 37 : 0;
        using Tensor q = RandomTensor(new TensorShape(batch, heads, 1, d), false, new Random(seed));
        using Tensor k = RandomTensor(new TensorShape(batch, kvHeads, capacity, d), f16, new Random(seed + 1));
        using Tensor v = RandomTensor(new TensorShape(batch, kvHeads, capacity, d), f16, new Random(seed + 2));
        using Tensor actual = new(q.Shape, DType.F32);
        using Tensor expected = new(q.Shape, DType.F32);
        float scale = 1f / MathF.Sqrt(d);
        RunAttention(backend, actual, q, k, v, length, group, device, scale, softcap, window);
        AttentionCall reference = AttentionCall.FromFlashAttention(expected, q, k, v, length, true, length - 1,
            scale, softcap, null, window, null);
        AttentionHostReference.Run(in reference);
        backend.OffloadActivation(actual);
        float maxError = 0f;
        ReadOnlySpan<float> got = actual.AsReadOnlySpan<float>();
        ReadOnlySpan<float> want = expected.AsReadOnlySpan<float>();
        for (int i = 0; i < got.Length; i++)
        {
            float error = Math.Abs(got[i] - want[i]);
            maxError = Math.Max(maxError, error);
            Assert.True(error <= (f16 ? 3e-3f : 2e-4f),
                $"D={d} group={group} length={length} mode={mode} f16={f16} device={device} batch={batch} index={i}: got={got[i]}, want={want[i]}, error={error}");
        }
        return maxError;
    }

    private static void RunAttention(CudaBackend backend, Tensor output, Tensor q, Tensor k, Tensor v, int length,
        int group, bool device, float scale, float softcap, int window)
    {
        if (!device)
        {
            backend.FlashAttention(output, q, k, v, length, group, true, length - 1, scale, softcap, null, window, null);
            return;
        }
        ulong pos = backend.AllocDevicePos();
        try
        {
            backend.WriteDevicePos(pos, length, length - 1);
            backend.FlashAttentionDev(output, q, k, v, 0, group, true, 0, scale, pos, softcap, window);
            backend.ThrowOnDevicePosError(pos);
        }
        finally
        {
            backend.FreeDevicePos(pos);
        }
    }

    private static Tensor RandomTensor(TensorShape shape, bool f16, Random rng)
    {
        Tensor tensor = new(shape, f16 ? DType.F16 : DType.F32);
        if (f16)
        {
            Span<Half> values = tensor.AsSpan<Half>();
            for (int i = 0; i < values.Length; i++) values[i] = (Half)((rng.NextDouble() - 0.5) * 0.4);
        }
        else
        {
            Span<float> values = tensor.AsSpan<float>();
            for (int i = 0; i < values.Length; i++) values[i] = (float)((rng.NextDouble() - 0.5) * 0.4);
        }
        return tensor;
    }

    private static string PtxDir()
    {
        string output = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (Directory.Exists(output)) return output;
        for (DirectoryInfo? dir = new(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "src", "HartsyInference.Cuda", "Ptx");
            if (Directory.Exists(candidate)) return candidate;
        }
        throw new DirectoryNotFoundException("src/HartsyInference.Cuda/Ptx not found above the test output directory.");
    }
}
