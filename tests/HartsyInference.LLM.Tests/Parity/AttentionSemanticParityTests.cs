using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Vulkan;
using Xunit;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>Contract-negative controls prove score ordering and zero-output exclusions on executed CUDA rows.</summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class AttentionSemanticParityTests
{
    [CudaTheory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void FlashSoftcapPrecedesAlibi(bool halfKv, bool causal)
    {
        using IDisposable policy = Policy(false).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 1, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 64), halfKv ? DType.F16 : DType.F32);
        using Tensor v = new(k.Shape, k.DType);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor slopes = new(new TensorShape(1), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 2; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, key == 0 ? 2 : 10);
        slopes.AsSpan<float>()[0] = 4;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        backend.FlashAttention(output, q, k, v, 2, 1, causal, 1, 1, softcap: 1, alibiSlopes: slopes);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Implementation == "flash.decode");
        double expected = (2 * Math.Exp(-4) + 10) / (Math.Exp(-4) + 1);
        foreach (float actual in output.AsReadOnlySpan<float>())
            Assert.InRange(Math.Abs(actual - expected), 0, 2e-5 * expected);
    }

    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f16", false)]
    [InlineData("bf16", false)]
    [InlineData("f16", true)]
    [InlineData("bf16", true)]
    public void FullyExcludedRowsAreZero(string dtype, bool cudnn)
    {
        using IDisposable policy = Policy(cudnn).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        using Tensor q = new(new TensorShape(1, 1, 2, 64), storage);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(1, 1, 2, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 3; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        bias.AsSpan<float>().Fill(float.NegativeInfinity);
        bias.AsSpan<float>()[4] = 0;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        RunDense(backend, output, q, k, v, bias, cudnn);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == (cudnn ? "cudnn.f16" : "materialized"));
        for (int c = 0; c < 64; c++)
        {
            Assert.Equal(0, Read(output, c));
            Assert.Equal(13, Read(output, 64 + c));
        }
    }

    [CudaTheory]
    [InlineData(false, float.NegativeInfinity, false)]
    [InlineData(true, float.NegativeInfinity, false)]
    [InlineData(false, float.PositiveInfinity, true)]
    [InlineData(true, float.PositiveInfinity, true)]
    [InlineData(false, float.NaN, true)]
    [InlineData(true, float.NaN, true)]
    public void EmptyFlashRowsPreserveSinkSemantics(bool halfKv, float sinkValue, bool expectNaN)
    {
        using IDisposable policy = Policy(false).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 1, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 2, 64), halfKv ? DType.F16 : DType.F32);
        using Tensor v = new(k.Shape, k.DType);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor sink = new(new TensorShape(1), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        Fill(v, 7);
        sink.AsSpan<float>()[0] = sinkValue;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        // A noncausal local window beyond the valid prefix is empty, while KV storage remains valid.
        backend.FlashAttention(output, q, k, v, 2, 1, false, 3, 1, sink: sink, slidingWindow: 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Implementation == "flash.decode");
        foreach (float actual in output.AsReadOnlySpan<float>())
            if (expectNaN) Assert.True(float.IsNaN(actual)); else Assert.Equal(0, actual);
    }

    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f16", false)]
    [InlineData("bf16", false)]
    [InlineData("f16", true)]
    [InlineData("bf16", true)]
    public void BiasRetainsF32RangeAndPropagatesNonFiniteScores(string dtype, bool cudnn)
    {
        using IDisposable policy = Policy(cudnn).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        CheckBias(backend, storage, cudnn, "cuda", cudnn ? "cudnn.f16" : "materialized");
    }

    /// <summary>The tiled kernel takes this F32 call by default and the scalar kernel with the knob off: both keep the control.</summary>
    [VulkanTheory]
    [InlineData(VulkanTiledAttention.Auto, "tiled.f32")]
    [InlineData(VulkanTiledAttention.Off, "flash.scalar")]
    public void VulkanBiasRetainsF32RangeAndPropagatesNonFiniteScores(VulkanTiledAttention tiled, string implementation)
    {
        using IBackend backend = new VulkanBackend { TiledAttention = tiled };
        CheckBias(backend, DType.F32, false, "vulkan", implementation);
    }

    [VulkanTheory]
    [InlineData(VulkanTiledAttention.Auto, "tiled.f32")]
    [InlineData(VulkanTiledAttention.Off, "flash.scalar")]
    public void VulkanFullyExcludedRowsAreZero(VulkanTiledAttention tiled, string implementation)
    {
        using IBackend backend = new VulkanBackend { TiledAttention = tiled };
        using Tensor q = new(new TensorShape(1, 1, 2, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), DType.F32);
        using Tensor v = new(k.Shape, DType.F32);
        using Tensor output = new(q.Shape, DType.F32);
        using Tensor bias = new(new TensorShape(2, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 3; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        bias.AsSpan<float>().Fill(float.NegativeInfinity);
        bias.AsSpan<float>()[4] = 0;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        backend.ScaledDotProductAttention(output, q, k, v, bias, 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "vulkan" && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == implementation);
        for (int c = 0; c < 64; c++)
        {
            Assert.Equal(0, Read(output, c));
            Assert.Equal(13, Read(output, 64 + c));
        }
    }

    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f16", false)]
    [InlineData("bf16", false)]
    [InlineData("f16", true)]
    public void ExclusionsRespectBatchHeadAndReductionTails(string dtype, bool cudnn)
    {
        using IDisposable policy = Policy(cudnn).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        const int heads = 2, keys = 516, rows = 3, dim = 64;
        using Tensor q = new(new TensorShape(2, heads, rows, dim), storage);
        using Tensor k = new(new TensorShape(2, heads, keys, dim), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(2, heads, rows, keys), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        bias.AsSpan<float>().Fill(float.NegativeInfinity);
        for (int bh = 0; bh < 4; bh++)
        {
            for (int key = 0; key < keys; key++)
                for (int c = 0; c < dim; c++) Write(v, (bh * keys + key) * dim + c, key % 7 + bh * 2);
            bias.AsSpan<float>()[(bh * rows + 1) * keys + bh * 17] = 0;
            bias.AsSpan<float>()[(bh * rows + 2) * keys + keys - 1] = 0;
        }
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        RunDense(backend, output, q, k, v, bias, cudnn);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == (cudnn ? "cudnn.f16" : "materialized"));
        for (int bh = 0; bh < 4; bh++)
            for (int c = 0; c < dim; c++)
            {
                Assert.Equal(0, Read(output, (bh * rows) * dim + c));
                Assert.Equal(bh * 17 % 7 + bh * 2, Read(output, (bh * rows + 1) * dim + c));
                Assert.Equal((keys - 1) % 7 + bh * 2, Read(output, (bh * rows + 2) * dim + c));
            }
    }

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void SplitFlashPropagatesNaN(bool halfKv)
    {
        using IDisposable policy = Policy(false).With(EngineKnobs.FlashSplitForce, true)
            .With(EngineKnobs.FlashDecodeGqa, false).With(EngineKnobs.FlashSplitOff, false).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        using Tensor q = new(new TensorShape(1, 1, 1, 64), DType.F32);
        using Tensor k = new(new TensorShape(1, 1, 64, 64), halfKv ? DType.F16 : DType.F32);
        using Tensor v = new(k.Shape, k.DType);
        using Tensor output = new(q.Shape, DType.F32);
        Fill(q, 1);
        Fill(k, 1);
        Fill(v, 7);
        q.AsSpan<float>()[0] = float.NaN;
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        backend.FlashAttention(output, q, k, v, 64, 1, true, 63, 1, softcap: 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Branch == "F.5");
        foreach (float actual in output.AsReadOnlySpan<float>()) Assert.True(float.IsNaN(actual));
    }

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeHalfPrefillPropagatesNaN(bool causal)
    {
        using IDisposable policy = Policy(false).With(EngineKnobs.FlashF16, true)
            .With(EngineKnobs.Fa2Prefill, true).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        int dim = causal ? 64 : 256;
        using Tensor q = new(new TensorShape(1, 1, 64, dim), causal ? DType.F32 : DType.F16);
        using Tensor k = new(new TensorShape(1, 1, 64, dim), DType.F16);
        using Tensor v = new(k.Shape, DType.F16);
        using Tensor output = new(q.Shape, q.DType);
        Fill(q, 0);
        Fill(k, 0);
        Fill(v, 7);
        Write(q, 0, float.NaN);
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        if (causal) backend.FlashAttention(output, q, k, v, 64, 1, true, 0, 1);
        else backend.ScaledDotProductAttention(output, q, k, v, null, 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Branch == (causal ? "F.3" : "S.2"));
        for (int c = 0; c < dim; c++) Assert.True(float.IsNaN(Read(output, c)));
        for (int c = dim; c < output.ElementCount; c++)
            Assert.InRange(Math.Abs(Read(output, c) - 7), 0, 3e-3 * 7);
    }

    /// <summary>The attention softmax replaced the legacy softmax (and separate F32 mask add) on the materialized and
    /// query-tiled rows. Finite rows must stay bit-identical to it, so a request whose attention never excludes a whole
    /// row (Krea 2's strict-F32 text encoder, with its finite <c>-1e30</c> causal sentinel) keeps its exact output.</summary>
    [CudaTheory]
    [InlineData("f32", false)]
    [InlineData("f32", true)]
    [InlineData("f16", false)]
    public unsafe void AttentionSoftmaxReproducesLegacySoftmaxBitsOnFiniteRows(string dtype, bool masked)
    {
        const int rows = 37, columns = 1031;
        bool half = dtype == "f16";
        int elementSize = half ? 2 : 4, count = rows * columns;
        using CudaContext context = new();
        using CudaStream stream = new();
        using CudaKernels kernels = new(CudaPtxDirectory.Resolve(), context);
        Random random = new(20261116);
        float[] scores = new float[count], bias = new float[count];
        for (int i = 0; i < count; i++)
        {
            scores[i] = (float)(random.NextDouble() * 24 - 12);
            int row = i / columns, column = i % columns;
            bias[i] = column > row * 28 ? -1e30f : (float)(random.NextDouble() - 0.5);
        }
        Half[] halves = Array.ConvertAll(scores, x => (Half)x);
        ulong legacy = 0, current = 0, mask = 0;
        try
        {
            legacy = CudaMemory.AllocatePersistent((nuint)(count * elementSize));
            current = CudaMemory.AllocatePersistent((nuint)(count * elementSize));
            mask = CudaMemory.AllocatePersistent((nuint)(count * 4));
            if (half)
                fixed (Half* source = halves)
                {
                    CudaMemory.CopyHostToDeviceAsync(legacy, source, (nuint)(count * 2), stream.Handle);
                    CudaMemory.CopyHostToDeviceAsync(current, source, (nuint)(count * 2), stream.Handle);
                }
            else
                fixed (float* source = scores)
                {
                    CudaMemory.CopyHostToDeviceAsync(legacy, source, (nuint)(count * 4), stream.Handle);
                    CudaMemory.CopyHostToDeviceAsync(current, source, (nuint)(count * 4), stream.Handle);
                }
            fixed (float* source = bias) CudaMemory.CopyHostToDeviceAsync(mask, source, (nuint)(count * 4), stream.Handle);

            if (masked) kernels.LaunchAdd(legacy, legacy, mask, count, stream.Handle);
            if (half) kernels.LaunchSoftmaxF16(legacy, columns, rows, stream.Handle);
            else kernels.LaunchSoftmax(legacy, columns, rows, stream.Handle);
            kernels.LaunchAttentionSoftmax(current, masked ? mask : 0, columns, rows, half, stream.Handle);

            byte[] expected = new byte[count * elementSize], actual = new byte[count * elementSize];
            fixed (byte* target = expected) CudaMemory.CopyDeviceToHostAsync(target, legacy, (nuint)expected.Length, stream.Handle);
            fixed (byte* target = actual) CudaMemory.CopyDeviceToHostAsync(target, current, (nuint)actual.Length, stream.Handle);
            stream.Synchronize();
            int first = expected.AsSpan().SequenceCompareTo(actual) == 0 ? -1 : Mismatch(expected, actual) / elementSize;
            Assert.True(first < 0, $"First differing element {first} (row {first / columns}, column {first % columns}).");
        }
        finally
        {
            stream.Synchronize();
            foreach (ulong buffer in (ReadOnlySpan<ulong>)[legacy, current, mask]) if (buffer != 0) CudaMemory.Free(buffer);
        }

        static int Mismatch(byte[] a, byte[] b)
        {
            for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return i;
            return -1;
        }
    }

    [CudaTheory]
    [InlineData(false, "nan")]
    [InlineData(true, "nan")]
    [InlineData(false, "empty")]
    [InlineData(true, "empty")]
    [InlineData(false, "finite")]
    [InlineData(true, "finite")]
    [InlineData(false, "value-nan")]
    [InlineData(true, "value-nan")]
    public unsafe void CombinePreservesNormalizationAndSidecar(bool fused, string state)
    {
        using CudaContext context = new();
        using CudaStream stream = new();
        using CudaKernels kernels = new(CudaPtxDirectory.Resolve(), context);
        ulong[] buffers = new ulong[7];
        int[] elements = [64, 64, 2, 2, 2, 2, 128];
        try
        {
            for (int i = 0; i < buffers.Length; i++) buffers[i] = CudaMemory.AllocatePersistent((nuint)(elements[i] * 4));
            Span<float> maxima = stackalloc float[2];
            Span<float> sums = stackalloc float[2];
            Span<float> accumulators = stackalloc float[128];
            maxima.Fill(state == "empty" ? -float.MaxValue : 0);
            sums.Fill(state == "empty" ? 0 : state == "nan" ? float.NaN : 1);
            accumulators.Fill(state == "empty" ? 0 : 6);
            if (state == "value-nan") accumulators[5] = float.NaN;
            fixed (float* source = maxima) CudaMemory.CopyHostToDeviceAsync(buffers[4], source, 8, stream.Handle);
            fixed (float* source = sums) CudaMemory.CopyHostToDeviceAsync(buffers[5], source, 8, stream.Handle);
            fixed (float* source = accumulators) CudaMemory.CopyHostToDeviceAsync(buffers[6], source, 512, stream.Handle);
            if (fused)
                kernels.LaunchFlashAttentionCombineQ8(buffers[0], buffers[1], buffers[2], buffers[3], buffers[4],
                    buffers[5], buffers[6], 1, 1, 1, 64, 2, stream.Handle);
            else
                kernels.LaunchFlashAttentionCombine(buffers[0], buffers[4], buffers[5], buffers[6], 1, 1, 1, 64, 2, stream.Handle);
            Span<float> actual = stackalloc float[64];
            Span<float> scales = stackalloc float[2];
            fixed (float* target = actual) CudaMemory.CopyDeviceToHostAsync(target, buffers[0], 256, stream.Handle);
            if (fused)
                fixed (float* target = scales) CudaMemory.CopyDeviceToHostAsync(target, buffers[2], 8, stream.Handle);
            stream.Synchronize();
            for (int c = 0; c < actual.Length; c++)
                if (state == "nan" || state == "value-nan" && c == 5) Assert.True(float.IsNaN(actual[c]));
                else Assert.Equal(state == "empty" ? 0 : 6, actual[c]);
            if (fused)
                for (int block = 0; block < scales.Length; block++)
                    if (state == "nan" || state == "value-nan" && block == 0) Assert.True(float.IsNaN(scales[block]));
                    else Assert.True(float.IsFinite(scales[block]));
        }
        finally
        {
            stream.Synchronize();
            foreach (ulong buffer in buffers) if (buffer != 0) CudaMemory.Free(buffer);
        }
    }

    [CudaTheory]
    [InlineData("f32", -1e30f)]
    [InlineData("f16", -1e30f)]
    [InlineData("bf16", -1e30f)]
    [InlineData("f16", -float.MaxValue)]
    public void FiniteSentinelRetainsF32AdditionSemantics(string dtype, float sentinel)
    {
        CheckNonzeroScores(dtype, sentinel, 13);
    }

    [CudaTheory]
    [InlineData("f32")]
    [InlineData("f16")]
    [InlineData("bf16")]
    public void NonzeroScoresAndFiniteBiasCompose(string dtype)
    {
        double expected = (7 * Math.Exp(-2) + 13 * Math.Exp(-1) + 19) / (Math.Exp(-2) + Math.Exp(-1) + 1);
        CheckNonzeroScores(dtype, 0, expected);
    }

    private static void CheckBias(IBackend backend, DType storage, bool cudnn, string site, string implementation)
    {
        using Tensor q = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(3, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        for (int key = 0; key < 3; key++)
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        float[] values = [100_000, 100_001, float.NegativeInfinity, 0, float.NaN, float.NegativeInfinity,
            0, float.PositiveInfinity, float.NegativeInfinity];
        values.CopyTo(bias.AsSpan<float>());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        RunDense(backend, output, q, k, v, bias, cudnn);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == site && e.Kind == AttentionEvidenceKind.Execution
            && e.Implementation == implementation);
        double expected = (7 + 13 * Math.E) / (1 + Math.E);
        double tolerance = storage == DType.F32 ? 2e-5 * expected : storage == DType.F16 ? 0.02 : 0.1;
        for (int c = 0; c < 64; c++)
        {
            Assert.InRange(Math.Abs(Read(output, c) - expected), 0, tolerance);
            Assert.True(float.IsNaN(Read(output, 64 + c)));
            Assert.True(float.IsNaN(Read(output, 128 + c)));
        }
    }

    private static KnobProfile Policy(bool cudnn) => KnobProfile.Create("semantic-tests")
        .With(EngineKnobs.SdpaCudnn, cudnn).With(EngineKnobs.SageAttn, false)
        .With(EngineKnobs.FlashF16, false).With(EngineKnobs.SdpaForceTiled, false)
        .With(EngineKnobs.SdpaNoTf32, true);

    private static void CheckNonzeroScores(string dtype, float sentinel, double expected)
    {
        using IDisposable policy = Policy(false).Push();
        using IBackend backend = new CudaBackend(0, CudaPtxDirectory.Resolve());
        DType storage = dtype == "f16" ? DType.F16 : dtype == "bf16" ? DType.BF16 : DType.F32;
        using Tensor q = new(new TensorShape(1, 1, 1, 64), storage);
        using Tensor k = new(new TensorShape(1, 1, 3, 64), storage);
        using Tensor v = new(k.Shape, storage);
        using Tensor output = new(q.Shape, storage);
        using Tensor bias = new(new TensorShape(1, 3), DType.F32);
        Fill(q, 0);
        Fill(k, 0);
        Write(q, 0, 1);
        for (int key = 0; key < 3; key++)
        {
            Write(k, key * 64, sentinel == 0 ? key * 2 : key * 64);
            bias.AsSpan<float>()[key] = sentinel - key;
            for (int c = 0; c < 64; c++) Write(v, key * 64 + c, 7 + key * 6);
        }
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(32);
        backend.ScaledDotProductAttention(output, q, k, v, bias, 1);
        backend.Sync();
        Assert.Contains(evidence.Snapshot(), e => e.Site == "cuda" && e.Implementation == "materialized");
        double relativeTolerance = storage == DType.F32 ? 2e-5 : storage == DType.F16 ? 3e-3 : 8e-3;
        for (int c = 0; c < 64; c++)
            Assert.InRange(Math.Abs(Read(output, c) - expected), 0, relativeTolerance * expected);
    }

    private static void RunDense(IBackend backend, Tensor output, Tensor q, Tensor k, Tensor v, Tensor bias, bool cudnn)
    {
        if (cudnn && q.DType == DType.BF16)
            backend.Attention(AttentionCall.FromScaledDotProductAttention(output, q, k, v, bias, 1, false)
                with { Numerics = AttentionNumerics.Bf16Storage });
        else
            backend.ScaledDotProductAttention(output, q, k, v, bias, 1);
    }

    private static void Fill(Tensor tensor, float value)
    {
        for (int i = 0; i < tensor.ElementCount; i++) Write(tensor, i, value);
    }

    private static void Write(Tensor tensor, int index, float value)
    {
        if (tensor.DType == DType.F32) tensor.AsSpan<float>()[index] = value;
        else if (tensor.DType == DType.F16) tensor.AsSpan<Half>()[index] = (Half)value;
        else tensor.AsSpan<ushort>()[index] = (ushort)(BitConverter.SingleToUInt32Bits(value) >> 16);
    }

    private static float Read(Tensor tensor, int index) => tensor.DType == DType.F32 ? tensor.AsReadOnlySpan<float>()[index]
        : tensor.DType == DType.F16 ? (float)tensor.AsReadOnlySpan<Half>()[index]
        : BitConverter.UInt32BitsToSingle((uint)tensor.AsReadOnlySpan<ushort>()[index] << 16);
}
