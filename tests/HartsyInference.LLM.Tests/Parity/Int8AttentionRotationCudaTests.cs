using System.Diagnostics;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>CUDA INT8 attention v2 with a rotation (int8_attn_v2_rot.ptx between int8_attn_v2.ptx's quant and attention
/// launches) against <see cref="Int8AttentionV2Reference"/>.
/// <para>The bounds were fixed before the first run. Quantization: every Q, K and V byte equal and every Q, K and V scale
/// and K mean bit-equal to <see cref="Int8AttentionV2Reference.Quantize"/>, a NaN matching any NaN (the reference writes
/// the canonical one, the card whichever the arithmetic produced); keys past the valid ones hold zero bytes and scale 1.
/// Output with u8 P: bit-equal to the reference replayed on the kernel's own exp2 (an F16 output: to that rounded to
/// F16), a NaN matching any NaN. Output with F16 P, which sums on the tensor cores in another order: relative L2 at most
/// 1e-5 against the reference on its own exp2.</para>
/// <para>Most cases run 64 to 333 tokens, the route cases with <c>numerics.int8AttnV2MinTokens</c> lowered to 64. The
/// cases named for the default admitted size run 1,024 query rows and 1,024 or 1,100 keys, the route ones with that knob
/// untouched, under the same bounds (fixed before their first run too).</para></summary>
[Trait("Category", "GpuIntegration")]
[Collection("AttentionSemantics")]
public sealed class Int8AttentionRotationCudaTests(ITestOutputHelper output)
{
    private const int HeadDim = 128;
    private static readonly float Scale = 1f / MathF.Sqrt(HeadDim);

    /// <summary>The kernel's bytes, scales and means are the reference's for each transform, on F32 and F16 operands and
    /// both V scale groupings: two batches of three heads, a query tail and a key tail inside a tile. The untransformed
    /// row through the rotation kernel is also what the shipped quant launch writes, output included.</summary>
    [CudaTheory]
    [MemberData(nameof(QuantizeCases))]
    public void QuantizedOperandsEqualTheReference(Int8Rotation rotation, bool f16, bool tileScales)
    {
        if (!TryOpen(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        const int batch = 2, heads = 3, queryRows = 200, keys = 333;
        DType dtype = f16 ? DType.F16 : DType.F32;
        using Tensor query = Random(new TensorShape(batch, heads, queryRows, HeadDim), 11, 0f, 0f, dtype);
        using Tensor key = Random(new TensorShape(batch, heads, keys, HeadDim), 12, 3f, 6f, dtype);
        using Tensor value = Random(key.Shape, 13, 1f, 0f, dtype);
        using Tensor result = new(query.Shape, dtype);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(result, query, key, value, null, Scale, false);
        Int8AttentionV2Pv pv = tileScales ? Int8AttentionV2Pv.U8TileScales : Int8AttentionV2Pv.U8SequenceScales;
        string what = $"{rotation}, {dtype}, {pv}";

        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, Options(pv, rotation));
        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(in call, pv, 64, null, rotation, rotationKernel: true);
        AssertOperands(expected, gpu, batch * heads, queryRows, keys, tileScales, what);

        if (rotation != Int8Rotation.None) return;
        Int8V2Readback shipped = cuda.RunInt8AttentionV2ForTest(in call, pv, 64);
        Assert.True(shipped.Q8.AsSpan().SequenceEqual(gpu.Q8), $"Q bytes against the shipped quant launch, {what}");
        Assert.True(shipped.K8.AsSpan().SequenceEqual(gpu.K8), $"K bytes against the shipped quant launch, {what}");
        Assert.True(shipped.V8.AsSpan().SequenceEqual(gpu.V8), $"V bytes against the shipped quant launch, {what}");
        AssertFloats(shipped.QScale, gpu.QScale, $"Q scales against the shipped quant launch, {what}");
        AssertFloats(shipped.KScale, gpu.KScale, $"K scales against the shipped quant launch, {what}");
        AssertFloats(shipped.VScale, gpu.VScale, $"V scales against the shipped quant launch, {what}");
        Assert.True(shipped.Output.AsSpan().SequenceEqual(gpu.Output), $"output against the shipped launches, {what}");
    }

    public static IEnumerable<object[]> QuantizeCases()
    {
        Int8Rotation[] rotations = [Int8Rotation.None, Int8Rotation.H4, Int8Rotation.SignedH128, Int8Rotation.Pinned];
        foreach (Int8Rotation rotation in rotations)
            foreach (bool f16 in new[] { false, true })
                foreach (bool tileScales in new[] { false, true })
                    yield return [rotation, f16, tileScales];
    }

    /// <summary>The rows the rules bend on, each in a head of its own so one does not hide another: a Q row holding a NaN
    /// (a NaN scale, zero bytes, a NaN output row), an all-zero Q row (the scale floor of 1), a Q row holding an infinity,
    /// an all-zero K row among ordinary keys (it is minus the mean once centered), a K head holding one NaN (the mean, so
    /// every scale of that head, is NaN) and an all-zero K head (every scale 1, every byte 0).</summary>
    [CudaTheory]
    [InlineData(Int8Rotation.None)]
    [InlineData(Int8Rotation.H4)]
    [InlineData(Int8Rotation.SignedH128)]
    public void EdgeRowsFollowTheReference(Int8Rotation rotation)
    {
        if (!TryOpen(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        const int heads = 4, queryRows = 70, keys = 200;
        using Tensor query = Random(new TensorShape(1, heads, queryRows, HeadDim), 21, 0f, 0f, DType.F32);
        using Tensor key = Random(new TensorShape(1, heads, keys, HeadDim), 22, 2f, 0f, DType.F32);
        using Tensor value = Random(key.Shape, 23, 1f, 0f, DType.F32);
        Span<float> q = query.AsSpan<float>(), k = key.AsSpan<float>();
        q[3 * HeadDim + 17] = float.NaN;
        q.Slice(5 * HeadDim, HeadDim).Clear();
        q[9 * HeadDim + 40] = float.PositiveInfinity;
        k.Slice((1 * keys + 7) * HeadDim, HeadDim).Clear();
        k[(2 * keys + 11) * HeadDim + 3] = float.NaN;
        k.Slice(3 * keys * HeadDim, keys * HeadDim).Clear();
        using Tensor result = new(query.Shape, DType.F32);
        using Tensor referenceResult = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(result, query, key, value, null, Scale, false);
        AttentionCall reference = call with { Output = AttentionOperand.HeadMajor(referenceResult) };
        Int8AttentionV2Pv pv = Int8AttentionV2Pv.U8TileScales;
        Int8AttentionV2Options options = Options(pv, rotation);

        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, options);
        Assert.True(float.IsNaN(expected.QueryScales[3]));
        Assert.Equal(1f, expected.QueryScales[5]);
        Assert.True(float.IsPositiveInfinity(expected.QueryScales[9]) || float.IsNaN(expected.QueryScales[9]));
        Assert.True(expected.KeyScales[1 * keys + 7] > 0f && expected.KeyScales[1 * keys + 7] != 1f);
        Assert.All(expected.KeyScales.AsSpan(2 * keys, keys).ToArray(), scale => Assert.True(float.IsNaN(scale)));
        Assert.All(expected.KeyScales.AsSpan(3 * keys, keys).ToArray(), scale => Assert.Equal(1f, scale));

        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(in call, pv, 64, null, rotation, rotationKernel: true);
        AssertOperands(expected, gpu, heads, queryRows, keys, tileScales: true, $"edge rows, {rotation}");

        Int8AttentionV2Reference.Run(in reference, Replayed(cuda, in reference, options));
        float[] replayed = referenceResult.AsReadOnlySpan<float>().ToArray();
        Assert.All(replayed.AsSpan(3 * HeadDim, HeadDim).ToArray(), x => Assert.True(float.IsNaN(x)));
        Assert.Contains(replayed, x => !float.IsNaN(x));
        AssertFloats(replayed, Floats(gpu.Output, DType.F32), $"output on the kernel's exp2, edge rows, {rotation}");
    }

    /// <summary>The attention launch over the rotated bytes: the reference's output on the kernel's exp2, bit for bit, at
    /// both tile heights.</summary>
    [CudaTheory]
    [MemberData(nameof(OutputCases))]
    public void OutputEqualsTheReferenceOnTheKernelsExp2(Int8Rotation rotation, bool f16, bool tileScales, int blockRows)
    {
        if (!TryOpen(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        const int batch = 2, heads = 3, queryRows = 200, keys = 333;
        DType dtype = f16 ? DType.F16 : DType.F32;
        using Tensor query = Random(new TensorShape(batch, heads, queryRows, HeadDim), 31, 0f, 0f, dtype);
        using Tensor key = Random(new TensorShape(batch, heads, keys, HeadDim), 32, 3f, 6f, dtype);
        using Tensor value = Random(key.Shape, 33, 1f, 0f, dtype);
        using Tensor result = new(query.Shape, dtype);
        using Tensor referenceResult = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(result, query, key, value, null, Scale, false);
        AttentionCall reference = call with { Output = AttentionOperand.HeadMajor(referenceResult) };
        Int8AttentionV2Pv pv = tileScales ? Int8AttentionV2Pv.U8TileScales : Int8AttentionV2Pv.U8SequenceScales;
        Int8AttentionV2Options options = Options(pv, rotation);

        float[] actual = Floats(cuda.RunInt8AttentionV2ForTest(in call, pv, blockRows, null, rotation).Output, dtype);
        Int8AttentionV2Reference.Run(in reference, Replayed(cuda, in reference, options));
        float[] replayed = referenceResult.AsReadOnlySpan<float>().ToArray();
        if (f16)
            for (int i = 0; i < replayed.Length; i++) replayed[i] = (float)(Half)replayed[i];
        AssertFloats(replayed, actual, $"output on the kernel's exp2, {rotation}, {dtype}, {pv}, BR {blockRows}");
    }

    public static IEnumerable<object[]> OutputCases()
    {
        foreach (Int8Rotation rotation in new[] { Int8Rotation.H4, Int8Rotation.SignedH128 })
        {
            foreach (bool f16 in new[] { false, true })
                foreach (bool tileScales in new[] { false, true })
                    yield return [rotation, f16, tileScales, 64];
            yield return [rotation, false, true, 128];
        }
    }

    /// <summary>The F16 PV entries read the same rotated Q and K: Q and K bytes and scales are the reference's, and the
    /// output is within the F16 PV bound of the reference on its own exp2.</summary>
    [CudaTheory]
    [InlineData(Int8Rotation.H4)]
    [InlineData(Int8Rotation.SignedH128)]
    public void TheF16ProbabilityRowReadsTheRotatedOperands(Int8Rotation rotation)
    {
        if (!TryOpen(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        const int heads = 3, queryRows = 200, keys = 333;
        using Tensor query = Random(new TensorShape(1, heads, queryRows, HeadDim), 41, 0f, 0f, DType.F32);
        using Tensor key = Random(new TensorShape(1, heads, keys, HeadDim), 42, 3f, 6f, DType.F32);
        using Tensor value = Random(key.Shape, 43, 1f, 0f, DType.F32);
        using Tensor result = new(query.Shape, DType.F32);
        using Tensor referenceResult = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(result, query, key, value, null, Scale, false);
        AttentionCall reference = call with { Output = AttentionOperand.HeadMajor(referenceResult) };
        Int8AttentionV2Options options = Options(Int8AttentionV2Pv.F16, rotation);

        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, options);
        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(in call, Int8AttentionV2Pv.F16, 64, null, rotation);
        AssertQueriesAndKeys(expected, gpu, heads, keys, $"F16 PV, {rotation}");

        Int8AttentionV2Reference.Run(in reference, options);
        ReadOnlySpan<float> want = referenceResult.AsReadOnlySpan<float>();
        float[] actual = Floats(gpu.Output, DType.F32);
        double error = 0, norm = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            error += ((double)actual[i] - want[i]) * ((double)actual[i] - want[i]);
            norm += (double)want[i] * want[i];
        }
        double relative = Math.Sqrt(error / norm);
        output.WriteLine($"F16 PV, {rotation}: relative L2 {relative:E2} against the reference on its own exp2");
        Assert.True(relative <= 1e-5, $"F16 PV, {rotation}: relative L2 {relative:E2} against 1e-5");
    }

    /// <summary>A grouped call on the public route with <c>numerics.int8AttnV2Rotate</c>: the widening row hands it to the
    /// INT8 v2 row, which rotates by the pinned rule (the 4-point transform up to 256 keys, the signed 128-point one past
    /// that) and matches the reference on the grouped call. Without the knob the same call runs unrotated, as before.</summary>
    [CudaTheory]
    [InlineData(160, 200, Int8Rotation.H4)]
    [InlineData(300, 300, Int8Rotation.SignedH128)]
    public void AGroupedCallOnTheRowRotatesByThePinnedRuleOnlyWhenAsked(int queryRows, int keys, Int8Rotation pinned)
    {
        if (!TryOpen(out CudaBackend probe)) return;
        probe.Dispose();
        const int heads = 4, kvHeads = 2;
        Assert.Equal(pinned, Int8AttentionRotation.Select(HeadDim, keys));
        using Tensor query = Random(new TensorShape(1, heads, queryRows, HeadDim), 51, 0f, 0f, DType.F32);
        using Tensor key = Random(new TensorShape(1, kvHeads, keys, HeadDim), 52, 3f, 6f, DType.F32);
        using Tensor value = Random(key.Shape, 53, 1f, 0f, DType.F32);
        using Tensor referenceResult = new(query.Shape, DType.F32);
        AttentionCall reference = AttentionCall.FromScaledDotProductAttention(referenceResult, query, key, value, null, Scale, false);
        float[] rotated = Route(rotate: true), plain = Route(rotate: false);

        float[] Route(bool rotate)
        {
            using IDisposable policy = Policy(rotate).Push();
            using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
            using Tensor result = new(query.Shape, DType.F32);
            cuda.ScaledDotProductAttention(result, query, key, value, null, Scale);
            cuda.Sync();
            Assert.Equal(1, cuda.Int8AttentionV2ExecutionCount);
            Int8AttentionV2Options options = Options(Int8AttentionV2Pv.U8TileScales, rotate ? Int8Rotation.Pinned : Int8Rotation.None);
            Int8AttentionV2Reference.Run(in reference, Replayed(cuda, in reference, options));
            AssertFloats(referenceResult.AsReadOnlySpan<float>(), result.AsReadOnlySpan<float>(),
                $"grouped output on the kernel's exp2, {queryRows} rows, {keys} keys, rotate {rotate}");
            return result.AsReadOnlySpan<float>().ToArray();
        }

        Assert.False(rotated.AsSpan().SequenceEqual(plain), "the rotated and unrotated rows wrote the same output");
    }

    /// <summary>At a size the rows admit by default (1,024 query rows; 1,100 keys, so eight whole key tiles and a short
    /// one): bytes, scales, means and the u8 output for both transforms. Past 256 keys the pinned rule picks only the
    /// signed 128-point transform, so the 4-point one is reachable at this size through the hook alone.</summary>
    [CudaTheory]
    [InlineData(Int8Rotation.H4, false)]
    [InlineData(Int8Rotation.H4, true)]
    [InlineData(Int8Rotation.SignedH128, false)]
    [InlineData(Int8Rotation.SignedH128, true)]
    public void OperandsAndOutputAtTheDefaultAdmittedSizeEqualTheReference(Int8Rotation rotation, bool f16)
    {
        if (!TryOpen(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        const int heads = 2, queryRows = 1024, keys = 1100;
        DType dtype = f16 ? DType.F16 : DType.F32;
        using Tensor query = Random(new TensorShape(1, heads, queryRows, HeadDim), 81, 0f, 0f, dtype);
        using Tensor key = Random(new TensorShape(1, heads, keys, HeadDim), 82, 3f, 6f, dtype);
        using Tensor value = Random(key.Shape, 83, 1f, 0f, dtype);
        using Tensor result = new(query.Shape, dtype);
        using Tensor referenceResult = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(result, query, key, value, null, Scale, false);
        AttentionCall reference = call with { Output = AttentionOperand.HeadMajor(referenceResult) };
        Int8AttentionV2Pv pv = Int8AttentionV2Pv.U8TileScales;
        Int8AttentionV2Options options = Options(pv, rotation);
        string what = $"{rotation}, {dtype}, {queryRows} rows, {keys} keys";

        Int8AttentionV2Operands expected = Int8AttentionV2Reference.Quantize(in call, options);
        Int8V2Readback gpu = cuda.RunInt8AttentionV2ForTest(in call, pv, 64, null, rotation);
        AssertOperands(expected, gpu, heads, queryRows, keys, tileScales: true, what);

        long start = Stopwatch.GetTimestamp();
        Int8AttentionV2Reference.Run(in reference, Replayed(cuda, in reference, options));
        output.WriteLine($"{what}: the reference twice and its exp2 table took {Stopwatch.GetElapsedTime(start).TotalSeconds:F1} s");
        float[] replayed = referenceResult.AsReadOnlySpan<float>().ToArray();
        if (f16)
            for (int i = 0; i < replayed.Length; i++) replayed[i] = (float)(Half)replayed[i];
        AssertFloats(replayed, Floats(gpu.Output, dtype), $"output on the kernel's exp2, {what}");
    }

    /// <summary>The public route with only <c>numerics.int8AttnV2</c> and <c>numerics.int8AttnV2Rotate</c> set, so the
    /// row admits the call at its default size: 1,024 query rows and keys, F32 and F16 operands, and four query heads
    /// on two key heads. The INT8 v2 row runs once and the output is the reference's with the pinned transform.</summary>
    [CudaTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void ACallOfTheDefaultAdmittedSizeRotatesOnTheRow(bool f16, bool grouped)
    {
        if (!TryOpen(out CudaBackend probe)) return;
        probe.Dispose();
        const int tokens = 1024, kvHeads = 2;
        int heads = grouped ? 4 : 2;
        DType dtype = f16 ? DType.F16 : DType.F32;
        Assert.Equal(Int8Rotation.SignedH128, Int8AttentionRotation.Select(HeadDim, tokens));
        using Tensor query = Random(new TensorShape(1, heads, tokens, HeadDim), 91, 0f, 0f, dtype);
        using Tensor key = Random(new TensorShape(1, kvHeads, tokens, HeadDim), 92, 3f, 6f, dtype);
        using Tensor value = Random(key.Shape, 93, 1f, 0f, dtype);
        using Tensor result = new(query.Shape, dtype);
        using Tensor referenceResult = new(query.Shape, DType.F32);
        AttentionCall reference = AttentionCall.FromScaledDotProductAttention(referenceResult, query, key, value, null, Scale, false);
        string what = $"{dtype}, {heads} heads on {kvHeads}, {tokens} tokens";

        // SageAttn stays at its default, as on the default route: a planner that ranks another row first fails the count below.
        using IDisposable policy = KnobProfile.Create("int8v2-rotation-default-size")
            .With(EngineKnobs.Int8AttnV2, true).With(EngineKnobs.Int8AttnV2Rotate, true).Push();
        Assert.Equal(tokens, EngineKnobs.Int8AttnV2MinTokens.Value);
        using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
        using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
        cuda.ScaledDotProductAttention(result, query, key, value, null, Scale);
        cuda.Sync();
        AttentionEvidenceEvent[] executions = evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution).ToArray();
        Assert.Equal(1, cuda.Int8AttentionV2ExecutionCount);
        Assert.Contains(executions, e => e.Implementation == "int8.v2" && e.KvHeads == heads);
        Assert.Equal(grouped, executions.Any(e => e.Implementation == "widen.preferred"));

        Int8AttentionV2Reference.Run(in reference, Replayed(cuda, in reference, Options(Int8AttentionV2Pv.U8TileScales, Int8Rotation.Pinned)));
        float[] replayed = referenceResult.AsReadOnlySpan<float>().ToArray();
        float[] actual = new float[replayed.Length];
        if (f16)
        {
            ReadOnlySpan<Half> halves = result.AsReadOnlySpan<Half>();
            for (int i = 0; i < replayed.Length; i++) (replayed[i], actual[i]) = ((float)(Half)replayed[i], (float)halves[i]);
        }
        else result.AsReadOnlySpan<float>().CopyTo(actual);
        AssertFloats(replayed, actual, $"output on the kernel's exp2 through the row, {what}");
    }

    /// <summary>The transforms the reference refuses at head dim 128 are refused here, before any launch.</summary>
    [CudaTheory]
    [InlineData(Int8Rotation.H64)]
    [InlineData(Int8Rotation.PaddedH128)]
    public void TransformsOfAnotherHeadDimAreRefused(Int8Rotation rotation)
    {
        if (!TryOpen(out CudaBackend cuda)) return;
        using CudaBackend owner = cuda;
        using Tensor query = Random(new TensorShape(1, 1, 64, HeadDim), 61, 0f, 0f, DType.F32);
        using Tensor result = new(query.Shape, DType.F32);
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(result, query, query, query, null, Scale, false);
        Assert.Throws<ArgumentException>(() => Int8AttentionV2Reference.Quantize(in call, Options(Int8AttentionV2Pv.U8TileScales, rotation)));
        Assert.Throws<ArgumentException>(
            () => cuda.RunInt8AttentionV2ForTest(in call, Int8AttentionV2Pv.U8TileScales, 64, null, rotation));
    }

    /// <summary>The rotation kernel's files are never candidates for <c>int8_attn_v2</c>, whose only builds are sm_80 and
    /// up: a Turing card must find no INT8 v2 module, not the rotation floor build under the attention kernel's name.</summary>
    [CudaFact]
    public void TheRotationFilesAreNotVariantsOfTheAttentionKernel()
    {
        string ptx = CudaPtxDirectory.Resolve();
        Assert.Equal("int8_attn_v2.ptx", Path.GetFileName(CudaKernels.PtxPath(ptx, "int8_attn_v2", CudaArch.Turing)));
        Assert.Equal("int8_attn_v2.ptx", Path.GetFileName(CudaKernels.PtxPath(ptx, "int8_attn_v2", 86)));
        Assert.Equal("int8_attn_v2_rot.sm75.ptx", Path.GetFileName(CudaKernels.PtxPath(ptx, "int8_attn_v2_rot", CudaArch.Turing)));
        Assert.Equal("int8_attn_v2_rot.ptx", Path.GetFileName(CudaKernels.PtxPath(ptx, "int8_attn_v2_rot", 86)));
    }

    /// <summary>A rotation file that does not load leaves the backend and the unrotated rows working, and a call asked to
    /// rotate takes another row instead of running unrotated.</summary>
    [CudaFact]
    public void ARotationKernelThatDoesNotLoadSendsACallAskedToRotateToAnotherRow()
    {
        if (!TryOpen(out CudaBackend probe)) return;
        probe.Dispose();
        const int heads = 2, tokens = 160;
        using Tensor query = Random(new TensorShape(1, heads, tokens, HeadDim), 71, 0f, 0f, DType.F32);
        using Tensor key = Random(new TensorShape(1, heads, tokens, HeadDim), 72, 3f, 6f, DType.F32);
        using Tensor value = Random(key.Shape, 73, 1f, 0f, DType.F32);
        using Tensor rotated = new(query.Shape, DType.F32), plain = new(query.Shape, DType.F32);
        Func<string, Exception?>? before = CudaKernels.ModuleLoadFailureForTests;
        CudaKernels.ModuleLoadFailureForTests = path => path.EndsWith("int8_attn_v2_rot.ptx", StringComparison.Ordinal)
            ? new InvalidOperationException("injected: the rotation PTX does not load") : before?.Invoke(path);
        try
        {
            using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
            using (Policy(rotate: true).Push())
            {
                cuda.ScaledDotProductAttention(rotated, query, key, value, null, Scale);
                cuda.Sync();
            }
            Assert.Equal(0, cuda.Int8AttentionV2ExecutionCount);
            Assert.DoesNotContain(rotated.AsReadOnlySpan<float>().ToArray(), float.IsNaN);
            using (Policy(rotate: false).Push())
            {
                cuda.ScaledDotProductAttention(plain, query, key, value, null, Scale);
                cuda.Sync();
            }
            Assert.Equal(1, cuda.Int8AttentionV2ExecutionCount);
        }
        finally
        {
            CudaKernels.ModuleLoadFailureForTests = before;
        }
    }

    private static KnobProfile Policy(bool rotate) => KnobProfile.Create("int8v2-rotation")
        .With(EngineKnobs.Int8AttnV2, true).With(EngineKnobs.Int8AttnV2MinTokens, 64).With(EngineKnobs.SageAttn, false)
        .With(EngineKnobs.Int8AttnV2Rotate, rotate);

    private static Int8AttentionV2Options Options(Int8AttentionV2Pv pv, Int8Rotation rotation) => new()
    {
        Rotation = rotation, ValueScalesPerTile = pv == Int8AttentionV2Pv.U8TileScales,
        Probabilities = pv == Int8AttentionV2Pv.F16 ? Int8ProbabilityFormat.F16 : Int8ProbabilityFormat.U8,
    };

    private bool TryOpen(out CudaBackend cuda)
    {
        cuda = null!;
        // int8_attn_v2.ptx is built for sm_80 and up, so a Turing card has no attention launch for the rotated rows.
        if (CudaContext.QuerySm(0) < CudaArch.Ampere)
        {
            output.WriteLine("SKIPPED: INT8 attention v2 needs sm_80 or newer; nothing was checked.");
            return false;
        }
        cuda = new CudaBackend(0, CudaPtxDirectory.Resolve());
        return true;
    }

    /// <summary>Uniform values in [-1, 1) plus a component every token of a head shares (what centering removes) and an
    /// offset on channel 5 (one large channel, what a rotation spreads).</summary>
    private static Tensor Random(TensorShape shape, int seed, float shared, float outlier, DType dtype)
    {
        Tensor tensor = new(shape, DType.F32);
        Random random = new(seed);
        int heads = (int)(shape[0] * shape[1]), tokens = (int)shape[2], width = (int)shape[3];
        float[] common = new float[heads * width];
        for (int i = 0; i < common.Length; i++) common[i] = (random.NextSingle() * 2f - 1f) * shared;
        Span<float> values = tensor.AsSpan<float>();
        for (int i = 0; i < values.Length; i++)
        {
            int head = i / (tokens * width), channel = i % width;
            values[i] = random.NextSingle() * 2f - 1f + common[head * width + channel] + (channel == 5 ? outlier : 0f);
        }
        if (dtype == DType.F32) return tensor;
        using Tensor wide = tensor;
        return wide.CastTo(dtype);
    }

    /// <summary>The reference with the kernel's exp2: the arguments it asks for, evaluated on the card, replayed.</summary>
    private static Int8AttentionV2Options Replayed(CudaBackend cuda, in AttentionCall call, Int8AttentionV2Options options)
    {
        HashSet<float> arguments = [];
        Int8AttentionV2Reference.Run(in call, options with { Exp2 = x => { arguments.Add(x); return (float)Math.Pow(2.0, x); } });
        float[] asked = [.. arguments];
        float[] values = cuda.Int8V2Exp2ForTest(asked);
        Dictionary<float, float> table = new(asked.Length);
        for (int i = 0; i < asked.Length; i++) table[asked[i]] = values[i];
        return options with { Exp2 = x => table[x] };
    }

    private static float[] Floats(byte[] bytes, DType dtype)
    {
        float[] values = new float[bytes.Length / (dtype == DType.F32 ? 4 : 2)];
        for (int i = 0; i < values.Length; i++)
            values[i] = dtype == DType.F32 ? BitConverter.ToSingle(bytes, i * 4) : (float)BitConverter.ToHalf(bytes, i * 2);
        return values;
    }

    private static void AssertQueriesAndKeys(Int8AttentionV2Operands expected, Int8V2Readback gpu, int batchHeads, int keys, string what)
    {
        int kvPad = (int)gpu.Layout.KvPad;
        AssertFloats(expected.KeyMeans, gpu.KMean, $"key means, {what}");
        AssertFloats(expected.QueryScales, gpu.QScale, $"query scales, {what}");
        AssertBytes(expected.Queries, gpu.Q8, $"query bytes, {what}");
        float[] keyScales = new float[batchHeads * keys];
        sbyte[] keyBytes = new sbyte[batchHeads * keys * HeadDim];
        for (int bh = 0; bh < batchHeads; bh++)
        {
            gpu.KScale.AsSpan(bh * kvPad, keys).CopyTo(keyScales.AsSpan(bh * keys));
            gpu.K8.AsSpan(bh * kvPad * HeadDim, keys * HeadDim).CopyTo(keyBytes.AsSpan(bh * keys * HeadDim));
            foreach (float scale in gpu.KScale.AsSpan(bh * kvPad + keys, kvPad - keys))
                Assert.True(scale == 1f, $"key scale past the valid keys is {scale:R}, {what}");
            foreach (sbyte b in gpu.K8.AsSpan((bh * kvPad + keys) * HeadDim, (kvPad - keys) * HeadDim))
                Assert.True(b == 0, $"key byte past the valid keys is {b}, {what}");
        }
        AssertFloats(expected.KeyScales, keyScales, $"key scales, {what}");
        AssertBytes(expected.Keys, keyBytes, $"key bytes, {what}");
    }

    private static void AssertOperands(Int8AttentionV2Operands expected, Int8V2Readback gpu, int batchHeads, int queryRows, int keys,
        bool tileScales, string what)
    {
        Assert.Equal(batchHeads * queryRows, gpu.QScale.Length);
        AssertQueriesAndKeys(expected, gpu, batchHeads, keys, what);
        AssertFloats(expected.ValueScales, gpu.VScale, $"value scales ({(tileScales ? "per tile" : "per head")}), {what}");
        int kvPad = (int)gpu.Layout.KvPad;
        sbyte[] valueBytes = new sbyte[batchHeads * keys * HeadDim];
        for (int bh = 0; bh < batchHeads; bh++)
            for (int channel = 0; channel < HeadDim; channel++)
                for (int j = 0; j < kvPad; j++)
                {
                    sbyte b = gpu.V8[(bh * HeadDim + channel) * kvPad + Int8AttentionV2Layout.PhysicalKey(j)];
                    if (j < keys) valueBytes[(bh * keys + j) * HeadDim + channel] = b;
                    else Assert.True(b == 0, $"value byte past the valid keys is {b}, {what}");
                }
        AssertBytes(expected.Values, valueBytes, $"value bytes, {what}");
    }

    /// <summary>Bit-equal floats, a NaN matching any NaN.</summary>
    private static void AssertFloats(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int misses = 0, first = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            bool same = float.IsNaN(expected[i]) ? float.IsNaN(actual[i])
                : BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]);
            if (same) continue;
            if (first < 0) first = i;
            misses++;
        }
        Assert.True(misses == 0, $"{what}: {misses} of {expected.Length} differ; element {first} is "
            + $"{(first < 0 ? 0f : actual[first]):R}, expected {(first < 0 ? 0f : expected[first]):R}");
    }

    private static void AssertBytes(ReadOnlySpan<sbyte> expected, ReadOnlySpan<sbyte> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        int misses = 0, first = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] == actual[i]) continue;
            if (first < 0) first = i;
            misses++;
        }
        Assert.True(misses == 0, $"{what}: {misses} of {expected.Length} differ; element {first} is "
            + $"{(first < 0 ? 0 : actual[first])}, expected {(first < 0 ? 0 : expected[first])}");
    }
}
