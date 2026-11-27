using System.Diagnostics;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.LLM.Tests.Parity;

/// <summary>A descriptive timing of CUDA's INT8 attention v2 row with and without the rotation, beside the F32 dense row
/// the same call takes with default knobs. It asserts nothing about speed, only that each arm ran the row it is meant
/// to measure, and is a <c>GpuBenchmark</c>, which the CPU lane and the GPU lane both leave out: run it alone on an
/// idle card.
/// <para>One call of 1,024 query rows and 1,024 keys, 8 heads, head dim 128, F32, every operand already on the device.
/// Each timing is one <c>ScaledDotProductAttention</c> and a sync: the median of 7 after 2 warm-ups, the arms taken in
/// turn within every round. Past 256 keys the pinned rule picks the signed 128-point transform.</para>
/// <para>Expectation written before the first run: the rotated row is slower than the unrotated one by 5 to 20 percent
/// of the whole call, because it quantizes K twice and adds a launch while the attention launch, which should dominate,
/// is the same; and the unrotated INT8 row is faster than the F32 dense row, between 1.2 and 4 times.</para></summary>
[Trait("Category", "GpuBenchmark")]
[Collection("AttentionSemantics")]
public sealed class Int8AttentionRotationCudaTimingTests(ITestOutputHelper output)
{
    private const int Heads = 8, Tokens = 1024, HeadDim = 128, Runs = 7, WarmUps = 2;

    [CudaFact]
    public void TimesTheRotatedRowBesideTheUnrotatedRowAndF32()
    {
        if (CudaContext.QuerySm(0) < CudaArch.Ampere)
        {
            output.WriteLine("SKIPPED: INT8 attention v2 needs sm_80 or newer; nothing was timed.");
            return;
        }
        float scale = 1f / MathF.Sqrt(HeadDim);
        using CudaBackend cuda = new(0, CudaPtxDirectory.Resolve());
        TensorShape shape = new(1, Heads, Tokens, HeadDim);
        using Tensor query = OnDevice(cuda, shape, 1), key = OnDevice(cuda, shape, 2), value = OnDevice(cuda, shape, 3);
        cuda.Sync();
        (string Name, KnobProfile Policy, bool Int8)[] arms =
        [
            ("INT8 v2, unrotated", Policy("int8v2-timing-plain").With(EngineKnobs.Int8AttnV2, true).With(EngineKnobs.Int8AttnV2Rotate, false), true),
            ("INT8 v2, rotated (SignedH128)", Policy("int8v2-timing-rotated").With(EngineKnobs.Int8AttnV2, true).With(EngineKnobs.Int8AttnV2Rotate, true), true),
            ("F32 dense, default knobs", Policy("int8v2-timing-f32").With(EngineKnobs.Int8AttnV2, false), false),
        ];
        using Tensor plain = new(shape, DType.F32), rotated = new(shape, DType.F32), dense = new(shape, DType.F32);
        Tensor[] results = [plain, rotated, dense];

        // One untimed call an arm with the evidence on: the row each arm takes.
        string[] routes = new string[arms.Length];
        for (int arm = 0; arm < arms.Length; arm++)
        {
            long before = cuda.Int8AttentionV2ExecutionCount;
            using AttentionEvidenceSession evidence = AttentionEvidence.Begin(64);
            Call(arm);
            AttentionEvidenceEvent[] executions = evidence.Snapshot().Where(e => e.Kind == AttentionEvidenceKind.Execution).ToArray();
            routes[arm] = string.Join(" > ", executions.Select(e => $"{e.Implementation} ({e.Branch})"));
            Assert.NotEmpty(executions);
            Assert.Equal(arms[arm].Int8 ? 1 : 0, cuda.Int8AttentionV2ExecutionCount - before);
            Assert.Equal(arms[arm].Int8, executions.Any(e => e.Implementation == "int8.v2"));
            Assert.DoesNotContain(executions, e => !arms[arm].Int8 && e.Implementation is { } id && (id.StartsWith("int8", StringComparison.Ordinal)
                || id.StartsWith("sage", StringComparison.Ordinal)));
        }
        Assert.False(plain.AsReadOnlySpan<float>().SequenceEqual(rotated.AsReadOnlySpan<float>()), "the rotated arm wrote the unrotated output");

        double[][] ms = [new double[Runs], new double[Runs], new double[Runs]];
        long timedFrom = cuda.Int8AttentionV2ExecutionCount;
        for (int round = 0; round < WarmUps + Runs; round++)
            for (int turn = 0; turn < arms.Length; turn++)
            {
                // The first arm of a round moves on by one each round, so no arm always follows the same one.
                int arm = (turn + round) % arms.Length;
                double elapsed = Call(arm);
                if (round >= WarmUps) ms[arm][round - WarmUps] = elapsed;
            }
        Assert.Equal(2 * (WarmUps + Runs), cuda.Int8AttentionV2ExecutionCount - timedFrom);

        double[] medians = new double[arms.Length];
        for (int arm = 0; arm < arms.Length; arm++)
        {
            Array.Sort(ms[arm]);
            medians[arm] = ms[arm][Runs / 2];
            output.WriteLine($"{arms[arm].Name}: median {medians[arm]:F2} ms, least {ms[arm][0]:F2}, most {ms[arm][Runs - 1]:F2}; route {routes[arm]}");
        }
        // That the arms attended the same call: INT8 is an approximation, so this is a loose sanity bound only.
        ReadOnlySpan<float> want = dense.AsReadOnlySpan<float>();
        double worstPlain = Worst(want, plain.AsReadOnlySpan<float>()), worstRotated = Worst(want, rotated.AsReadOnlySpan<float>());
        output.WriteLine($"{Heads} heads, {Tokens} rows, {Tokens} keys, D {HeadDim}: rotated over unrotated {medians[1] / medians[0]:F3}, "
            + $"F32 dense over unrotated {medians[2] / medians[0]:F2}, F32 dense over rotated {medians[2] / medians[1]:F2}; "
            + $"largest |INT8 - F32| unrotated {worstPlain:G3}, rotated {worstRotated:G3}");
        Assert.True(worstPlain < 0.25 && worstRotated < 0.25, $"INT8 and F32 outputs differ by {worstPlain} and {worstRotated}");
        // A second clock for the unrotated row: device events around its three launches, which leave out the host's share.
        AttentionCall call = AttentionCall.FromScaledDotProductAttention(plain, query, key, value, null, scale, false);
        float[] stages = cuda.Int8V2StageTimesForTest(in call, Int8AttentionV2Pv.U8TileScales, 64, 20);
        output.WriteLine($"unrotated launches by device events, mean of 20: stats {stages[0]:F3} ms, quant {stages[1]:F3} ms, "
            + $"attention {stages[2]:F3} ms, together {stages[0] + stages[1] + stages[2]:F3} ms");

        double Call(int arm)
        {
            using IDisposable policy = arms[arm].Policy.Push();
            long start = Stopwatch.GetTimestamp();
            cuda.ScaledDotProductAttention(results[arm], query, key, value, null, scale);
            cuda.Sync();
            return Stopwatch.GetElapsedTime(start).TotalMilliseconds;
        }
    }

    private static KnobProfile Policy(string name) => KnobProfile.Create(name);

    private static double Worst(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double worst = 0;
        for (int i = 0; i < a.Length; i++) worst = Math.Max(worst, Math.Abs(a[i] - b[i]));
        return worst;
    }

    /// <summary>Random values made by an op of the backend, so they live on the device and no timing includes an upload.</summary>
    private static Tensor OnDevice(CudaBackend cuda, TensorShape shape, int seed)
    {
        using Tensor host = new(shape, DType.F32);
        Random random = new(seed);
        Span<float> values = host.AsSpan<float>();
        for (int i = 0; i < values.Length; i++) values[i] = random.NextSingle() * 2f - 1f;
        Tensor device = new(shape, DType.F32);
        cuda.Tanh(device, host);
        return device;
    }
}
