using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Tests.Common.Attention;
using static HartsyInference.Tests.Common.Attention.AttentionGolden;
using Owner = HartsyInference.Tests.Common.Attention.GoldenOwner;
using Weights = HartsyInference.Tests.Common.Attention.GoldenWeights;

namespace HartsyInference.Audio.Tests.Attention;

/// <summary>Forward passes of the Wave E audio, TTS and ASR attention sites on tiny pseudo-random weights and inputs, each reduced to
/// SHA-256 digests of its output bits. Everything runs on the CPU backend through the sites' public/internal constructors, so the
/// same cases produce the golden fixture from the pre-migration source and must reproduce it after the attention refactor. Weights
/// come from the per-case-salted <c>Weights</c> harness (never a static RNG counter, which would make them order-dependent).</summary>
internal static unsafe partial class AttentionWaveEGoldenCases
{
    private const float NegInf = -1e9f;

    private static readonly Dictionary<string, Func<CpuBackend, string>> Cases = Build();

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } = [.. Cases.Keys];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        return Cases.TryGetValue(name, out Func<CpuBackend, string>? run)
            ? run(cpu)
            : throw new ArgumentException($"Unknown attention golden case '{name}'.", nameof(name));
    }

    private static Dictionary<string, Func<CpuBackend, string>> Build()
    {
        Dictionary<string, Func<CpuBackend, string>> d = new(StringComparer.Ordinal);
        RegisterAsr(d);
        RegisterTts1(d);
        RegisterTts2(d);
        RegisterDevice1(d);
        RegisterDevice2(d);
        RegisterDevice3(d);
        RegisterEnc1(d);
        RegisterEnc2(d);
        RegisterHost1(d);
        RegisterHost2(d);
        return d;
    }

    private static Tensor Seq(Owner o, uint seed, float scale, params long[] dims) => o.Own(Seeded(new TensorShape(dims), seed, scale));

    /// <summary>Additive F32 mask <c>[1, 1, rows, cols]</c>: 0 where <paramref name="allowed"/>, a large negative elsewhere.</summary>
    private static Tensor AdditiveMask(int rows, int cols, Func<int, int, bool> allowed)
    {
        Tensor mask = new(new TensorShape(1, 1, rows, cols), DType.F32);
        float* p = (float*)mask.DataPointer;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                p[i * cols + j] = allowed(i, j) ? 0f : NegInf;
        return mask;
    }
}
