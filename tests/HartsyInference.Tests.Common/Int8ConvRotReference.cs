namespace HartsyInference.Tests.Common;

/// <summary>The comfy-kitchen eager INT8 / INT8-ConvRot dump consumed by the CPU and CUDA parity tests.</summary>
/// <remarks>Regenerate with <c>tests/python-reference/int8_convrot_reference.py</c> under the ComfyUI venv. The
/// <c>.bin</c> is gitignored by the repo-wide <c>*.bin</c> rule, so every caller must treat a null load as "skip".
/// Lives here rather than in one test project because both <c>HartsyInference.Diffusion.Tests</c> (CPU/codec) and
/// <c>HartsyInference.Cuda.Tests</c> (resident int8 GEMM) read the same layout.</remarks>
public sealed class Int8ConvRotReference
{
    /// <summary>Normalized regular Hadamard matrices, keyed by group size, row-major <c>[size, size]</c>.</summary>
    public required IReadOnlyDictionary<int, float[]> Hadamards { get; init; }

    /// <summary>One entry per <c>(N, K, M, G)</c> shape the generator dumped.</summary>
    public required IReadOnlyList<Case> Cases { get; init; }

    /// <summary>Locates the dump, honouring the <c>INT8_CONVROT_REFERENCE_BIN</c> override, or null when absent.</summary>
    public static string? FixturePath()
    {
        string? env = Environment.GetEnvironmentVariable("INT8_CONVROT_REFERENCE_BIN");
        if (!string.IsNullOrWhiteSpace(env)) return File.Exists(env) ? env : null;

        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "tests", "python-reference", "int8_convrot_reference.bin");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>Loads the dump, or returns null when it has not been generated on this machine.</summary>
    public static Int8ConvRotReference? TryLoad()
    {
        string? path = FixturePath();
        if (path is null) return null;   // tier-lint: guarded

        using BinaryReader reader = new BinaryReader(File.OpenRead(path));
        Dictionary<int, float[]> hadamards = new();
        int hadamardCount = reader.ReadInt32();
        for (int i = 0; i < hadamardCount; i++)
        {
            int size = reader.ReadInt32();
            hadamards[size] = ReadFloats(reader, (long)size * size);
        }

        List<Case> cases = new();
        int caseCount = reader.ReadInt32();
        for (int i = 0; i < caseCount; i++)
        {
            int n = reader.ReadInt32(), k = reader.ReadInt32(), m = reader.ReadInt32(), group = reader.ReadInt32();
            cases.Add(new Case
            {
                OutFeatures = n,
                InFeatures = k,
                Rows = m,
                GroupSize = group,
                Weight = ReadFloats(reader, (long)n * k),
                Quant = ReadSBytes(reader, (long)n * k),
                RowScale = ReadFloats(reader, n),
                Activation = ReadFloats(reader, (long)m * k),
                Bias = ReadFloats(reader, n),
                Output = ReadFloats(reader, (long)m * n),
                OutputWithBias = ReadFloats(reader, (long)m * n),
            });
        }
        return new Int8ConvRotReference { Hadamards = hadamards, Cases = cases };
    }

    /// <summary>Relative L2 of <paramref name="actual"/> against <paramref name="expected"/>.</summary>
    public static double RelL2(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = (double)actual[i] - expected[i];
            num += diff * diff;
            den += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static float[] ReadFloats(BinaryReader reader, long count)
    {
        byte[] raw = reader.ReadBytes(checked((int)(count * sizeof(float))));
        float[] values = new float[count];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return values;
    }

    private static sbyte[] ReadSBytes(BinaryReader reader, long count)
    {
        byte[] raw = reader.ReadBytes(checked((int)count));
        sbyte[] values = new sbyte[count];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return values;
    }

    /// <summary>One dumped shape: the original weight, its int8 packing, the activation, and eager's two outputs.</summary>
    public sealed record Case
    {
        /// <summary>Weight rows (<c>N</c>).</summary>
        public required int OutFeatures { get; init; }

        /// <summary>Weight columns (<c>K</c>).</summary>
        public required int InFeatures { get; init; }

        /// <summary>Activation rows (<c>M</c>).</summary>
        public required int Rows { get; init; }

        /// <summary>ConvRot group size, 0 when the layer was quantized unrotated.</summary>
        public required int GroupSize { get; init; }

        /// <summary>The pre-quantization <c>[N, K]</c> weight.</summary>
        public required float[] Weight { get; init; }

        /// <summary>The <c>[N, K]</c> int8 weight, already rotated when <see cref="GroupSize"/> is non-zero.</summary>
        public required sbyte[] Quant { get; init; }

        /// <summary>Per-output-row dequant scale.</summary>
        public required float[] RowScale { get; init; }

        /// <summary>The <c>[M, K]</c> activation.</summary>
        public required float[] Activation { get; init; }

        /// <summary>Per-output-row bias.</summary>
        public required float[] Bias { get; init; }

        /// <summary>Eager <c>int8_linear</c> output with no bias.</summary>
        public required float[] Output { get; init; }

        /// <summary>Eager <c>int8_linear</c> output with <see cref="Bias"/> applied.</summary>
        public required float[] OutputWithBias { get; init; }

        /// <summary>Describes the shape for assert messages.</summary>
        public override string ToString() => $"N{OutFeatures} K{InFeatures} M{Rows} G{GroupSize}";
    }
}
