using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using Xunit;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>Checks <c>IBackend.Na3d</c> against the LTX-2 reference's own eager neighborhood-attention fallback
/// (<c>ltx_core/model/video_vae/transformer/fallback_na/eager.py</c>, the vendored comfy-kitchen backend). Shared
/// q/k/v come from the fixture rather than a shared seed, since C# and torch do not generate the same randoms.
/// Regenerate with <c>tests/python-reference/na3d_reference.py</c>.</summary>
public sealed unsafe class Na3dParityTests
{
    private static string? FixturePath()
    {
        string? env = Environment.GetEnvironmentVariable("NA3D_REFERENCE_BIN");
        if (!string.IsNullOrWhiteSpace(env)) return File.Exists(env) ? env : null;

        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "tests", "python-reference", "na3d_reference.bin");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    private static Tensor FromFloats(long[] shape, float[] data)
    {
        Tensor t = new Tensor(new TensorShape(shape), DType.F32);
        data.AsSpan().CopyTo(new Span<float>((void*)t.DataPointer, data.Length));
        return t;
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void MatchesTheReferenceEagerNa3d()
    {
        string? path = FixturePath();
        if (path is null) return;   // tier-lint: guarded

        using BinaryReader reader = new BinaryReader(File.OpenRead(path));
        int caseCount = reader.ReadInt32();
        Assert.True(caseCount > 0);
        IBackend backend = new CpuBackend();

        for (int c = 0; c < caseCount; c++)
        {
            int b = reader.ReadInt32(), t = reader.ReadInt32(), h = reader.ReadInt32(), w = reader.ReadInt32();
            int heads = reader.ReadInt32(), headDim = reader.ReadInt32();
            int kt = reader.ReadInt32(), kh = reader.ReadInt32(), kw = reader.ReadInt32();
            long[] shape = [b, t, h, w, heads, headDim];
            int n = b * t * h * w * heads * headDim;

            float[] Read()
            {
                float[] values = new float[n];
                for (int i = 0; i < n; i++) values[i] = reader.ReadSingle();
                return values;
            }

            using Tensor q = FromFloats(shape, Read());
            using Tensor k = FromFloats(shape, Read());
            using Tensor v = FromFloats(shape, Read());
            float[] expected = Read();
            using Tensor actual = new Tensor(new TensorShape(shape), DType.F32);

            backend.Na3d(actual, q, k, v, kt, kh, kw, scale: 1.0f);

            float* got = (float*)actual.DataPointer;
            double num = 0, den = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = got[i] - expected[i];
                num += diff * diff;
                den += (double)expected[i] * expected[i];
            }
            double relL2 = den > 0 ? Math.Sqrt(num / den) : Math.Sqrt(num);
            Assert.True(relL2 < 1e-6,
                $"case {c} ({b},{t},{h},{w},{heads},{headDim}) kernel ({kt},{kh},{kw}): relL2 {relL2:E3}");
        }
    }
}
