using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>The per-element and per-point embeddings with one caller each: DIAMOND's 256-level pixel quantization over
/// <c>[-1, 1]</c> and Hunyuan3D's Fourier embedding of query coordinates. The fixtures were recorded against
/// the entry points <see cref="IBackend.FakeQuantizeUniform"/> and <see cref="IBackend.FourierEmbed"/> replaced. F32 everywhere.</summary>
public static class PointCases
{
    public const string Family = "point";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];

    /// <summary>A clamp, a truncation and two exact scalings.</summary>
    private static readonly Acceptance Exact = Acceptance.Exact;

    /// <summary>sin and cos follow each backend's math library.</summary>
    private static readonly Acceptance Transcendental = Acceptance.Within(1e-5, 1e-4, "sin and cos follow each backend's math library");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];
        // Values beyond ±1 exercise the clamp; the rest land on one of 256 levels.
        cases.Add(new(Family, "fake_quantize_f32", seed => new Operands(seed).In("x", new TensorShape(2, 3, 8), 1.5f).Out("out", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => be.FakeQuantizeUniform(t.Outputs["out"], t.Inputs["x"], -1f, 1f, 256), _ => true, All, Exact));
        // Eight octave bands over [count, 3] coordinates: x, then sin and cos per coordinate and band → 3·(2·8+1) = 51 wide.
        cases.Add(new(Family, "fourier_embed_f32", seed => new Operands(seed).In("coords", new TensorShape(5, 3), 2f).Out("out", new TensorShape(5, 51)).Tensors,
            (be, t) => be.FourierEmbed(t.Outputs["out"], t.Inputs["coords"], new FourierEmbedSpec(3, 8, IncludeInput: true)), _ => true, All, Transcendental));
        return cases;
    }

    /// <summary>Builds one case's operands: seeded F32 inputs (a distinct salt each) at a chosen scale and zeroed outputs.</summary>
    private sealed class Operands(int seed)
    {
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape, float scale = 1f)
        {
            Tensors.Inputs[name] = Seeded.F32(shape, seed, _salt++, scale);
            return this;
        }

        public Operands Out(string name, TensorShape shape)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape);
            return this;
        }
    }
}
