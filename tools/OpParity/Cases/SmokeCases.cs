using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Small ops every backend serves, used to prove the record/check round trip on a machine without a GPU.</summary>
public static class SmokeCases
{
    public const string Family = "smoke";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];

    public static IReadOnlyList<OpCase> Cases() =>
    [
        new(Family, "add_f32_4x256", seed => Binary(seed, 4, 256), (be, t) => be.Add(t.Outputs["out"], t.Inputs["a"], t.Inputs["b"]), _ => true, All, Acceptance.Exact),
        new(Family, "transpose2d_f32_3x16x32", seed =>
            {
                OpTensors t = new();
                t.Inputs["x"] = Seeded.F32(new TensorShape(3, 16, 32), seed, 1);
                t.Outputs["out"] = Seeded.Zeros(new TensorShape(3, 32, 16));
                return t;
            },
            (be, t) => be.Transpose2D(t.Outputs["out"], t.Inputs["x"], 16, 32), _ => true, All, Acceptance.Exact),
        new(Family, "layernorm_f32_8x64", seed =>
            {
                OpTensors t = new();
                t.Inputs["x"] = Seeded.F32(new TensorShape(8, 64), seed, 1);
                t.Inputs["w"] = Seeded.F32(new TensorShape(64), seed, 2, 0.5f);
                t.Inputs["b"] = Seeded.F32(new TensorShape(64), seed, 3, 0.1f);
                t.Outputs["out"] = Seeded.Zeros(new TensorShape(8, 64));
                return t;
            },
            (be, t) => be.LayerNorm(t.Outputs["out"], t.Inputs["x"], t.Inputs["w"], t.Inputs["b"], 1e-5f), _ => true, All,
            Acceptance.Within(1e-5, 1e-4, "reduction order differs between the SIMD, CUDA and Vulkan kernels")),
    ];

    private static OpTensors Binary(int seed, int rows, int cols)
    {
        OpTensors t = new();
        t.Inputs["a"] = Seeded.F32(new TensorShape(rows, cols), seed, 1);
        t.Inputs["b"] = Seeded.F32(new TensorShape(rows, cols), seed, 2);
        t.Outputs["out"] = Seeded.Zeros(new TensorShape(rows, cols));
        return t;
    }
}
