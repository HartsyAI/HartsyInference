using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>3D neighborhood attention as the LTX-2.5 video decoder calls it: every query attends a fixed
/// <c>kT×kH×kW</c> window that slides inward at the borders, over <c>[batch, T, H, W, heads, headDim]</c> F32 grids, with the
/// output either the same grid or its flat <c>[tokens, heads·headDim]</c> form, the scale folded into q or applied to the scores,
/// and a window longer than an axis collapsing to that axis. The fixtures were recorded against the entry point
/// <see cref="IBackend.NeighborhoodAttention"/> replaced.</summary>
public static class NeighborhoodCases
{
    public const string Family = "neighborhood";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];

    /// <summary>The softmax's exp and the window's accumulation order follow each backend.</summary>
    private static readonly Acceptance Softmax = Acceptance.Within(1e-5, 1e-4, "the softmax's exp and the window's accumulation order follow each backend");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];
        cases.Add(new(Family, "na3d_grid_f32", seed => Qkv(seed, 1, 3, 4, 4, 2, 8).Out("out", new TensorShape([1, 3, 4, 4, 2, 8])).Tensors,
            (be, t) => be.NeighborhoodAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], new NeighborhoodWindow(2, 3, 3), 1f), _ => true, All, Softmax));
        // The flat output form the projection that follows consumes directly.
        cases.Add(new(Family, "na3d_flat_f32", seed => Qkv(seed, 1, 3, 4, 4, 2, 8).Out("out", new TensorShape(48, 16)).Tensors,
            (be, t) => be.NeighborhoodAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], new NeighborhoodWindow(2, 3, 3), 1f), _ => true, All, Softmax));
        // A window longer than the temporal axis collapses to it; the scale is applied to the scores.
        cases.Add(new(Family, "na3d_clamped_scaled_f32", seed => Qkv(seed, 2, 2, 3, 5, 1, 4).Out("out", new TensorShape([2, 2, 3, 5, 1, 4])).Tensors,
            (be, t) => be.NeighborhoodAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], new NeighborhoodWindow(5, 2, 2), 0.35f), _ => true, All, Softmax));
        // The decoder's head width, which the query-tiled CUDA kernel serves.
        cases.Add(new(Family, "na3d_head64_f32", seed => Qkv(seed, 1, 2, 8, 8, 1, 64).Out("out", new TensorShape(128, 64)).Tensors,
            (be, t) => be.NeighborhoodAttention(t.Outputs["out"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], new NeighborhoodWindow(3, 5, 5), 1f), _ => true, All, Softmax));
        return cases;
    }

    private static Operands Qkv(int seed, int batch, int t, int h, int w, int heads, int headDim)
    {
        TensorShape shape = new([batch, t, h, w, heads, headDim]);
        return new Operands(seed).In("q", shape).In("k", shape).In("v", shape);
    }

    /// <summary>Builds one case's operands: seeded F32 inputs (a distinct salt each) and zeroed outputs.</summary>
    private sealed class Operands(int seed)
    {
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape)
        {
            Tensors.Inputs[name] = Seeded.F32(shape, seed, _salt++);
            return this;
        }

        public Operands Out(string name, TensorShape shape)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape);
            return this;
        }
    }
}
