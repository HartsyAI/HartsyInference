using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Guidance;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Every guided update the samplers apply, one case per caller parameter set: the in-place guided Euler step in its
/// plain, globally renormalized (Lance) and per-row normalized (Lumina 2) forms, the logit blend the autoregressive audio
/// decoders run as a zero-latent step, and MiniMax Music 3's guided top-K logits. The fixtures were recorded against
/// the entry points <see cref="IBackend.GuidedEulerStep"/> and <see cref="IBackend.TopKGuidedLogits"/> replaced. Every operand is F32,
/// the only dtype these steps take.</summary>
public static class GuidanceCases
{
    public const string Family = "guidance";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];

    /// <summary>The update is a handful of multiplies and adds in a fixed order; the norms sum in F64 sequentially on the host and
    /// through the same block reduction on CUDA.</summary>
    private static readonly Acceptance Exact = Acceptance.Exact;

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // The plain step, g·cond + (1−g)·uncond folded into z += v·delta: the flow-match pipelines, with a negative delta where the
        // model predicts −v.
        cases.Add(new(Family, "euler_lerp_f32", seed => new Operands(seed).InPlace("z", new TensorShape(2, 3, 8)).In("cond", new TensorShape(2, 3, 8)).In("uncond", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["cond"], t.Inputs["uncond"], GuidedEulerSpec.Lerp(3.5f, -0.125f)); }, _ => true, All, Exact));
        // No guidance: both predictions are the same tensor and g = 1 (Flux, Krea 2, the cond-only Lance and Lumina steps).
        cases.Add(new(Family, "euler_lerp_shared_f32", seed => new Operands(seed).InPlace("z", new TensorShape(1, 4, 16)).In("v", new TensorShape(1, 4, 16)).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["v"], t.Inputs["v"], GuidedEulerSpec.Lerp(1f, 0.2f)); }, _ => true, All, Exact));
        // The logit blend: a zeroed accumulator and delta 1 leave g·cond + (1−g)·uncond (CSM, MiniMax Music 3's residual draw).
        cases.Add(new(Family, "euler_lerp_logits_f32", seed => new Operands(seed).InPlace("z", new TensorShape(1, 1, 32), zero: true).In("cond", new TensorShape(1, 1, 32), 4f).In("uncond", new TensorShape(1, 1, 32), 4f).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["cond"], t.Inputs["uncond"], GuidedEulerSpec.Lerp(2f, 1f)); }, _ => true, All, Exact));

        // Globally renormalized (Lance): uncond + g·(cond − uncond), scaled by clamp(‖cond‖/(‖guided‖ + 1e-8), min, 1) over the whole tensor.
        cases.Add(new(Family, "euler_global_norm_f32", seed => new Operands(seed).InPlace("z", new TensorShape(2, 3, 8)).In("cond", new TensorShape(2, 3, 8)).In("uncond", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["cond"], t.Inputs["uncond"], GuidedEulerSpec.GlobalNorm(4f, -0.1f, 0.5f)); }, _ => true, All, Exact));
        // A larger tensor than one block reduces, so the CUDA partial sums take more than one block.
        cases.Add(new(Family, "euler_global_norm_wide_f32", seed => new Operands(seed).InPlace("z", new TensorShape(1, 24, 64)).In("cond", new TensorShape(1, 24, 64)).In("uncond", new TensorShape(1, 24, 64)).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["cond"], t.Inputs["uncond"], GuidedEulerSpec.GlobalNorm(2.5f, -0.05f, 0.1f)); }, _ => true, All, Exact));
        // Per-row normalized (Lumina 2): the same blend scaled by ‖cond‖/(‖guided‖ + eps) over each last-dimension row.
        cases.Add(new(Family, "euler_row_norm_f32", seed => new Operands(seed).InPlace("z", new TensorShape(2, 3, 8)).In("cond", new TensorShape(2, 3, 8)).In("uncond", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["cond"], t.Inputs["uncond"], GuidedEulerSpec.RowNorm(4f, -0.1f)); }, _ => true, All, Exact));
        cases.Add(new(Family, "euler_row_norm_wide_f32", seed => new Operands(seed).InPlace("z", new TensorShape(1, 5, 320)).In("cond", new TensorShape(1, 5, 320)).In("uncond", new TensorShape(1, 5, 320)).Tensors,
            (be, t) => { Prime(t, "z"); be.GuidedEulerStep(t.Outputs["z"], t.Inputs["cond"], t.Inputs["uncond"], GuidedEulerSpec.RowNorm(3f, 0.07f, 1e-6f)); }, _ => true, All, Exact));

        // Guided top-K logits (MiniMax Music 3): the conditional row's top-K keep uncond + (cond − uncond)·g, the rest become −inf.
        // CUDA only on the baseline, so the other backends are judged against the host reference; the −inf entries are the contract.
        cases.Add(new(Family, "topk_guided_logits_f32", seed => NonFinite(new Operands(seed).In("cond", new TensorShape(1, 40), 3f).In("uncond", new TensorShape(1, 40), 3f).Out("out", new TensorShape(1, 40)).Tensors, "out"),
            (be, t) => be.TopKGuidedLogits(t.Outputs["out"], t.Inputs["cond"], t.Inputs["uncond"], 6, 1.5f), _ => true, All, Exact,
            t => TopKGuidedReference.Apply(t.Outputs["out"], t.Inputs["cond"], t.Inputs["uncond"], 6, 1.5f)));
        return cases;
    }

    private static void Prime(OpTensors t, string name) => t.Inputs[name + "_init"].AsReadOnlySpan<byte>().CopyTo(t.Outputs[name].AsSpan<byte>());

    private static OpTensors NonFinite(OpTensors t, string output)
    {
        t.MayBeNonFinite.Add(output);
        return t;
    }

    /// <summary>Builds one case's operands: seeded inputs (a distinct salt each) at a chosen scale, zeroed outputs and in-place operands
    /// recorded as an input and written as an output.</summary>
    private sealed class Operands(int seed)
    {
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape, float scale = 1f)
        {
            Tensors.Inputs[name] = Seeded.F32(shape, seed, _salt++, scale);
            return this;
        }

        public Operands InPlace(string name, TensorShape shape, bool zero = false)
        {
            Tensors.Inputs[name + "_init"] = zero ? Seeded.Zeros(shape) : Seeded.F32(shape, seed, _salt++);
            Tensors.Outputs[name] = Seeded.Zeros(shape);
            return this;
        }

        public Operands Out(string name, TensorShape shape)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape);
            return this;
        }
    }
}
