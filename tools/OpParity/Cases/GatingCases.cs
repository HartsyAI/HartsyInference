using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Backends.Routing;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Every gated activation and gate the models apply, one case per caller parameter set: the gated linear units
/// (SwiGLU and GeGLU over the leading gate half, GEGLU with the tanh or erf gate over the trailing half, MOSS-Image's
/// SiTU-GLU, the Q8-emitting decode form), the group gates (token masking, LTX-2's doubled-sigmoid head gate, the
/// MoE gated accumulate), the fused linear epilogues (GELU, head-gated projection) and the dense MoE top-k gate table, expressed
/// through <see cref="IBackend.Glu"/>, <see cref="IBackend.GroupScale"/>, <see cref="IBackend.LinearFused"/> and the dense output
/// of <see cref="IBackend.MoeRoute"/> exactly as the callers build them. The fixtures were recorded against the entry points these
/// replaced. Activations are seeded F32 with the F16 and BF16 forms each backend serves.</summary>
public static class GatingCases
{
    public const string Family = "gating";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];
    private static readonly string[] Gpu = ["cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];
    private static readonly string[] Vulkan = ["vulkan"];

    /// <summary>A pure multiply by a stored gate: the same products on every backend.</summary>
    private static readonly Acceptance Exact = Acceptance.Exact;

    /// <summary>sigmoid, tanh, erf and exp follow each backend's math library (CUDA's fast-math intrinsics on the device).</summary>
    private static readonly Acceptance Transcendental = Acceptance.Within(1e-5, 1e-4, "the activation's exp, tanh or erf follows each backend's math library");

    /// <summary>The replaced F16 <c>[gate | up]</c> entry had no Vulkan kernel: its host default sliced the F16 halves through the F32 slice
    /// shader, so the bytes it recorded there are not a reference; the F16 shader is judged against the host reference at one F16 step.</summary>
    private static readonly Acceptance GluF16Host = Acceptance.AgainstHost(2e-3, 1e-2, "the replaced entry's Vulkan F16 path sliced F16 halves through an F32 shader; the F16 output is judged against the reference");

    /// <summary>The accumulate contracts the gate multiply-add into an FMA on a device where the replaced entry ran a host loop.</summary>
    private static readonly Acceptance Fma = Acceptance.Within(1e-6, 1e-5, "the gate multiply-add contracts into an FMA where the replaced entry ran a host loop");

    /// <summary>A GEMM reduces in a backend-specific order before the epilogue.</summary>
    private static readonly Acceptance Gemm = Acceptance.Within(2e-5, 1e-4, "the GEMM reduction order differs per backend");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // Gated FFN over [gate | up]: act(gate) · up, SiLU or GELU-tanh; F16 on the GPUs, BF16 on CUDA.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(bf16: true))
        {
            cases.Add(new(Family, "glu_silu_first_" + suffix, seed => new Operands(seed, dtype).In("gateUp", new TensorShape(1, 3, 16)).Out("out", new TensorShape(1, 3, 8)).Tensors,
                (be, t) => be.Glu(t.Outputs["out"], t.Inputs["gateUp"], GluSpec.SwiGlu), serves, required, dtype == DType.F16 ? GluF16Host : Transcendental));
            cases.Add(new(Family, "glu_gelu_first_" + suffix, seed => new Operands(seed, dtype).In("gateUp", new TensorShape(1, 3, 16)).Out("out", new TensorShape(1, 3, 8)).Tensors,
                (be, t) => be.Glu(t.Outputs["out"], t.Inputs["gateUp"], GluSpec.GeGluFirst), serves, required, dtype == DType.F16 ? GluF16Host : Transcendental));
            // GEGLU over [value | gate] with the tanh GELU.
            cases.Add(new(Family, "geglu_tanh_second_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 8)).Out("out", new TensorShape(2, 3, 4)).Tensors,
                (be, t) => be.Glu(t.Outputs["out"], t.Inputs["x"], GluSpec.GeGlu), serves, required, Transcendental));
        }
        // GEGLU with the exact (erf) GELU gate: F32 everywhere, F16 on Vulkan.
        cases.Add(new(Family, "geglu_erf_second_f32", seed => new Operands(seed).In("proj", new TensorShape(1, 3, 8)).Out("out", new TensorShape(1, 3, 4)).Tensors,
            (be, t) => be.Glu(t.Outputs["out"], t.Inputs["proj"], GluSpec.GeGluErf), _ => true, All, Transcendental));
        cases.Add(new(Family, "geglu_erf_second_f16", seed => new Operands(seed, DType.F16).In("proj", new TensorShape(1, 3, 8)).Out("out", new TensorShape(1, 3, 4)).Tensors,
            (be, t) => be.Glu(t.Outputs["out"], t.Inputs["proj"], GluSpec.GeGluErf), be => be.Device.IsVulkan, Vulkan, Transcendental));
        // SiTU-GLU over [u | g] (MOSS-Image), F32 and F16 everywhere.
        foreach ((string suffix, DType dtype) in new[] { ("f32", DType.F32), ("f16", DType.F16) })
        {
            cases.Add(new(Family, "situ_second_" + suffix, seed => new Operands(seed, dtype).In("pair", new TensorShape(1, 3, 8)).Out("out", new TensorShape(1, 3, 4)).Tensors,
                (be, t) => be.Glu(t.Outputs["out"], t.Inputs["pair"], GluSpec.Situ), _ => true, All, Transcendental));
        }
        // The decode-step SwiGLU that may emit a Q8_1 sidecar: one row, a width of whole 32-blocks; the output is the plain GLU's.
        cases.Add(new(Family, "glu_silu_emit_q8", seed => new Operands(seed).In("gateUp", new TensorShape(1, 1, 128)).Out("out", new TensorShape(1, 1, 64)).Tensors,
            (be, t) => be.GluEmitQ8(t.Outputs["out"], t.Inputs["gateUp"], GluSpec.SwiGlu), _ => true, All, Transcendental));

        // Token masking: each row scaled by its own factor.
        cases.Add(new(Family, "mask_rows_f32", seed => new Operands(seed).In("x", new TensorShape(2, 3, 8)).In("mask", new TensorShape(6)).Out("out", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => be.GroupScale(t.Outputs["out"], t.Inputs["x"], t.Inputs["mask"], GroupScaleSpec.PerRow()), _ => true, All, Exact));
        // LTX-2 head gate, in place: each head's slice scaled by 2·sigmoid of its logit; F16 on CUDA.
        cases.Add(new(Family, "head_gate_f32", seed => new Operands(seed).InPlace("x", new TensorShape(3, 16)).In("logits", new TensorShape(3, 2)).Tensors,
            (be, t) => { Prime(t, "x"); be.GroupScale(t.Outputs["x"], t.Outputs["x"], t.Inputs["logits"], GroupScaleSpec.PerGroup(8, 2, GateFunction.TwoSigmoid)); }, _ => true, All, Transcendental));
        cases.Add(new(Family, "head_gate_f16", seed => new Operands(seed, DType.F16).InPlace("x", new TensorShape(3, 16)).In("logits", new TensorShape(3, 2)).Tensors,
            (be, t) => { Prime(t, "x"); be.GroupScale(t.Outputs["x"], t.Outputs["x"], t.Inputs["logits"], GroupScaleSpec.PerGroup(8, 2, GateFunction.TwoSigmoid)); }, be => be.Device.IsCuda, Cuda, Transcendental));
        // MoE gated accumulate, in place: out += gate[row, expert] · value over a dense [rows, experts] gate table with zeros.
        cases.Add(new(Family, "row_gated_accumulate_f32", seed => new Operands(seed).InPlace("acc", new TensorShape(1, 3, 8)).In("value", new TensorShape(1, 3, 8)).Gates("gate", 3, 4).Tensors,
            (be, t) => { Prime(t, "acc"); be.GroupScale(t.Outputs["acc"], t.Inputs["value"], t.Inputs["gate"], GroupScaleSpec.Column(4, 1, accumulate: true)); }, _ => true, All, Fma));

        // Fused linear epilogues: GELU after the projection; the doubled-sigmoid head gate before it. The replaced entry gated the input in
        // place, so the recorded "x" output is the input as it was left; the fused call scales a copy and leaves "x" at its primed value,
        // which the fixture compares against the gated bytes — a known disagreement judged against the host reference.
        cases.Add(new(Family, "linear_gelu_f32", seed => new Operands(seed).In("x", new TensorShape(3, 4)).In("w", new TensorShape(8, 4)).In("b", new TensorShape(8)).Out("out", new TensorShape(3, 8)).Tensors,
            (be, t) => be.LinearFused(t.Outputs["out"], t.Inputs["x"], t.Inputs["w"], t.Inputs["b"], LinearFusion.Gelu), _ => true, All, Gemm));
        cases.Add(new(Family, "linear_head_gated_f32", seed => new Operands(seed).InPlace("x", new TensorShape(3, 16)).In("w", new TensorShape(8, 16)).In("logits", new TensorShape(3, 2)).Out("out", new TensorShape(3, 8)).Tensors,
            (be, t) => { Prime(t, "x"); be.LinearFused(t.Outputs["out"], t.Outputs["x"], t.Inputs["w"], null, LinearFusion.HeadGated(t.Inputs["logits"], 2, 8)); }, _ => true, All,
            Acceptance.AgainstHost(2e-5, 1e-4, "the fused projection no longer gates its input in place, so the recorded gated input differs by design")));

        // Dense MoE top-k gate table (HiDream): softmax, top-2 of 4, renormalized, zeros elsewhere.
        cases.Add(new(Family, "moe_topk_gate_f32", seed => new Operands(seed).In("logits", new TensorShape(2, 3, 4)).Out("weights", new TensorShape(2, 3, 4)).Tensors,
            (be, t) => RouteDense(be, t.Outputs["weights"], t.Inputs["logits"], 2), _ => true, All, Transcendental));
        return cases;
    }

    /// <summary>The dense top-k gate table of the HiDream MoE: softmax, top-k, renormalized with the 1e-20 floor, through the route's dense output.</summary>
    private static void RouteDense(IBackend be, Tensor weights, Tensor logits, int topK)
    {
        int experts = (int)logits.Shape[logits.Shape.Rank - 1];
        long tokens = logits.ElementCount / experts;
        using Tensor idx = Seeded.Zeros(new TensorShape(tokens, topK), DType.I32);
        using Tensor w = Seeded.Zeros(new TensorShape(tokens, topK), DType.F32);
        be.MoeRoute(idx, w, logits, new MoeRouteArgs(experts, topK, MoeRouteScoring.Softmax, Renormalize: true, RenormEpsilon: 1e-20f), dense: weights);
    }

    /// <summary>The activation dtypes a case runs in: F32 everywhere, F16 on the GPUs, BF16 on CUDA when asked.</summary>
    private static IEnumerable<(string Suffix, DType DType, Func<IBackend, bool> Serves, string[] Required)> Activations(bool bf16)
    {
        yield return ("f32", DType.F32, _ => true, All);
        yield return ("f16", DType.F16, be => be.Device.IsGpu, Gpu);
        if (bf16) yield return ("bf16", DType.BF16, be => be.Device.IsCuda, Cuda);
    }

    /// <summary>Copies the recorded contents of in-place operand <paramref name="name"/> (input <c>{name}_init</c>) into its output before the op runs.</summary>
    private static void Prime(OpTensors t, string name) => t.Inputs[name + "_init"].AsReadOnlySpan<byte>().CopyTo(t.Outputs[name].AsSpan<byte>());

    /// <summary>Builds one case's operands: seeded inputs (a distinct salt each), zeroed outputs, in-place operands recorded as an input and written as an output, and a sparse gate table.</summary>
    private sealed class Operands(int seed, DType? dtype = null)
    {
        private readonly DType _dtype = dtype ?? DType.F32;
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape, DType? inputDtype = null)
        {
            Tensors.Inputs[name] = Seeded.Of(shape, inputDtype ?? _dtype, seed, _salt++);
            return this;
        }

        public Operands Out(string name, TensorShape shape)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape, _dtype);
            return this;
        }

        public Operands InPlace(string name, TensorShape shape)
        {
            Tensors.Inputs[name + "_init"] = Seeded.Of(shape, _dtype, seed, _salt++);
            Tensors.Outputs[name] = Seeded.Zeros(shape, _dtype);
            return this;
        }

        /// <summary>A dense <c>[rows, experts]</c> gate table with every other entry zero, as a top-k gate writes it.</summary>
        public Operands Gates(string name, int rows, int experts)
        {
            Tensor t = Seeded.Of(new TensorShape(rows, experts), DType.F32, seed, _salt++);
            Span<float> g = t.AsSpan<float>();
            for (int i = 0; i < g.Length; i++) if ((i & 1) == 0) g[i] = 0f;
            Tensors.Inputs[name] = t;
            return this;
        }
    }
}
