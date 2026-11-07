using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Every gated activation and gate the models apply, one case per caller parameter set: the gated linear units
/// (SwiGLU and GeGLU over the leading gate half, GEGLU with the tanh or erf gate over the trailing half, MOSS-Image's
/// SiTU-GLU, the Q8-emitting decode form), the group gates (token masking, LTX-2's doubled-sigmoid head gate, the
/// MoE gated accumulate), the fused linear epilogues (GELU, head-gated projection) and the dense MoE top-k gate table.
/// Activations are seeded F32 with the F16 and BF16 forms each backend serves.</summary>
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

    /// <summary>A GEMM reduces in a backend-specific order before the epilogue.</summary>
    private static readonly Acceptance Gemm = Acceptance.Within(2e-5, 1e-4, "the GEMM reduction order differs per backend");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // Gated FFN over [gate | up]: act(gate) · up, SiLU or GELU-tanh; F16 on the GPUs, BF16 on CUDA.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(bf16: true))
        {
            cases.Add(new(Family, "glu_silu_first_" + suffix, seed => new Operands(seed, dtype).In("gateUp", new TensorShape(1, 3, 16)).Out("out", new TensorShape(1, 3, 8)).Tensors,
                (be, t) => be.GluActivate(t.Outputs["out"], t.Inputs["gateUp"], 8, false), serves, required, Transcendental));
            cases.Add(new(Family, "glu_gelu_first_" + suffix, seed => new Operands(seed, dtype).In("gateUp", new TensorShape(1, 3, 16)).Out("out", new TensorShape(1, 3, 8)).Tensors,
                (be, t) => be.GluActivate(t.Outputs["out"], t.Inputs["gateUp"], 8, true), serves, required, Transcendental));
            // GEGLU over [value | gate] with the tanh GELU.
            cases.Add(new(Family, "geglu_tanh_second_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 8)).Out("out", new TensorShape(2, 3, 4)).Tensors,
                (be, t) => be.GeGlu(t.Outputs["out"], t.Inputs["x"]), serves, required, Transcendental));
        }
        // GEGLU with the exact (erf) GELU gate: F32 everywhere, F16 on Vulkan.
        cases.Add(new(Family, "geglu_erf_second_f32", seed => new Operands(seed).In("proj", new TensorShape(1, 3, 8)).Out("out", new TensorShape(1, 3, 4)).Tensors,
            (be, t) => be.GegluErf(t.Outputs["out"], t.Inputs["proj"], 3, 4), _ => true, All, Transcendental));
        cases.Add(new(Family, "geglu_erf_second_f16", seed => new Operands(seed, DType.F16).In("proj", new TensorShape(1, 3, 8)).Out("out", new TensorShape(1, 3, 4)).Tensors,
            (be, t) => be.GegluErf(t.Outputs["out"], t.Inputs["proj"], 3, 4), be => be.Device.IsVulkan, Vulkan, Transcendental));
        // SiTU-GLU over [u | g] (MOSS-Image), F32 and F16 everywhere.
        foreach ((string suffix, DType dtype) in new[] { ("f32", DType.F32), ("f16", DType.F16) })
        {
            cases.Add(new(Family, "situ_second_" + suffix, seed => new Operands(seed, dtype).In("pair", new TensorShape(1, 3, 8)).Out("out", new TensorShape(1, 3, 4)).Tensors,
                (be, t) => be.SituGlu(t.Outputs["out"], t.Inputs["pair"]), _ => true, All, Transcendental));
        }
        // The decode-step SwiGLU that may emit a Q8_1 sidecar: one row, a width of whole 32-blocks; the output is the plain GLU's.
        cases.Add(new(Family, "glu_silu_emit_q8", seed => new Operands(seed).In("gateUp", new TensorShape(1, 1, 128)).Out("out", new TensorShape(1, 1, 64)).Tensors,
            (be, t) => be.GluActivateEmitQ8(t.Outputs["out"], t.Inputs["gateUp"], 64, false), _ => true, All, Transcendental));

        // Token masking: each row scaled by its own factor.
        cases.Add(new(Family, "mask_rows_f32", seed => new Operands(seed).In("x", new TensorShape(2, 3, 8)).In("mask", new TensorShape(6)).Out("out", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => be.MaskRows(t.Outputs["out"], t.Inputs["x"], t.Inputs["mask"]), _ => true, All, Exact));
        // LTX-2 head gate, in place: each head's slice scaled by 2·sigmoid of its logit; F16 on CUDA.
        cases.Add(new(Family, "head_gate_f32", seed => new Operands(seed).InPlace("x", new TensorShape(3, 16)).In("logits", new TensorShape(3, 2)).Tensors,
            (be, t) => { Prime(t, "x"); be.Ltx2HeadGate(t.Outputs["x"], t.Inputs["logits"], 3, 2, 8); }, _ => true, All, Transcendental));
        cases.Add(new(Family, "head_gate_f16", seed => new Operands(seed, DType.F16).InPlace("x", new TensorShape(3, 16)).In("logits", new TensorShape(3, 2)).Tensors,
            (be, t) => { Prime(t, "x"); be.Ltx2HeadGate(t.Outputs["x"], t.Inputs["logits"], 3, 2, 8); }, be => be.Device.IsCuda, Cuda, Transcendental));
        // MoE gated accumulate, in place: out += gate[row, expert] · value over a dense [rows, experts] gate table with zeros.
        cases.Add(new(Family, "row_gated_accumulate_f32", seed => new Operands(seed).InPlace("acc", new TensorShape(1, 3, 8)).In("value", new TensorShape(1, 3, 8)).Gates("gate", 3, 4).Tensors,
            (be, t) => { Prime(t, "acc"); be.RowGatedAccumulate(t.Outputs["acc"], t.Inputs["value"], t.Inputs["gate"], 4, 1); }, _ => true, All, Exact));

        // Fused linear epilogues: GELU after the projection; the doubled-sigmoid head gate before it (the input is gated in place by the composed form).
        cases.Add(new(Family, "linear_gelu_f32", seed => new Operands(seed).In("x", new TensorShape(3, 4)).In("w", new TensorShape(8, 4)).In("b", new TensorShape(8)).Out("out", new TensorShape(3, 8)).Tensors,
            (be, t) => be.LinearGelu(t.Outputs["out"], t.Inputs["x"], t.Inputs["w"], t.Inputs["b"]), _ => true, All, Gemm));
        cases.Add(new(Family, "linear_head_gated_f32", seed => new Operands(seed).InPlace("x", new TensorShape(3, 16)).In("w", new TensorShape(8, 16)).In("logits", new TensorShape(3, 2)).Out("out", new TensorShape(3, 8)).Tensors,
            (be, t) => { Prime(t, "x"); be.LinearHeadGated(t.Outputs["out"], t.Outputs["x"], t.Inputs["w"], null, t.Inputs["logits"], 2, 8); }, _ => true, All, Gemm));

        // Dense MoE top-k gate table (HiDream): softmax, top-2 of 4, renormalized, zeros elsewhere.
        cases.Add(new(Family, "moe_topk_gate_f32", seed => new Operands(seed).In("logits", new TensorShape(2, 3, 4)).Out("weights", new TensorShape(2, 3, 4)).Tensors,
            (be, t) => be.MoeTopKGate(t.Outputs["weights"], t.Inputs["logits"], 2), _ => true, All, Transcendental));
        return cases;
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
