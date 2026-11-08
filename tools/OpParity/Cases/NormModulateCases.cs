using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Every normalization-plus-modulation the models apply, one case per caller parameter set, expressed through
/// <see cref="IBackend.NormModulate"/> and <see cref="IBackend.ChunkActivate"/> exactly as the callers build them: the DiT adaLN
/// forms (broadcast affine, row-indexed affine, LayerNorm and RMSNorm modulation, LTX-2's RMS modulation, Oasis's packed
/// adaLN), the plain non-affine LayerNorm, the channel-axis norms of the VAEs (Wan 2.2's L2 form, TRELLIS's channel
/// LayerNorm) and the four-way modulation split. The fixtures were recorded against the entry points these replaced. Activations are seeded F32, with the F16 and BF16 forms each backend
/// serves; modulation vectors, weights and biases stay F32 as the models allocate them.</summary>
public static class NormModulateCases
{
    public const string Family = "normmod";

    private const float Eps = 1e-6f;

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];
    private static readonly string[] Gpu = ["cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];

    /// <summary>Pure elementwise forms: the same products and sums on every backend.</summary>
    private static readonly Acceptance Exact = Acceptance.Exact;

    /// <summary>A norm reduces its row in an order that differs per backend and per kernel generation.</summary>
    private static readonly Acceptance Norm = Acceptance.Within(1e-5, 1e-4, "the row reduction order differs per backend and kernel");

    /// <summary>An F16 output is formed in F32 and rounded once at the end; a kernel that rounded an intermediate, or a host loop that
    /// computed in F16, lands one F16 step away.</summary>
    private static readonly Acceptance NormF16 = Acceptance.Within(4e-3, 1e-2, "an F16 output rounds once from the F32 row; the replaced path rounded differently");

    /// <summary>The channel-axis norms accumulate in F64 on the host and CUDA but in F32 on Vulkan.</summary>
    private static readonly Acceptance ChannelNorm = Acceptance.Within(2e-5, 1e-4, "Vulkan accumulates the channel reduction in F32");

    /// <summary>tanh differs by implementation between the host and the device libraries.</summary>
    private static readonly Acceptance Tanh = Acceptance.Within(1e-6, 1e-5, "tanh differs by implementation");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // Broadcast affine over the last dim: scale/shift [B, D] over [B, S, D]; scale-only; F16 and BF16 activations.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(be => be.Device.IsGpu, Gpu, bf16: true))
        {
            cases.Add(new(Family, "affine_lastdim_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 8)).In("scale", new TensorShape(2, 8), DType.F32).In("shift", new TensorShape(2, 8), DType.F32).Out("out", new TensorShape(2, 3, 8)).Tensors,
                (be, t) => be.AffineBroadcastLastDim(t.Outputs["out"], t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"]), serves, required, Exact));
            cases.Add(new(Family, "affine_lastdim_noshift_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 8)).In("scale", new TensorShape(2, 8), DType.F32).Out("out", new TensorShape(2, 3, 8)).Tensors,
                (be, t) => be.AffineBroadcastLastDim(t.Outputs["out"], t.Inputs["x"], t.Inputs["scale"], null), serves, required, Exact));
        }
        // Row-indexed affine: a small [modRows, D] table gathered per row, the 1 + folded in; shift optional.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(be => be.Device.IsCuda, Cuda, bf16: true))
        {
            cases.Add(new(Family, "affine_rowindexed_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(1, 4, 8)).In("scale", new TensorShape(2, 8), DType.F32).In("shift", new TensorShape(2, 8), DType.F32).Index("rows", [1, 0, 1, 1]).Out("out", new TensorShape(1, 4, 8)).Tensors,
                (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.None, 0f).WithModulation(ModulationSpec.RowIndexed(t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"], t.Inputs["rows"], ScaleMode.OnePlus))), serves, required, Exact));
            cases.Add(new(Family, "affine_rowindexed_noshift_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(1, 4, 8)).In("scale", new TensorShape(2, 8), DType.F32).Index("rows", [1, 0, 1, 1]).Out("out", new TensorShape(1, 4, 8)).Tensors,
                (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.None, 0f).WithModulation(ModulationSpec.RowIndexed(t.Inputs["x"], t.Inputs["scale"], null, t.Inputs["rows"], ScaleMode.OnePlus))), serves, required, Exact));
        }
        // Non-affine LayerNorm over the last dim.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(be => be.Device.IsGpu, Gpu, bf16: false))
        {
            cases.Add(new(Family, "layernorm_noaffine_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 8)).Out("out", new TensorShape(2, 3, 8)).Tensors,
                (be, t) => be.LayerNormNoAffine(t.Outputs["out"], t.Inputs["x"], Eps), serves, required, NormFor(dtype)));
        }
        // LayerNorm then adaLN: (1 + scale[b]) · norm(x) + shift[b]; F16; and an F32 row into an F16 output (CUDA).
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(be => be.Device.IsGpu, Gpu, bf16: false))
        {
            cases.Add(new(Family, "layernorm_modulate_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 8)).In("scale", new TensorShape(2, 8), DType.F32).In("shift", new TensorShape(2, 8), DType.F32).Out("out", new TensorShape(2, 3, 8)).Tensors,
                (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.LayerNorm, Eps).WithModulation(ModulationSpec.Broadcast(t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"], ScaleMode.OnePlus))), serves, required, NormFor(dtype)));
        }
        cases.Add(new(Family, "layernorm_modulate_f32_f16", seed => new Operands(seed).In("x", new TensorShape(2, 3, 8)).In("scale", new TensorShape(2, 8)).In("shift", new TensorShape(2, 8)).Out("out", new TensorShape(2, 3, 8), DType.F16).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.LayerNorm, Eps).WithModulation(ModulationSpec.Broadcast(t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"], ScaleMode.OnePlus))), be => be.Device.IsCuda, Cuda, NormF16));
        // Weighted RMSNorm then adaLN, scale/shift [B, D] or [D]; no shift; F16; mixed F32/F16 in either direction.
        cases.Add(new(Family, "rmsnorm_modulate_f32", seed => new Operands(seed).In("x", new TensorShape(2, 3, 8)).In("w", new TensorShape(8)).In("scale", new TensorShape(2, 8)).In("shift", new TensorShape(2, 8)).Out("out", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.RmsNorm, Eps).WithAffine(t.Inputs["w"], null).WithModulation(ModulationSpec.Broadcast(t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"], ScaleMode.OnePlus))), _ => true, All, Norm));
        cases.Add(new(Family, "rmsnorm_modulate_shared_noshift", seed => new Operands(seed).In("x", new TensorShape(2, 3, 8)).In("w", new TensorShape(8)).In("scale", new TensorShape(8)).Out("out", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.RmsNorm, Eps).WithAffine(t.Inputs["w"], null).WithModulation(ModulationSpec.Broadcast(t.Inputs["x"], t.Inputs["scale"], null, ScaleMode.OnePlus))), _ => true, All, Norm));
        foreach ((string name, DType inDtype, DType outDtype) in new[] { ("rmsnorm_modulate_f16", DType.F16, DType.F16), ("rmsnorm_modulate_f32_f16", DType.F32, DType.F16), ("rmsnorm_modulate_f16_f32", DType.F16, DType.F32) })
        {
            cases.Add(new(Family, name, seed => new Operands(seed, inDtype).In("x", new TensorShape(2, 3, 8)).In("w", new TensorShape(8), DType.F32).In("scale", new TensorShape(2, 8), DType.F32).In("shift", new TensorShape(2, 8), DType.F32).Out("out", new TensorShape(2, 3, 8), outDtype).Tensors,
                (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.RmsNorm, Eps).WithAffine(t.Inputs["w"], null).WithModulation(ModulationSpec.Broadcast(t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"], ScaleMode.OnePlus))), _ => true, All, NormFor(outDtype)));
        }
        // LTX-2: unweighted RMSNorm then (1 + scale[d]) and shift[d] shared by every row; F16 on the GPUs. The recorded "ones" input was
        // the unit weight the replaced entry took; the call no longer reads it.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in Activations(be => be.Device.IsGpu, Gpu, bf16: false))
        {
            cases.Add(new(Family, "ltx2_rms_modulate_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(3, 16)).Ones("ones", 16).In("scale", new TensorShape(16), DType.F32).In("shift", new TensorShape(16), DType.F32).Out("out", new TensorShape(3, 16)).Tensors,
                (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.RmsNorm, Eps).WithModulation(ModulationSpec.Broadcast(t.Inputs["x"], t.Inputs["scale"], t.Inputs["shift"], ScaleMode.OnePlus))), serves, required, NormFor(dtype)));
        }
        // Oasis: LayerNorm then (1 + scale) and shift read from a packed modulation row [frames, modStride] at offsets, one row per sp tokens.
        cases.Add(new(Family, "oasis_adaln_f32", seed => new Operands(seed).In("x", new TensorShape(6, 8)).In("mod", new TensorShape(2, 48)).Out("out", new TensorShape(6, 8)).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Rows(t.Outputs["out"], t.Inputs["x"], NormKind.LayerNorm, Eps).WithModulation(ModulationSpec.Packed(t.Inputs["mod"], 48, scaleOffset: 8, shiftOffset: 0, rowsPerMod: 3, ScaleMode.OnePlus))), _ => true, All, Norm));
        // Wan 2.2 channel RMS norm: x · sqrt(C) / max(‖x‖₂, eps) · gamma over the channel axis of [B, C, T, H, W]; no gamma; BF16 (CUDA).
        cases.Add(new(Family, "wan_rms_channel_f32", seed => new Operands(seed).In("x", new TensorShape([2, 4, 3, 2, 2])).In("gamma", new TensorShape(4)).Out("out", new TensorShape([2, 4, 3, 2, 2])).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Channel(t.Outputs["out"], t.Inputs["x"], NormKind.L2Norm, 1e-12f).WithAffine(t.Inputs["gamma"], null)), _ => true, All, ChannelNorm));
        cases.Add(new(Family, "wan_rms_channel_nogamma", seed => new Operands(seed).In("x", new TensorShape(1, 4, 2, 3)).Out("out", new TensorShape(1, 4, 2, 3)).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Channel(t.Outputs["out"], t.Inputs["x"], NormKind.L2Norm, 1e-8f)), _ => true, All, ChannelNorm));
        cases.Add(new(Family, "wan_rms_channel_bf16", seed => new Operands(seed, DType.BF16).In("x", new TensorShape([2, 4, 3, 2, 2])).In("gamma", new TensorShape(4), DType.F32).Out("out", new TensorShape([2, 4, 3, 2, 2])).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Channel(t.Outputs["out"], t.Inputs["x"], NormKind.L2Norm, 1e-12f).WithAffine(t.Inputs["gamma"], null)), be => be.Device.IsCuda, Cuda, ChannelNorm));
        // TRELLIS channel LayerNorm over [N, C, D, H, W] with a [C] affine (the host default served every backend).
        cases.Add(new(Family, "channel_layernorm3d_f32", seed => new Operands(seed).In("x", new TensorShape([1, 4, 2, 2, 2])).In("w", new TensorShape(4)).In("b", new TensorShape(4)).Out("out", new TensorShape([1, 4, 2, 2, 2])).Tensors,
            (be, t) => be.NormModulate(NormModulateCall.Channel(t.Outputs["out"], t.Inputs["x"], NormKind.LayerNorm, 1e-5f).WithAffine(t.Inputs["w"], t.Inputs["b"])), _ => true, All, ChannelNorm));
        // Four-way modulation split: proj [B, 4D] into 1 + x for the scales and tanh(x) for the gates.
        cases.Add(new(Family, "modulation_split4_f32", seed => new Operands(seed).In("proj", new TensorShape(2, 32)).Out("scaleMsa", new TensorShape(2, 8)).Out("gateMsa", new TensorShape(2, 8)).Out("scaleMlp", new TensorShape(2, 8)).Out("gateMlp", new TensorShape(2, 8)).Tensors,
            (be, t) => be.ChunkActivate([t.Outputs["scaleMsa"], t.Outputs["gateMsa"], t.Outputs["scaleMlp"], t.Outputs["gateMlp"]], t.Inputs["proj"], [ChunkActivation.OnePlus, ChunkActivation.Tanh, ChunkActivation.OnePlus, ChunkActivation.Tanh]), _ => true, All, Tanh));
        return cases;
    }

    /// <summary>The acceptance of a row norm whose output is stored as <paramref name="outDtype"/>.</summary>
    private static Acceptance NormFor(DType outDtype) => outDtype == DType.F16 ? NormF16 : Norm;

    /// <summary>The activation dtypes a case runs in: F32 everywhere, F16 where <paramref name="f16"/> says a backend serves it, and BF16 on CUDA when <paramref name="bf16"/> is set.</summary>
    private static IEnumerable<(string Suffix, DType DType, Func<IBackend, bool> Serves, string[] Required)> Activations(Func<IBackend, bool> f16, string[] f16Required, bool bf16)
    {
        yield return ("f32", DType.F32, _ => true, All);
        yield return ("f16", DType.F16, f16, f16Required);
        if (bf16) yield return ("bf16", DType.BF16, be => be.Device.IsCuda, Cuda);
    }

    /// <summary>Builds one case's operands: seeded inputs (a distinct salt each), zeroed outputs, index rows and a ones vector.</summary>
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

        public Operands Index(string name, ReadOnlySpan<int> rows)
        {
            Tensor t = Seeded.Zeros(new TensorShape(rows.Length), DType.I32);
            rows.CopyTo(t.AsSpan<int>());
            Tensors.Inputs[name] = t;
            return this;
        }

        public Operands Ones(string name, int width)
        {
            Tensor t = Seeded.Zeros(new TensorShape(width), DType.F32);
            t.AsSpan<float>().Fill(1f);
            Tensors.Inputs[name] = t;
            return this;
        }

        public Operands Out(string name, TensorShape shape, DType? outputDtype = null)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape, outputDtype ?? _dtype);
            return this;
        }
    }
}
