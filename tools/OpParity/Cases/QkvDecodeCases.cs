using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>The graph-decode QKV epilogues as <c>GenericTransformer.ForwardGraphDecodeStep</c> calls them: one token's
/// projection output split into q, k and v, optionally QK-normed per head or over the full width, rotated by the row
/// the device position buffer names, q emitted head-major and k, v scattered into their caches at that position. One
/// case per packing × norm × pairing, and the F16 cache forms CUDA serves. The caches are in-place operands, so the
/// recorded contents are copied into them before the op runs; the position buffer holds kvLen 3, query offset 3, which
/// is the slot the step writes.</summary>
public static class QkvDecodeCases
{
    public const string Family = "qkvdecode";

    private const int Hq = 2, Hkv = 1, D = 8, MaxSeq = 5, Position = 3;
    private const float Eps = 1e-6f;

    private static readonly string[] Gpu = ["cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];

    /// <summary>The fused kernels are unchanged and the composed path runs the same ops in the same order.</summary>
    private static readonly Acceptance Exact = Acceptance.Exact;

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];
        foreach ((string suffix, bool interleaved) in new[] { ("split", false), ("interleaved", true) })
        {
            // Fused [q | k | v] projection, no norm.
            cases.Add(new(Family, "qkv_fused_" + suffix, seed => Build(seed).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).Tensors,
                (be, t) => RunAt(be, t, pos => be.QkvRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qkv"],
                    t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, interleaved, pos)), ServesDevicePositions, Gpu, Exact));
            // Separate q, k, v (the composed path after caller-side norms).
            cases.Add(new(Family, "separate_" + suffix, seed => Build(seed).In("q", new TensorShape(1, Hq, 1, D)).In("k", new TensorShape(1, Hkv, 1, D)).In("v", new TensorShape(1, Hkv, 1, D)).Tensors,
                (be, t) => RunAt(be, t, pos => be.RopeScatterKvDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"],
                    t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, interleaved, pos)), ServesDevicePositions, Gpu, Exact));
            // Fused projection with per-head RMS QK-norm (Qwen3 / Gemma-3).
            cases.Add(new(Family, "qkv_norm_perhead_" + suffix, seed => Build(seed).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(D)).In("kw", new TensorShape(D)).Tensors,
                (be, t) => RunAt(be, t, pos => be.QkvNormRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qkv"], t.Inputs["qw"], t.Inputs["kw"], Eps,
                    t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, interleaved, pos)), ServesDevicePositions, Gpu, Exact));
        }
        // Partial rotary width on the split-half pairing (the interleaved kernels take partial widths through identity table rows).
        cases.Add(new(Family, "separate_split_partial4", seed => Build(seed).In("q", new TensorShape(1, Hq, 1, D)).In("k", new TensorShape(1, Hkv, 1, D)).In("v", new TensorShape(1, Hkv, 1, D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.RopeScatterKvDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["q"], t.Inputs["k"], t.Inputs["v"],
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 4, false, pos)), ServesDevicePositions, Gpu, Exact));
        // [q | k] projection with v on its own (mixed-dtype v), no norm.
        cases.Add(new(Family, "qk_plus_v_split", seed => Build(seed).In("qk", new TensorShape(1, 1, (Hq + Hkv) * D)).In("v", new TensorShape(1, Hkv, 1, D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkRopeScatterVDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qk"], t.Inputs["v"],
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), ServesDevicePositions, Gpu, Exact));
        // [q | k] projection with v on its own, per-head RMS QK-norm.
        cases.Add(new(Family, "qk_norm_perhead_v_split", seed => Build(seed).In("qk", new TensorShape(1, 1, (Hq + Hkv) * D)).In("v", new TensorShape(1, Hkv, 1, D)).In("qw", new TensorShape(D)).In("kw", new TensorShape(D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkNormRopeScatterVDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qk"], t.Inputs["v"], t.Inputs["qw"], t.Inputs["kw"], Eps,
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), ServesDevicePositions, Gpu, Exact));
        // Full-width RMS QK-norm (OLMoE): one norm over every q head and one over every k head, fused [q | k | v] or [q | k] + v.
        cases.Add(new(Family, "qkv_norm_full_split", seed => Build(seed).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(Hq * D)).In("kw", new TensorShape(Hkv * D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkNormFullRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qkv"], null, t.Inputs["qw"], t.Inputs["kw"], Eps,
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), ServesDevicePositions, Gpu, Exact));
        cases.Add(new(Family, "qk_norm_full_v_split", seed => Build(seed).In("qk", new TensorShape(1, 1, (Hq + Hkv) * D)).In("v", new TensorShape(1, Hkv, 1, D)).In("qw", new TensorShape(Hq * D)).In("kw", new TensorShape(Hkv * D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkNormFullRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qk"], t.Inputs["v"], t.Inputs["qw"], t.Inputs["kw"], Eps,
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), ServesDevicePositions, Gpu, Exact));
        // F16 caches (CUDA): the scatter narrows k and v with saturation; q stays F32.
        cases.Add(new(Family, "qkv_fused_split_f16kv", seed => Build(seed, DType.F16).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkvRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qkv"],
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), be => be.SupportsF16KvCache, Cuda, Exact));
        cases.Add(new(Family, "qkv_norm_perhead_split_f16kv", seed => Build(seed, DType.F16).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(D)).In("kw", new TensorShape(D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkvNormRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qkv"], t.Inputs["qw"], t.Inputs["kw"], Eps,
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), be => be.SupportsF16KvCache, Cuda, Exact));
        cases.Add(new(Family, "qkv_norm_full_split_f16kv", seed => Build(seed, DType.F16).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(Hq * D)).In("kw", new TensorShape(Hkv * D)).Tensors,
            (be, t) => RunAt(be, t, pos => be.QkNormFullRopeScatterDecodeStep(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["qkv"], null, t.Inputs["qw"], t.Inputs["kw"], Eps,
                t.Inputs["cos"], t.Inputs["sin"], Hq, Hkv, D, 0, false, pos)), be => be.SupportsF16KvCache, Cuda, Exact));
        return cases;
    }

    /// <summary>A backend that keeps the decode position on the device (every graph decode consumer, or the text-LLM graph alone).</summary>
    private static bool ServesDevicePositions(IBackend be) => be.GraphDecodeSupported || be.LlmGraphDecodeSupported;

    /// <summary>The operands every case shares: the q output, the two caches at their recorded contents, and full tables <c>[MaxSeq, D]</c>.</summary>
    private static Operands Build(int seed, DType? cacheDtype = null)
        => new Operands(seed).Out("q", new TensorShape(1, Hq, 1, D)).InPlace("kCache", new TensorShape(1, Hkv, MaxSeq, D), cacheDtype ?? DType.F32).InPlace("vCache", new TensorShape(1, Hkv, MaxSeq, D), cacheDtype ?? DType.F32)
            .In("cos", new TensorShape(MaxSeq, D)).In("sin", new TensorShape(MaxSeq, D));

    /// <summary>Primes the caches, points the device position buffer at <see cref="Position"/>, runs the step and frees the buffer.</summary>
    private static void RunAt(IBackend be, OpTensors t, Action<ulong> step)
    {
        Prime(t, "kCache");
        Prime(t, "vCache");
        ulong pos = be.AllocDevicePos();
        if (pos == 0) throw new NotSupportedException("The backend has no device position buffer.");
        try
        {
            be.WriteDevicePos(pos, Position, Position);
            step(pos);
        }
        finally { be.FreeDevicePos(pos); }
    }

    /// <summary>Copies the recorded contents of in-place operand <paramref name="name"/> (input <c>{name}_init</c>) into its output before the op runs.</summary>
    private static void Prime(OpTensors t, string name) => t.Inputs[name + "_init"].AsReadOnlySpan<byte>().CopyTo(t.Outputs[name].AsSpan<byte>());

    /// <summary>Builds one case's operands: seeded F32 inputs (a distinct salt each), zeroed outputs, and in-place operands recorded as an input and written as an output.</summary>
    private sealed class Operands(int seed)
    {
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape)
        {
            Tensors.Inputs[name] = Seeded.Of(shape, DType.F32, seed, _salt++);
            return this;
        }

        public Operands Out(string name, TensorShape shape)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape, DType.F32);
            return this;
        }

        public Operands InPlace(string name, TensorShape shape, DType dtype)
        {
            Tensors.Inputs[name + "_init"] = Seeded.Of(shape, dtype, seed, _salt++);
            Tensors.Outputs[name] = Seeded.Zeros(shape, dtype);
            return this;
        }
    }
}
