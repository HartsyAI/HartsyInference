using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Decode;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>The graph-decode QKV epilogue as <c>GenericTransformer.ForwardGraphStep</c> calls it, through
/// <see cref="IBackend.QkvDecodeEpilogue"/>: one token's projection output cut into q, k and v, optionally QK-normed per head
/// or over the full width, rotated by the row the device position buffer names, q emitted head-major and k, v scattered
/// into their caches at that position. One case per packing × norm × pairing, and the F16 cache forms CUDA serves. The
/// fixtures were recorded against the six entry points this replaced. The caches are in-place operands, so the recorded
/// contents are copied into them before the op runs; the position buffer holds kvLen 3, query offset 3, which is the slot
/// the step writes.</summary>
public static class QkvDecodeCases
{
    public const string Family = "qkvdecode";

    private const int Hq = 2, Hkv = 1, D = 8, MaxSeq = 5, Position = 3;
    private const float Eps = 1e-6f;

    private static readonly string[] Gpu = ["cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];

    /// <summary>The fused kernels and the composed ops are unchanged, so a recorded path matches bit for bit; a path judged against the
    /// reference instead (its norm and rotation round differently) is held within these bounds.</summary>
    private static readonly Acceptance Exact = new(AcceptanceKind.BitIdentical, 1e-5, 1e-4, "the fused kernels and the composed ops are unchanged");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];
        foreach ((string suffix, bool interleaved) in new[] { ("split", false), ("interleaved", true) })
        {
            // Fused [q | k | v] projection, no norm.
            Add(cases, "qkv_fused_" + suffix, seed => Build(seed).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).Tensors,
                (t, pos) => Call(t, pos, interleaved).Fused(t.Inputs["qkv"]), ServesDevicePositions, Gpu);
            // Separate q, k, v (the composed path after caller-side norms).
            Add(cases, "separate_" + suffix, seed => Build(seed).In("q", new TensorShape(1, Hq, 1, D)).In("k", new TensorShape(1, Hkv, 1, D)).In("v", new TensorShape(1, Hkv, 1, D)).Tensors,
                (t, pos) => Call(t, pos, interleaved).Separate(t.Inputs["q"], t.Inputs["k"], t.Inputs["v"]), ServesDevicePositions, Gpu);
            // Fused projection with per-head RMS QK-norm (Qwen3 / Gemma-3).
            Add(cases, "qkv_norm_perhead_" + suffix, seed => Build(seed).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(D)).In("kw", new TensorShape(D)).Tensors,
                (t, pos) => Call(t, pos, interleaved).Fused(t.Inputs["qkv"]).WithNorm(QkvNormKind.PerHead, t.Inputs["qw"], t.Inputs["kw"], Eps), ServesDevicePositions, Gpu);
        }
        // Partial rotary width on the split-half pairing (interleaved pairing takes a partial width from identity table rows).
        Add(cases, "separate_split_partial4", seed => Build(seed).In("q", new TensorShape(1, Hq, 1, D)).In("k", new TensorShape(1, Hkv, 1, D)).In("v", new TensorShape(1, Hkv, 1, D)).Tensors,
            (t, pos) => Call(t, pos, false, 4).Separate(t.Inputs["q"], t.Inputs["k"], t.Inputs["v"]), ServesDevicePositions, Gpu);
        // [q | k] projection with v on its own (mixed-dtype v), no norm.
        Add(cases, "qk_plus_v_split", seed => Build(seed).In("qk", new TensorShape(1, 1, (Hq + Hkv) * D)).In("v", new TensorShape(1, Hkv, 1, D)).Tensors,
            (t, pos) => Call(t, pos, false).QkPlusV(t.Inputs["qk"], t.Inputs["v"]), ServesDevicePositions, Gpu);
        // [q | k] projection with v on its own, per-head RMS QK-norm.
        Add(cases, "qk_norm_perhead_v_split", seed => Build(seed).In("qk", new TensorShape(1, 1, (Hq + Hkv) * D)).In("v", new TensorShape(1, Hkv, 1, D)).In("qw", new TensorShape(D)).In("kw", new TensorShape(D)).Tensors,
            (t, pos) => Call(t, pos, false).QkPlusV(t.Inputs["qk"], t.Inputs["v"]).WithNorm(QkvNormKind.PerHead, t.Inputs["qw"], t.Inputs["kw"], Eps), ServesDevicePositions, Gpu);
        // Full-width RMS QK-norm (OLMoE): one norm over every q head and one over every k head, fused [q | k | v] or [q | k] + v.
        Add(cases, "qkv_norm_full_split", seed => Build(seed).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(Hq * D)).In("kw", new TensorShape(Hkv * D)).Tensors,
            (t, pos) => Call(t, pos, false).Fused(t.Inputs["qkv"]).WithNorm(QkvNormKind.FullWidth, t.Inputs["qw"], t.Inputs["kw"], Eps), ServesDevicePositions, Gpu);
        Add(cases, "qk_norm_full_v_split", seed => Build(seed).In("qk", new TensorShape(1, 1, (Hq + Hkv) * D)).In("v", new TensorShape(1, Hkv, 1, D)).In("qw", new TensorShape(Hq * D)).In("kw", new TensorShape(Hkv * D)).Tensors,
            (t, pos) => Call(t, pos, false).QkPlusV(t.Inputs["qk"], t.Inputs["v"]).WithNorm(QkvNormKind.FullWidth, t.Inputs["qw"], t.Inputs["kw"], Eps), ServesDevicePositions, Gpu);
        // F16 caches (CUDA): the scatter narrows k and v with saturation; q stays F32.
        Add(cases, "qkv_fused_split_f16kv", seed => Build(seed, DType.F16).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).Tensors,
            (t, pos) => Call(t, pos, false).Fused(t.Inputs["qkv"]), be => be.SupportsF16KvCache, Cuda);
        Add(cases, "qkv_norm_perhead_split_f16kv", seed => Build(seed, DType.F16).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(D)).In("kw", new TensorShape(D)).Tensors,
            (t, pos) => Call(t, pos, false).Fused(t.Inputs["qkv"]).WithNorm(QkvNormKind.PerHead, t.Inputs["qw"], t.Inputs["kw"], Eps), be => be.SupportsF16KvCache, Cuda);
        Add(cases, "qkv_norm_full_split_f16kv", seed => Build(seed, DType.F16).In("qkv", new TensorShape(1, 1, (Hq + 2 * Hkv) * D)).In("qw", new TensorShape(Hq * D)).In("kw", new TensorShape(Hkv * D)).Tensors,
            (t, pos) => Call(t, pos, false).Fused(t.Inputs["qkv"]).WithNorm(QkvNormKind.FullWidth, t.Inputs["qw"], t.Inputs["kw"], Eps), be => be.SupportsF16KvCache, Cuda);
        return cases;
    }

    /// <summary>One case: <paramref name="make"/> builds the call from the operands and the device position buffer; the run
    /// executes it on the backend and the oracle evaluates the contract reference at <see cref="Position"/>.</summary>
    private static void Add(List<OpCase> cases, string name, Func<int, OpTensors> build, Func<OpTensors, ulong, QkvDecodeCall> make, Func<IBackend, bool> supportedOn, string[] required)
        => cases.Add(new(Family, name, build, (be, t) => RunAt(be, t, pos => be.QkvDecodeEpilogue(make(t, pos))), supportedOn, required, Exact,
            t =>
            {
                Prime(t, "kCache");
                Prime(t, "vCache");
                // The reference takes the position itself and never reads the buffer; the handle only satisfies the contract.
                QkvDecodeReference.Apply(make(t, ReferenceHandle), Position);
            }));

    /// <summary>A stand-in device position handle for the host reference, which is given the position directly.</summary>
    private const ulong ReferenceHandle = 1;

    /// <summary>The shared operands of a case as a call builder: the q output, the caches, the tables, the pairing and the rotated width; the packing picks the sections.</summary>
    private static CallShape Call(OpTensors t, ulong pos, bool interleaved, int rotaryDim = 0)
        => new(t.Outputs["q"], t.Outputs["kCache"], t.Outputs["vCache"], t.Inputs["cos"], t.Inputs["sin"], rotaryDim, interleaved ? RotaryPairing.Interleaved : RotaryPairing.RotateHalf, pos);

    private readonly record struct CallShape(Tensor QOut, Tensor KCache, Tensor VCache, Tensor Cos, Tensor Sin, int RotaryDim, RotaryPairing Pairing, ulong Pos)
    {
        public QkvDecodeCall Fused(Tensor qkv) => QkvDecodeCall.Fused(QOut, KCache, VCache, qkv, Hq, Hkv, D, Cos, Sin, RotaryDim, Pairing, Pos);

        public QkvDecodeCall QkPlusV(Tensor qk, Tensor v) => QkvDecodeCall.QkPlusV(QOut, KCache, VCache, qk, v, Hq, Hkv, D, Cos, Sin, RotaryDim, Pairing, Pos);

        public QkvDecodeCall Separate(Tensor q, Tensor k, Tensor v) => QkvDecodeCall.Separate(QOut, KCache, VCache, q, k, v, Hq, Hkv, D, Cos, Sin, RotaryDim, Pairing, Pos);
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
