using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Host;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Rotary;
using HartsyInference.Core.Rope;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Every rotary position embedding the models apply, one case per caller parameter set, expressed through
/// <see cref="IBackend.Rotary"/> and <see cref="IBackend.RmsNormRotary"/> exactly as the callers build them. The
/// fixtures were recorded against the eleven entry points these replaced. The tables are seeded values, not real
/// angles: a rotation is two multiplies and an add per element whatever the table holds, so the fixtures still pin
/// the index maps and the arithmetic. Every rotation is in place, so the recorded input is copied into the output
/// before the op runs.</summary>
public static class RotaryCases
{
    public const string Family = "rotary";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];
    private static readonly string[] Gpu = ["cuda", "vulkan"];

    /// <summary>The generic kernel rounds each multiply and add once; the legacy kernels let the compiler contract them into FMAs, so a fixture may differ in the last bits.</summary>
    private static readonly Acceptance Rotation = Acceptance.Within(4e-6, 1e-4, "the generic rotary rounds each multiply and add once; the legacy kernels let the compiler contract them into FMAs");

    /// <summary>The fused norm reduces in a different order on each backend.</summary>
    private static readonly Acceptance NormRotation = Acceptance.Within(1e-5, 1e-4, "the fused RMS norm reduces in a different order per backend, then rotates");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // Oasis: head-major [b, heads, seq, headDim], interleaved pairs, partial rotary, tables [seq, rotDim] with duplicated pairs.
        cases.Add(new(Family, "oasis_interleaved_partial", seed => new Operands(seed).InPlace("x", new TensorShape(2, 2, 3, 8)).In("cos", new TensorShape(3, 4)).In("sin", new TensorShape(3, 4)).Tensors,
            (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.HeadMajor(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair, 4)); }, _ => true, All, Rotation));

        // Ideogram 4 / Ernie / F-Lite / Mert2: rotate-half on token-major q and k, GQA head counts, dual-angle tables [B, L, D].
        cases.Add(new(Family, "apply_rope_paired_gqa", seed => new Operands(seed).InPlace("q", new TensorShape(2, 3, 4, 8)).InPlace("k", new TensorShape(2, 3, 2, 8)).In("cos", new TensorShape(2, 3, 8)).In("sin", new TensorShape(2, 3, 8)).Tensors,
            (be, t) =>
            {
                Prime(t, "q");
                Prime(t, "k");
                be.Rotary(RotaryCall.TokenMajor(t.Outputs["q"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
                be.Rotary(RotaryCall.TokenMajor(t.Outputs["k"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf));
            },
            _ => true, All, Rotation));

        // GPT-J interleaved on token-major [B, L, H, D], prefix tables [B, L, D]; full and partial rotary.
        foreach ((string name, int rotaryDim) in new[] { ("apply_rope_interleaved_full", 0), ("apply_rope_interleaved_partial4", 4) })
        {
            cases.Add(new(Family, name, seed => new Operands(seed).InPlace("x", new TensorShape(2, 3, 2, 8)).In("cos", new TensorShape(2, 3, 8)).In("sin", new TensorShape(2, 3, 8)).Tensors,
                (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.TokenMajor(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix, rotaryDim)); }, _ => true, All, Rotation));
        }
        // DeepSeek V4.1: interleaved pairs on a slice [dimOffset, dimOffset + rotaryDim) of each vector, tables [B, L, rotaryDim/2].
        cases.Add(new(Family, "apply_rope_interleaved_offset", seed => new Operands(seed).InPlace("x", new TensorShape(1, 3, 2, 12)).In("cos", new TensorShape(1, 3, 2)).In("sin", new TensorShape(1, 3, 2)).Tensors,
            (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.TokenMajor(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix, 4, 6)); }, _ => true, All, Rotation));

        // Wan / Flux / HunyuanImage / Kandinsky 5 / Z-Image / SeedVR2 / LTX: interleaved on rows [S, heads·D], tables [S, D] shared across heads.
        // F16 activations were served by the GPUs only; the reference reads them now, so the CPU path compares to itself.
        foreach ((string suffix, DType dtype, string[] required) in new[] { ("f32", DType.F32, All), ("f16", DType.F16, Gpu) })
        {
            cases.Add(new(Family, "wan_interleaved_" + suffix, seed => new Operands(seed, dtype).InPlace("x", new TensorShape(3, 16)).In("cos", new TensorShape(3, 8), DType.F32).In("sin", new TensorShape(3, 8), DType.F32).Tensors,
                (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.Rows(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], 3, 2, 8, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair)); }, _ => true, required, Rotation));
            // LTX-2: rotate-half with one angle per pair, per-head tables [S, heads·D/2].
            cases.Add(new(Family, "ltx2_split_" + suffix, seed => new Operands(seed, dtype).InPlace("x", new TensorShape(3, 16)).In("cos", new TensorShape(3, 8), DType.F32).In("sin", new TensorShape(3, 8), DType.F32).Tensors,
                (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.SplitRows(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], 3, 2, 8)); }, _ => true, required, Rotation));
            // Rotate-half on token-major [B, L, H, D] with tables [B, L, D]: the LLM encoders and most DiT blocks.
            cases.Add(new(Family, "apply_rope_single_full_" + suffix, seed => new Operands(seed, dtype).InPlace("x", new TensorShape(2, 3, 2, 8)).In("cos", new TensorShape(2, 3, 8), DType.F32).In("sin", new TensorShape(2, 3, 8), DType.F32).Tensors,
                (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.TokenMajor(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf)); }, _ => true, required, Rotation));
        }
        cases.Add(new(Family, "apply_rope_single_partial4", seed => new Operands(seed).InPlace("x", new TensorShape(2, 3, 2, 8)).In("cos", new TensorShape(2, 3, 8)).In("sin", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.TokenMajor(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, 4)); }, _ => true, All, Rotation));
        // Head-major twin [B, H, L, D] with the same tables: Gemma 4, Llama-style encoders, MiniMax H3.
        cases.Add(new(Family, "apply_rope_single_headmajor", seed => new Operands(seed).InPlace("x", new TensorShape(2, 2, 3, 8)).In("cos", new TensorShape(2, 3, 8)).In("sin", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.HeadMajor(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf)); }, _ => true, All, Rotation));

        // Matrix-Game 3 sigma_theta: per-head tables [heads, S, D].
        cases.Add(new(Family, "wan_interleaved_perhead", seed => new Operands(seed).InPlace("x", new TensorShape(3, 16)).In("cos", new TensorShape(2, 3, 8)).In("sin", new TensorShape(2, 3, 8)).Tensors,
            (be, t) => { Prime(t, "x"); be.Rotary(RotaryCall.RowsPerHead(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], 3, 2, 8, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair)); }, _ => true, All, Rotation));
        // Matrix-Game 3 action module: [sp, heads, tt, D] with a grid-row table (frame, or frame·gh·gw + position).
        foreach ((string name, bool broadcast, int rows) in new[] { ("mg3_batched_broadcast", true, 3), ("mg3_batched_grid", false, 12) })
        {
            cases.Add(new(Family, name, seed => new Operands(seed).InPlace("x", new TensorShape(4, 2, 3, 8)).In("cos", new TensorShape(rows, 8)).In("sin", new TensorShape(rows, 8)).Tensors,
                (be, t) =>
                {
                    Prime(t, "x");
                    StridedView table = broadcast ? RotaryViews.SharedTable(t.Inputs["cos"].Shape, 4, 2) : RotaryViews.Explicit([4, 2, 3, 1, 1, 8], [8, 0, 4 * 8, 0, 0, 1]);
                    be.Rotary(new RotaryCall(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], RotaryViews.Create(RotaryViews.HeadMajor(t.Outputs["x"].Shape), table, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPair)));
                },
                _ => true, All, Rotation));
        }
        // LTX-2.5 NA decoder: three head-dim segments rotated by the token's t, h and w coordinates, tables [axis, pairs].
        cases.Add(new(Family, "ltx25_na_rope3d", seed => new Operands(seed).InPlace("x", new TensorShape([1, 2, 2, 3, 2, 12])).In("cosT", new TensorShape(2, 2)).In("sinT", new TensorShape(2, 2)).In("cosH", new TensorShape(2, 2)).In("sinH", new TensorShape(2, 2)).In("cosW", new TensorShape(3, 2)).In("sinW", new TensorShape(3, 2)).Tensors,
            (be, t) =>
            {
                Prime(t, "x");
                Tensor x = t.Outputs["x"];
                StridedView grid = RotaryViews.Grid(x.Shape);
                foreach ((string axisName, int axis, int width, int offset) in new[] { ("T", 0, 4, 0), ("H", 1, 4, 4), ("W", 2, 4, 8) })
                {
                    StridedView table = RotaryViews.AxisTable(grid[0], grid[1], grid[2], grid[3], grid[4], axis, width / 2);
                    be.Rotary(new RotaryCall(x, t.Inputs["cos" + axisName], t.Inputs["sin" + axisName], RotaryViews.Create(grid, table, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix, width, offset)));
                }
            },
            _ => true, All, Rotation));

        // Graph decode: one step's [1, H, 1, D] rotated by the row the device position buffer names (query offset 2 of a 5-row table).
        foreach ((string name, bool interleaved) in new[] { ("decode_step_splithalf", false), ("decode_step_interleaved", true) })
        {
            cases.Add(new(Family, name, seed => new Operands(seed).InPlace("x", new TensorShape(1, 2, 1, 8)).In("cos", new TensorShape(5, 8)).In("sin", new TensorShape(5, 8)).Tensors,
                (be, t) =>
                {
                    Prime(t, "x");
                    ulong pos = be.AllocDevicePos();
                    try
                    {
                        be.WriteDevicePos(pos, 3, 2);
                        be.Rotary(RotaryCall.DecodeStep(t.Outputs["x"], t.Inputs["cos"], t.Inputs["sin"], 0, interleaved, pos));
                    }
                    finally { be.FreeDevicePos(pos); }
                },
                ServesDevicePositions, Gpu, Rotation, t => DecodeStepOracle(t, interleaved)));
        }

        // LTX-2 fused QK path: full-row RMS norm, split rotation with per-head tables, head-major or token-major emit; and the no-rotation form.
        foreach ((string suffix, DType dtype, Func<IBackend, bool> serves, string[] required) in new (string, DType, Func<IBackend, bool>, string[])[] { ("f32", DType.F32, _ => true, All), ("f16", DType.F16, be => be.Device.IsCuda, Cuda) })
        {
            cases.Add(new(Family, "ltx2_qknorm_rope_headmajor_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(3, 16)).In("w", new TensorShape(16), DType.F32).In("cos", new TensorShape(3, 8), DType.F32).In("sin", new TensorShape(3, 8), DType.F32).Out("out", new TensorShape(1, 2, 3, 8)).Tensors,
                (be, t) => be.RmsNormRotary(t.Outputs["out"], t.Inputs["x"], t.Inputs["w"], 1e-6f, t.Inputs["cos"], t.Inputs["sin"], RotaryViews.SplitRows(3, 2, 8)), serves, required, NormRotation));
            cases.Add(new(Family, "ltx2_qknorm_rope_tokenmajor_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape(3, 16)).In("w", new TensorShape(16), DType.F32).In("cos", new TensorShape(3, 8), DType.F32).In("sin", new TensorShape(3, 8), DType.F32).Out("out", new TensorShape(3, 16)).Tensors,
                (be, t) => be.RmsNormRotary(t.Outputs["out"], t.Inputs["x"], t.Inputs["w"], 1e-6f, t.Inputs["cos"], t.Inputs["sin"], RotaryViews.SplitRows(3, 2, 8)), serves, required, NormRotation));
        }
        cases.Add(new(Family, "ltx2_qknorm_norope_headmajor", seed => new Operands(seed).In("x", new TensorShape(3, 16)).In("w", new TensorShape(16)).Out("out", new TensorShape(1, 2, 3, 8)).Tensors,
            (be, t) => be.RmsNormRotary(t.Outputs["out"], t.Inputs["x"], t.Inputs["w"], 1e-6f, null, null, RotaryViews.SplitRows(3, 2, 8)), _ => true, All, NormRotation));
        return cases;
    }

    /// <summary>Copies the recorded contents of in-place operand <paramref name="name"/> (input <c>{name}_init</c>) into its output before the op runs.</summary>
    /// <summary>Table row the decode-step cases rotate by: the query offset their device position buffer carries.</summary>
    private const int DecodeRow = 2;

    /// <summary>The decode-step rotation on the host, for a backend newly serving it: row <see cref="DecodeRow"/> of each table copied out
    /// as a one-row table and applied through the same views the device call builds, since the reference reads no position buffer.</summary>
    private static void DecodeStepOracle(OpTensors t, bool interleaved)
    {
        Prime(t, "x");
        Tensor x = t.Outputs["x"];
        using Tensor cos = TableRow(t.Inputs["cos"], DecodeRow);
        using Tensor sin = TableRow(t.Inputs["sin"], DecodeRow);
        int headDim = (int)x.Shape[x.Shape.Rank - 1];
        long heads = x.ElementCount / headDim;
        long entries = cos.Shape[1];
        StridedView vectors = RotaryViews.Explicit([1, heads, 1, 1, 1, headDim], [0, headDim, 0, 0, 0, 1]);
        StridedView table = RotaryViews.Explicit([1, heads, 1, 1, 1, entries], [0, 0, 0, 0, 0, 1]);
        RotarySpec spec = interleaved
            ? RotaryViews.Create(vectors, table, RotaryPairing.Interleaved, RopeTableLayout.InterleavedPrefix)
            : RotaryViews.Create(vectors, table, RotaryPairing.RotateHalf, RopeTableLayout.RotateHalf, 0);
        RotaryReference.Apply(new RotaryCall(x, cos, sin, spec));
    }

    /// <summary>Row <paramref name="row"/> of a <c>[rows, entries]</c> F32 table as its own <c>[1, entries]</c> tensor.</summary>
    private static Tensor TableRow(Tensor table, int row)
    {
        int entries = (int)table.Shape[1];
        Tensor one = Seeded.Zeros(new TensorShape(1, entries));
        table.AsReadOnlySpan<float>().Slice(row * entries, entries).CopyTo(one.AsSpan<float>());
        return one;
    }

    /// <summary>Whether the backend serves a device position buffer: the full graph-decode toolkit, or Vulkan's LLM graph decode, which carries the
    /// position buffer without the rest of it.</summary>
    private static bool ServesDevicePositions(IBackend be) => be.GraphDecodeSupported || be.LlmGraphDecodeSupported;

    private static void Prime(OpTensors t, string name) => t.Inputs[name + "_init"].AsReadOnlySpan<byte>().CopyTo(t.Outputs[name].AsSpan<byte>());

    /// <summary>Builds one case's operands: seeded inputs (a distinct salt each), zeroed outputs, and in-place operands recorded as an input and written as an output.</summary>
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
    }
}
