using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>Every row-index movement the models apply, one case per caller form: the row gather (MoE dispatch, the last-row
/// slice, position-row and embedding lookups, the sparse-conv rulebook and SeedVR2's window groups), the scatter-add with
/// and without per-row weights (MoE combine, deepstack injection, the rulebook's accumulate), the gather-add of an embedding
/// table into every row (Ideogram 4's image indicator), and TRELLIS's dense-grid scatter and gather of active voxels. The
/// fixtures were recorded against the entry points <see cref="IBackend.GatherRows(Tensor, Tensor, Tensor)"/>,
/// <see cref="IBackend.ScatterAddRows(Tensor, Tensor, Tensor, Tensor?)"/>, <see cref="IBackend.GatherAddRows"/>,
/// <see cref="IBackend.GridScatter"/> and <see cref="IBackend.GridGather"/> replaced. Rows are F32 and indices are I32 everywhere.</summary>
public static class RowIndexCases
{
    public const string Family = "rowindex";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];

    /// <summary>Row copies and single-visit accumulates: the same bytes and the same one add per element on every backend.</summary>
    private static readonly Acceptance Exact = Acceptance.Exact;

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // Row gather over a [rows, width] table: repeated rows are valid (MoE dispatch, position rows, embedding lookup).
        cases.Add(new(Family, "gather_rows_f32", seed => new Operands(seed).In("table", new TensorShape(6, 8)).Index("rows", [4, 0, 4, 2]).Out("out", new TensorShape(4, 8)).Tensors,
            (be, t) => be.GatherRows(t.Outputs["out"], t.Inputs["table"], t.Inputs["rows"]), _ => true, All, Exact));
        // The last-row slice of a [1, T, H] hidden state before the head projection.
        cases.Add(new(Family, "gather_last_row_f32", seed => new Operands(seed).In("hidden", new TensorShape(1, 5, 16)).Index("rows", [4]).Out("out", new TensorShape(1, 1, 16)).Tensors,
            (be, t) => be.GatherRows(t.Outputs["out"], t.Inputs["hidden"], t.Inputs["rows"]), _ => true, All, Exact));
        // The rulebook and window gathers address [1, rows, channels] features.
        cases.Add(new(Family, "gather_rows_tensor_f32", seed => new Operands(seed).In("feats", new TensorShape(1, 7, 4)).Index("rows", [6, 6, 1, 0, 3]).Out("out", new TensorShape(1, 5, 4)).Tensors,
            (be, t) => be.GatherRows(t.Outputs["out"], t.Inputs["feats"], t.Inputs["rows"]), _ => true, All, Exact));

        // Weighted scatter-add in place: out[rows[m]] += w[m] · in[m], each destination row visited once (MoE combine, deepstack).
        cases.Add(new(Family, "scatter_add_weighted_f32", seed => new Operands(seed).InPlace("acc", new TensorShape(1, 6, 8)).In("rows_in", new TensorShape(1, 3, 8)).Index("rows", [5, 0, 2]).In("weights", new TensorShape(3), 2f).Tensors,
            (be, t) => { Prime(t, "acc"); be.ScatterAddRows(t.Outputs["acc"], t.Inputs["rows_in"], t.Inputs["rows"], t.Inputs["weights"]); }, _ => true, All, Exact));
        // The shared-expert seed: every row scattered onto itself with its sigmoid gate.
        cases.Add(new(Family, "scatter_add_identity_f32", seed => new Operands(seed).InPlace("acc", new TensorShape(1, 4, 8), zero: true).In("rows_in", new TensorShape(1, 4, 8)).Index("rows", [0, 1, 2, 3]).In("weights", new TensorShape(4)).Tensors,
            (be, t) => { Prime(t, "acc"); be.ScatterAddRows(t.Outputs["acc"], t.Inputs["rows_in"], t.Inputs["rows"], t.Inputs["weights"]); }, _ => true, All, Exact));
        // Unweighted scatter-add (the rulebook's accumulate, SeedVR2's video rows).
        cases.Add(new(Family, "scatter_add_tensor_f32", seed => new Operands(seed).InPlace("acc", new TensorShape(1, 7, 4)).In("rows_in", new TensorShape(1, 3, 4)).Index("rows", [2, 6, 0]).Tensors,
            (be, t) => { Prime(t, "acc"); be.ScatterAddRows(t.Outputs["acc"], t.Inputs["rows_in"], t.Inputs["rows"]); }, _ => true, All, Exact));

        // Gather-add: every row of h adds the table row its index names; repeated indices are the common case.
        cases.Add(new(Family, "gather_add_rows_f32", seed => new Operands(seed).InPlace("h", new TensorShape(2, 3, 8)).In("table", new TensorShape(4, 8)).Index("rows", [1, 3, 3, 0, 2, 1]).Tensors,
            (be, t) => { Prime(t, "h"); be.GatherAddRows(t.Outputs["h"], t.Inputs["table"], t.Inputs["rows"]); }, _ => true, All, Exact));

        // Dense-grid scatter of [N, C] active-voxel features into [1, C, R, R, R] at (d, h, w) = coords[:, 1..3]; untouched cells keep their bytes.
        cases.Add(new(Family, "grid_scatter_f32", seed => new Operands(seed).InPlace("grid", new TensorShape([1, 3, 4, 4, 4])).In("feats", new TensorShape(5, 3)).Index("coords", Voxels).Tensors,
            (be, t) => { Prime(t, "grid"); be.GridScatter(t.Outputs["grid"], t.Inputs["feats"], t.Inputs["coords"], GridSpec.Cube(4, 4, 1)); }, _ => true, All, Exact));
        // Its inverse: [1, C, R, R, R] back to [N, C] at the same voxels.
        cases.Add(new(Family, "grid_gather_f32", seed => new Operands(seed).In("grid", new TensorShape([1, 3, 4, 4, 4])).Index("coords", Voxels).Out("feats", new TensorShape(5, 3)).Tensors,
            (be, t) => be.GridGather(t.Outputs["feats"], t.Inputs["grid"], t.Inputs["coords"], GridSpec.Cube(4, 4, 1)), _ => true, All, Exact));
        return cases;
    }

    /// <summary>Five active voxels of a 4³ grid as TRELLIS stores them: <c>[batch, d, h, w]</c> per row.</summary>
    private static ReadOnlySpan<int> Voxels => [0, 0, 0, 0, 0, 3, 3, 3, 0, 1, 2, 3, 0, 2, 0, 1, 0, 3, 1, 0];

    private static void Prime(OpTensors t, string name) => t.Inputs[name + "_init"].AsReadOnlySpan<byte>().CopyTo(t.Outputs[name].AsSpan<byte>());

    /// <summary>Builds one case's operands: seeded F32 inputs (a distinct salt each), I32 index rows, zeroed outputs and in-place
    /// operands recorded as an input and written as an output.</summary>
    private sealed class Operands(int seed)
    {
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape, float scale = 1f)
        {
            Tensors.Inputs[name] = Seeded.F32(shape, seed, _salt++, scale);
            return this;
        }

        public Operands Index(string name, ReadOnlySpan<int> rows)
        {
            Tensor t = Seeded.Zeros(new TensorShape(rows.Length), DType.I32);
            rows.CopyTo(t.AsSpan<int>());
            Tensors.Inputs[name] = t;
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
