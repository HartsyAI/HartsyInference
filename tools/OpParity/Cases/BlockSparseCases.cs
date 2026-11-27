using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Attention;
using HartsyInference.Core.Numerics;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>3D block-sparse attention (<see cref="IBackend.BlockSparseAttention"/>), judged against the CPU backend by the bars of
/// docs/BACKEND_OPS.md: the selected key blocks exactly, an F32 output within a relative RMS of 1e-5, a BF16 output within 1e-4 and
/// equal to the backend's own F32 result rounded to nearest even.</summary>
public static class BlockSparseCases
{
    public const string Family = "blocksparse";

    private static readonly string[] Required = ["cpu", "cuda", "vulkan"];

    private static readonly Acceptance Equal = Acceptance.AgainstHost(0, 0, "the selection is the reference's, and the rounding exact");
    private static readonly Acceptance Wide = Acceptance.AgainstHostRms(1e-5, "the softmax's exp and accumulation order follow each backend");
    private static readonly Acceptance Narrow = Acceptance.AgainstHostRms(1e-4, "a BF16 output can round to the neighbouring value");

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];
        // The shipped 4x4x4 blocks and sparsity at LongCat's head dim.
        BlockSparseAttentionOptions shipped = new() { QueryGrid = new(8, 16, 16), KeyGrid = new(8, 16, 16), Sparsity = 0.9375 };
        // Unequal grids and blocks, a query block longer than one CUDA tile and several sequences.
        BlockSparseAttentionOptions unequal = new()
        {
            QueryGrid = new(4, 8, 16), KeyGrid = new(8, 12, 16), QueryBlock = new(4, 4, 8), KeyBlock = new(2, 3, 4), Sparsity = 0.8,
        };
        Add(cases, "bsa_shipped_d128", shipped, 1, 2, 128, DType.F32);
        Add(cases, "bsa_unequal_d32", unequal, 2, 3, 32, DType.F32);
        Add(cases, "bsa_shipped_d64_bf16", shipped, 1, 2, 64, DType.BF16);
        return cases;
    }

    private static void Add(List<OpCase> cases, string name, BlockSparseAttentionOptions options, int batch, int heads, int dim, DType operands)
    {
        TensorShape queries = new(batch, heads, options.QueryGrid.Tokens, dim), keys = new(batch, heads, options.KeyGrid.Tokens, dim);
        TensorShape report = new(batch, heads, options.QueryBlocks, options.SelectedBlocks);
        OpTensors Build(int seed, string output, TensorShape shape, DType dtype)
        {
            OpTensors tensors = new();
            tensors.Inputs["q"] = Seeded.Of(queries, operands, seed, 1, 2f);
            tensors.Inputs["k"] = Seeded.Of(keys, operands, seed, 2, 2f);
            tensors.Inputs["v"] = Seeded.Of(keys, operands, seed, 3);
            tensors.Outputs[output] = Seeded.Zeros(shape, dtype);
            return tensors;
        }
        void Attend(IBackend backend, OpTensors t, Tensor output, Tensor? selected) =>
            backend.BlockSparseAttention(output, t.Inputs["q"], t.Inputs["k"], t.Inputs["v"], options, selected);

        // The report is I32; the comparison reads floats, which hold every block number exactly.
        cases.Add(new(Family, name + "_selected", seed => Build(seed, "selected", report, DType.F32), (be, t) =>
        {
            using Tensor output = new(queries, DType.F32);
            using Tensor selected = new(report, DType.I32);
            Attend(be, t, output, selected);
            ReadOnlySpan<int> blocks = selected.AsReadOnlySpan<int>();
            Span<float> copy = t.Outputs["selected"].AsSpan<float>();
            for (int i = 0; i < blocks.Length; i++) copy[i] = blocks[i];
        }, static be => be.HasBlockSparseAttention, Required, Equal));
        // How many BF16 values are not the backend's own F32 values rounded to nearest even: none.
        cases.Add(new(Family, name + "_bf16_rounded", seed => Build(seed, "mismatches", new TensorShape(1), DType.F32), (be, t) =>
        {
            using Tensor wide = new(queries, DType.F32), narrow = new(queries, DType.BF16);
            Attend(be, t, wide, null);
            Attend(be, t, narrow, null);
            ReadOnlySpan<float> unrounded = wide.AsReadOnlySpan<float>();
            ReadOnlySpan<ushort> rounded = narrow.AsReadOnlySpan<ushort>();
            int mismatches = 0;
            for (int i = 0; i < unrounded.Length; i++)
                if (!float.IsNaN(unrounded[i]) && rounded[i] != BFloat16Bits.FromFloat(unrounded[i])) mismatches++;
            t.Outputs["mismatches"].AsSpan<float>()[0] = mismatches;
        }, static be => be.HasBlockSparseAttention, Required, Equal));
        cases.Add(new(Family, name + "_f32", seed => Build(seed, "out", queries, DType.F32),
            (be, t) => Attend(be, t, t.Outputs["out"], null), static be => be.HasBlockSparseAttention, Required, Wide));
        cases.Add(new(Family, name + "_to_bf16", seed => Build(seed, "out", queries, DType.BF16),
            (be, t) => Attend(be, t, t.Outputs["out"], null), static be => be.HasBlockSparseAttention, Required, Narrow));
    }
}
