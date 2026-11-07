using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>The layout moves the models make (head splits and merges, frame slices and scatters, pads, broadcasts and
/// tap accumulation), one case per caller parameter set, expressed through <see cref="IBackend.CopyStrided"/>,
/// <see cref="IBackend.AccumulateStrided"/> and <see cref="IBackend.Pad"/> exactly as the callers build them. The
/// fixtures were recorded against the model-named ops these replaced, so a check proves the generic path moves the
/// same bytes. Every case is a pure copy, a zero fill or one add per element: the rule is byte equality.</summary>
public static class LayoutCases
{
    public const string Family = "layout";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];
    private static readonly string[] CudaVulkan = ["cuda", "vulkan"];

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // Oasis spatio-temporal block: frames=3, sp=8, heads=2, headDim=4 (dim 8); qkv rows [frames·sp, 3·dim].
        foreach ((string name, int part, bool temporal) in new[] { ("oasis_split_spatial_q", 0, false), ("oasis_split_spatial_k", 1, false), ("oasis_split_spatial_v", 2, false), ("oasis_split_temporal_k", 1, true) })
        {
            TensorShape outShape = temporal ? new TensorShape(8, 2, 3, 4) : new TensorShape(3, 2, 8, 4);
            cases.Add(new(Family, name, seed => new Operands(seed).In("qkv", new TensorShape(24, 24)).Out("out", outShape).Tensors,
                (be, t) => be.CopyStrided(t.Outputs["out"], t.Inputs["qkv"], StridedCopySpec.Gather(StridedView.Of(t.Inputs["qkv"].Shape).Reshape([3, 8, 3, 2, 4]).Select(2, part).Permute(OasisOrder(temporal)))),
                _ => true, All, Acceptance.Exact));
        }
        foreach ((string name, bool temporal) in new[] { ("oasis_merge_spatial", false), ("oasis_merge_temporal", true) })
        {
            TensorShape attnShape = temporal ? new TensorShape(8, 2, 3, 4) : new TensorShape(3, 2, 8, 4);
            cases.Add(new(Family, name, seed => new Operands(seed).In("attn", attnShape).Out("out", new TensorShape(24, 8)).Tensors,
                (be, t) => be.CopyStrided(t.Outputs["out"], t.Inputs["attn"], StridedCopySpec.Scatter(StridedView.Of(t.Outputs["out"].Shape).Reshape([3, 8, 2, 4]).Permute(OasisOrder(temporal)))),
                _ => true, All, Acceptance.Exact));
        }

        // Matrix-Game 3 action module: tt=3, sp=6, heads=2, headDim=4 (streamDim 8). The mouse stream takes QKV slots from
        // [tt·sp, 3·streamDim]; the keyboard stream takes Q from [tt·sp, streamDim] (stride 1) and broadcasts K/V over sp.
        foreach ((string name, int part, int stride) in new[] { ("mg3_split_qkv_temporal_q", 0, 3), ("mg3_split_qkv_temporal_k", 1, 3), ("mg3_split_qkv_temporal_v", 2, 3), ("mg3_split_q_temporal_stride1", 0, 1) })
        {
            cases.Add(new(Family, name, seed => new Operands(seed).In("qkv", new TensorShape(18, stride * 8)).Out("out", new TensorShape(6, 2, 3, 4)).Tensors,
                (be, t) => be.CopyStrided(t.Outputs["out"], t.Inputs["qkv"], StridedCopySpec.Gather(StridedView.Of(t.Inputs["qkv"].Shape).Reshape([3, 6, stride, 2, 4]).Select(2, part).Permute([1, 2, 0, 3]))),
                _ => true, All, Acceptance.Exact));
        }
        cases.Add(new(Family, "mg3_merge_temporal", seed => new Operands(seed).In("attn", new TensorShape(6, 2, 3, 4)).Out("out", new TensorShape(18, 8)).Tensors,
            (be, t) => be.CopyStrided(t.Outputs["out"], t.Inputs["attn"], StridedCopySpec.Scatter(StridedView.Of(t.Outputs["out"].Shape).Reshape([3, 6, 2, 4]).Permute([1, 2, 0, 3]))),
            _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "mg3_kv_expand", seed => new Operands(seed).In("kv", new TensorShape(3, 16)).Out("k", new TensorShape(6, 2, 3, 4)).Out("v", new TensorShape(6, 2, 3, 4)).Tensors,
            (be, t) =>
            {
                StridedView kvRows = StridedView.Of(t.Inputs["kv"].Shape).Reshape([3, 2, 2, 4]);
                be.CopyStrided(t.Outputs["k"], t.Inputs["kv"], StridedCopySpec.Gather(kvRows.Select(1, 0).Broadcast(0, 6).Permute([0, 2, 1, 3])));
                be.CopyStrided(t.Outputs["v"], t.Inputs["kv"], StridedCopySpec.Gather(kvRows.Select(1, 1).Broadcast(0, 6).Permute([0, 2, 1, 3])));
            },
            _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "mg3_mouse_mlp_concat", seed => new Operands(seed).In("hidden", new TensorShape(18, 5)).In("win", new TensorShape(3, 4)).Out("out", new TensorShape(18, 9)).Tensors,
            (be, t) =>
            {
                using Tensor window = new(new TensorShape(18, 4), DType.F32);
                be.CopyStrided(window, t.Inputs["win"], StridedCopySpec.Gather(StridedView.Of(t.Inputs["win"].Shape).Broadcast(1, 6)));
                be.Concat(t.Outputs["out"], [t.Inputs["hidden"], window], dim: 1);
            },
            _ => true, All, Acceptance.Exact));

        // Wan 2.2 VAE attention block: bt=2, c=3, h=2, w=4 (hw 8).
        cases.Add(new(Family, "wan_split_vae_qkv", seed => new Operands(seed).In("qkv", new TensorShape(2, 9, 2, 4)).Out("q", new TensorShape(2, 1, 8, 3)).Out("k", new TensorShape(2, 1, 8, 3)).Out("v", new TensorShape(2, 1, 8, 3)).Tensors,
            (be, t) =>
            {
                StridedView planes = StridedView.Of(t.Inputs["qkv"].Shape).Reshape([2, 3, 3, 8]);
                be.CopyStrided(t.Outputs["q"], t.Inputs["qkv"], StridedCopySpec.Gather(planes.Select(1, 0).Permute([0, 2, 1])));
                be.CopyStrided(t.Outputs["k"], t.Inputs["qkv"], StridedCopySpec.Gather(planes.Select(1, 1).Permute([0, 2, 1])));
                be.CopyStrided(t.Outputs["v"], t.Inputs["qkv"], StridedCopySpec.Gather(planes.Select(1, 2).Permute([0, 2, 1])));
            },
            _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "wan_vae_tokens_to_frame", seed => new Operands(seed).In("attn", new TensorShape(2, 1, 8, 3)).Out("out", new TensorShape(2, 3, 2, 4)).Tensors,
            (be, t) => be.CopyStrided(t.Outputs["out"], t.Inputs["attn"], StridedCopySpec.Gather(StridedView.Of(t.Inputs["attn"].Shape).Reshape([2, 8, 3]).Permute([0, 2, 1]))),
            _ => true, All, Acceptance.Exact));

        // GQA K/V repeat [B,Hkv,L,D] → [B,Hkv·g,L,D]: every backend copies any 16-bit word now; the baseline served F32
        // everywhere, F16 on both GPUs and BF16 on CUDA, so those are the fixtures and the rest compare to the reference.
        foreach ((string suffix, DType dtype, string[] required) in new[] { ("f32", DType.F32, All), ("f16", DType.F16, CudaVulkan), ("bf16", DType.BF16, Cuda) })
        {
            cases.Add(new(Family, "repeat_kv_heads_" + suffix, seed => new Operands(seed, dtype).In("kv", new TensorShape(2, 2, 5, 4)).Out("out", new TensorShape(2, 6, 5, 4)).Tensors,
                (be, t) => StridedCopies.RepeatInterleave(be, t.Outputs["out"], t.Inputs["kv"], 1, 3), _ => true, required, Acceptance.Exact));
        }
        cases.Add(new(Family, "repeat_kv_heads_broadcast_one", seed => new Operands(seed).In("kv", new TensorShape(1, 1, 5, 4)).Out("out", new TensorShape(1, 4, 5, 4)).Tensors,
            (be, t) => StridedCopies.RepeatInterleave(be, t.Outputs["out"], t.Inputs["kv"], 1, 4), _ => true, All, Acceptance.Exact));

        // Time-axis slices of [1,h,t,d]: paged-cache page chunks, the valid prefix, the last row, and the Qwen3.5 [q|gate] de-interleave.
        foreach ((string name, int t, int start, int len) in new[] { ("slice_time_prefix", 7, 0, 4), ("slice_time_mid", 7, 2, 3), ("slice_time_last", 7, 6, 1), ("slice_time_gate", 2, 1, 1) })
        {
            cases.Add(new(Family, name, seed => new Operands(seed).In("kv", new TensorShape(1, 2, t, 4)).Out("out", new TensorShape(1, 2, len, 4)).Tensors,
                (be, o) => StridedCopies.SliceAxis(be, o.Outputs["out"], o.Inputs["kv"], 2, start, len), _ => true, All, Acceptance.Exact));
        }

        // CausalConv3d frame ops on [B,C,T,H,W] with B=2, C=3, T=4, H=2, W=3, and the SeedVR2 pad. The baseline served F32
        // everywhere and BF16 on CUDA only; the generic ops serve BF16 everywhere, so the CPU and Vulkan BF16 paths compare to
        // the reference. The in-place ops record the output's initial contents as the `out_init` input and copy it in first.
        foreach ((string suffix, DType dtype) in new[] { ("f32", DType.F32), ("bf16", DType.BF16) })
        {
            string[] required = dtype == DType.F32 ? All : Cuda;
            cases.Add(new(Family, "extract_vae_frame_" + suffix, seed => new Operands(seed, dtype).In("src", new TensorShape([2, 3, 4, 2, 3])).Out("out", new TensorShape(2, 3, 2, 3)).Tensors,
                (be, t) => StridedCopies.SliceAxis(be, t.Outputs["out"], t.Inputs["src"], 2, 2, 1), _ => true, required, Acceptance.Exact));
            cases.Add(new(Family, "write_vae_frame_bias_" + suffix, seed => new Operands(seed, dtype).In("acc", new TensorShape(2, 3, 2, 3)).In("out_init", new TensorShape([2, 3, 4, 2, 3])).In("bias", new TensorShape(3), DType.F32).Out("out", new TensorShape([2, 3, 4, 2, 3])).Tensors,
                (be, t) => { Prime(t); WriteFrame(be, t.Outputs["out"], t.Inputs["acc"], t.Inputs["bias"], 1); }, _ => true, required, Acceptance.Exact));
            cases.Add(new(Family, "fill_bias_" + suffix, seed => new Operands(seed, dtype).In("bias", new TensorShape(3), DType.F32).Out("out", new TensorShape([1, 3, 4, 2, 3])).Tensors,
                (be, t) => FillBias(be, t.Outputs["out"], t.Inputs["bias"]), _ => true, required, Acceptance.Exact));
            cases.Add(new(Family, "accumulate_tap_dt1_stride1_" + suffix, seed => new Operands(seed, dtype).In("conv", new TensorShape(5, 3, 2, 3)).In("out_init", new TensorShape([1, 3, 2, 2, 3])).Out("out", new TensorShape([1, 3, 2, 2, 3])).Tensors,
                (be, t) => { Prime(t); AccumulateTap(be, t.Outputs["out"], t.Inputs["conv"], 1, 1); }, _ => true, required, Acceptance.Exact));
            cases.Add(new(Family, "seedvr2_pad_bottom_right_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 2, 2, 3, 4])).Out("out", new TensorShape([1, 2, 2, 4, 5])).Tensors,
                (be, t) => be.Pad(t.Outputs["out"], t.Inputs["x"], new PadSpec(PadMode.Zero, [0, 0, 0, 0, 0], [0, 0, 0, 1, 1])), _ => true, required, Acceptance.Exact));
        }
        cases.Add(new(Family, "write_vae_frame_nobias_f32", seed => new Operands(seed).In("acc", new TensorShape(2, 3, 2, 3)).In("out_init", new TensorShape([2, 3, 4, 2, 3])).Out("out", new TensorShape([2, 3, 4, 2, 3])).Tensors,
            (be, t) => { Prime(t); WriteFrame(be, t.Outputs["out"], t.Inputs["acc"], null, 3); }, _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "fill_bias_null_f32", seed => new Operands(seed).Out("out", new TensorShape([1, 3, 4, 2, 3])).Tensors,
            (be, t) => FillBias(be, t.Outputs["out"], null), _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "accumulate_tap_dt2_stride2_f32", seed => new Operands(seed).In("conv", new TensorShape(5, 3, 2, 3)).In("out_init", new TensorShape([1, 3, 2, 2, 3])).Out("out", new TensorShape([1, 3, 2, 2, 3])).Tensors,
            (be, t) => { Prime(t); AccumulateTap(be, t.Outputs["out"], t.Inputs["conv"], 2, 2); }, _ => true, All, Acceptance.Exact));

        // Ideogram 4 conditional latent: zero text rows before the projected image rows.
        cases.Add(new(Family, "scatter_rows_after", seed => new Operands(seed).In("img", new TensorShape(1, 4, 6)).Out("out", new TensorShape(1, 7, 6)).Tensors,
            (be, t) => be.Pad(t.Outputs["out"], t.Inputs["img"], PadSpec.Axis(PadMode.Zero, 3, 1, 3, 0)), _ => true, All, Acceptance.Exact));
        return cases;
    }

    /// <summary>Axis order of the Oasis head split: <c>[frame, position, head, dim]</c> to <c>[batch, head, seq, dim]</c>.</summary>
    private static ReadOnlySpan<int> OasisOrder(bool temporal) => temporal ? [1, 2, 0, 3] : [0, 2, 1, 3];

    /// <summary>CausalConv3d's frame write: scatter the frame into temporal slot <paramref name="to"/>, then add the per-channel bias in place.</summary>
    private static void WriteFrame(IBackend be, Tensor output, Tensor acc, Tensor? bias, int to)
    {
        StridedView slot = StridedView.Of(output.Shape).Slice(2, to, 1);
        be.CopyStrided(output, acc, StridedCopySpec.Scatter(slot));
        if (bias is null) return;
        Span<long> slotDims = stackalloc long[5];
        slot.CopyDimsTo(slotDims);
        be.AccumulateStrided(output, bias, new StridedCopySpec(slot, StridedView.VectorAlong(slotDims, 1)));
    }

    /// <summary>CausalConv3d's output start: the per-channel bias, or zero, broadcast over every position. A BF16 output
    /// starts from zero and adds the F32 bias, so the bias is rounded once.</summary>
    private static void FillBias(IBackend be, Tensor output, Tensor? bias)
    {
        Span<long> dims = stackalloc long[5];
        output.Shape.CopyDimsTo(dims);
        using Tensor zero = Seeded.Zeros(new TensorShape(1), output.DType);
        if (bias is null)
        {
            be.CopyStrided(output, zero, StridedCopySpec.Gather(new StridedView(dims, stackalloc long[5])));
            return;
        }
        if (output.DType == DType.F32)
        {
            be.CopyStrided(output, bias, StridedCopySpec.Gather(StridedView.VectorAlong(dims, 1)));
            return;
        }
        be.CopyStrided(output, zero, StridedCopySpec.Gather(new StridedView(dims, stackalloc long[5])));
        be.AccumulateStrided(output, bias, StridedCopySpec.Gather(StridedView.VectorAlong(dims, 1)));
    }

    /// <summary>CausalConv3d's tap accumulate: <c>out[co, to] += conv[to·strideT + dt, co]</c>.</summary>
    private static void AccumulateTap(IBackend be, Tensor output, Tensor conv, int dt, int strideT)
    {
        long tout = output.Shape[2];
        StridedView taps = StridedView.Of(output.Shape).Select(0, 0).Permute([1, 0, 2, 3]);
        be.AccumulateStrided(output, conv, new StridedCopySpec(taps, StridedView.Of(conv.Shape).Slice(0, dt, tout, strideT)));
    }

    /// <summary>Copies the recorded initial contents into the in-place output before the op runs; <c>check</c> clears outputs first.</summary>
    private static void Prime(OpTensors t) => t.Inputs["out_init"].AsReadOnlySpan<byte>().CopyTo(t.Outputs["out"].AsSpan<byte>());

    /// <summary>Builds one case's operands: seeded inputs (a distinct salt each) and zeroed outputs, in one storage dtype unless an input names its own.</summary>
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
    }
}
