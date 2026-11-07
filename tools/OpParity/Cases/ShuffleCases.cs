using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.OpParity.Cases;

/// <summary>The depth-to-space, space-to-depth, patchify and unpatchify moves the models make, one case per caller
/// parameter set, recorded against the current implementations. Every case is a pure permutation (one op also
/// drops a frame), so the rule is byte equality.</summary>
public static class ShuffleCases
{
    public const string Family = "shuffle";

    private static readonly string[] All = ["cpu", "cuda", "vulkan"];
    private static readonly string[] Cuda = ["cuda"];

    public static IReadOnlyList<OpCase> Cases()
    {
        List<OpCase> cases = [];

        // LTX latent upsampler: per-depth-slice nn.PixelShuffle, channel-outer nesting. [1, C·r², D, H, W] → [1, C, D, rH, rW].
        cases.Add(new(Family, "pixel_shuffle2d_r2", seed => new Operands(seed).In("x", new TensorShape([1, 8, 2, 3, 2])).Out("out", new TensorShape([1, 2, 2, 6, 4])).Tensors,
            (be, t) => be.PixelShuffle2d(t.Outputs["out"], t.Inputs["x"], 2), _ => true, All, Acceptance.Exact));
        // TRELLIS SS-VAE upsample: [1, C·r³, D, H, W] → [1, C, rD, rH, rW].
        cases.Add(new(Family, "pixel_shuffle3d_r2", seed => new Operands(seed).In("x", new TensorShape([1, 16, 2, 2, 3])).Out("out", new TensorShape([1, 2, 4, 4, 6])).Tensors,
            (be, t) => be.PixelShuffle3d(t.Outputs["out"], t.Inputs["x"], 2), _ => true, All, Acceptance.Exact));

        // SeedVR2 MAGViT shuffle: spatial only, and temporal with output frame 1 dropped. F32 everywhere; BF16 on CUDA.
        foreach ((string suffix, DType dtype) in new[] { ("f32", DType.F32), ("bf16", DType.BF16) })
        {
            Func<IBackend, bool> serves = be => dtype == DType.F32 || be.Device.IsCuda;
            string[] required = dtype == DType.F32 ? All : Cuda;
            cases.Add(new(Family, "seedvr2_pixel_shuffle_spatial_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 8, 2, 2, 3])).Out("out", new TensorShape([1, 2, 2, 4, 6])).Tensors,
                (be, t) => be.SeedVr2PixelShuffle(t.Outputs["out"], t.Inputs["x"], 2, 1), serves, required, Acceptance.Exact));
            cases.Add(new(Family, "seedvr2_pixel_shuffle_temporal_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 16, 2, 2, 3])).Out("out", new TensorShape([1, 2, 3, 4, 6])).Tensors,
                (be, t) => be.SeedVr2PixelShuffle(t.Outputs["out"], t.Inputs["x"], 2, 2), serves, required, Acceptance.Exact));
        }

        // LTX-2.5 channels-last 3-D shuffle: t=3, h=w=2, strides (2,2,2), outC=3 (projected rows [t·h·w, 24]); the causal
        // variant drops the duplicated leading frame; chunked calls fill one persistent output.
        cases.Add(new(Family, "ltx25_pixel_shuffle_drop", seed => new Operands(seed).In("proj", new TensorShape(12, 24)).Out("out", new TensorShape(80, 3)).Tensors,
            (be, t) => be.Ltx25PixelShuffle(t.Outputs["out"], t.Inputs["proj"], 0, 2, 2, 2, 2, 2, 3, 1, 4, 4), _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "ltx25_pixel_shuffle_drop_chunked", seed => new Operands(seed).In("proj0", new TensorShape(8, 24)).In("proj1", new TensorShape(4, 24)).Out("out", new TensorShape(80, 3)).Tensors,
            (be, t) =>
            {
                be.Ltx25PixelShuffle(t.Outputs["out"], t.Inputs["proj0"], 0, 2, 2, 2, 2, 2, 3, 1, 4, 4);
                be.Ltx25PixelShuffle(t.Outputs["out"], t.Inputs["proj1"], 2, 2, 2, 2, 2, 2, 3, 1, 4, 4);
            },
            _ => true, All, Acceptance.Exact));
        cases.Add(new(Family, "ltx25_pixel_shuffle_spatial", seed => new Operands(seed).In("proj", new TensorShape(12, 12)).Out("out", new TensorShape(48, 3)).Tensors,
            (be, t) => be.Ltx25PixelShuffle(t.Outputs["out"], t.Inputs["proj"], 0, 2, 2, 1, 2, 2, 3, 0, 4, 4), _ => true, All, Acceptance.Exact));

        // Wan 2.2 VAE DupUp3D ([1, inC, 2, 2, 2] → [1, outC, 2·fT − dropT, 4, 4]) and unpatchify; both copy by element width.
        foreach ((string suffix, DType dtype) in new[] { ("f32", DType.F32), ("bf16", DType.BF16) })
        {
            cases.Add(new(Family, "dup_up3d_rep8_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 4, 2, 2, 2])).Out("out", new TensorShape([1, 4, 4, 4, 4])).Tensors,
                (be, t) => be.DupUp3dVae(t.Outputs["out"], t.Inputs["x"], 2, 2, 0), _ => true, All, Acceptance.Exact));
            cases.Add(new(Family, "dup_up3d_rep8_first_chunk_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 4, 2, 2, 2])).Out("out", new TensorShape([1, 4, 3, 4, 4])).Tensors,
                (be, t) => be.DupUp3dVae(t.Outputs["out"], t.Inputs["x"], 2, 2, 1), _ => true, All, Acceptance.Exact));
            cases.Add(new(Family, "dup_up3d_rep4_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 8, 2, 2, 2])).Out("out", new TensorShape([1, 4, 4, 4, 4])).Tensors,
                (be, t) => be.DupUp3dVae(t.Outputs["out"], t.Inputs["x"], 2, 2, 0), _ => true, All, Acceptance.Exact));
            cases.Add(new(Family, "unpatchify_vae_p2_" + suffix, seed => new Operands(seed, dtype).In("x", new TensorShape([1, 8, 2, 2, 3])).Out("out", new TensorShape([1, 2, 2, 4, 6])).Tensors,
                (be, t) => be.UnpatchifyVae(t.Outputs["out"], t.Inputs["x"], 2), _ => true, All, Acceptance.Exact));
        }

        // DiT token grids: [B, hP·wP, C·p²] ↔ [B, C, hP·p, wP·p], both token packings, F32 and F16.
        foreach ((string suffix, DType dtype) in new[] { ("f32", DType.F32), ("f16", DType.F16) })
        {
            foreach ((string order, bool fastest) in new[] { ("inner_channel", true), ("channel_outer", false) })
            {
                cases.Add(new(Family, $"unpatchify_tokens_{order}_{suffix}", seed => new Operands(seed, dtype).In("tokens", new TensorShape(2, 6, 12)).Out("out", new TensorShape(2, 3, 4, 6)).Tensors,
                    (be, t) => be.UnpatchifyTokens(t.Outputs["out"], t.Inputs["tokens"], 3, 2, 3, 2, fastest), _ => true, All, Acceptance.Exact));
                cases.Add(new(Family, $"patchify_tokens_{order}_{suffix}", seed => new Operands(seed, dtype).In("x", new TensorShape(2, 3, 4, 6)).Out("out", new TensorShape(2, 6, 12)).Tensors,
                    (be, t) => be.PatchifyTokens(t.Outputs["out"], t.Inputs["x"], 2, fastest), _ => true, All, Acceptance.Exact));
            }
        }

        // Oasis per-frame unpatchify: proj [t·gh·gw, c·p²] with inner (py·p+px)·c+ci → [t, c, gh·p, gw·p].
        cases.Add(new(Family, "oasis_unpatchify", seed => new Operands(seed).In("proj", new TensorShape(12, 12)).Out("out", new TensorShape(2, 3, 4, 6)).Tensors,
            (be, t) => be.OasisUnpatchify(t.Outputs["out"], t.Inputs["proj"], 2, 3, 2, 3, 2), _ => true, All, Acceptance.Exact));
        return cases;
    }

    /// <summary>Builds one case's operands: seeded inputs (a distinct salt each) and zeroed outputs, in one storage dtype.</summary>
    private sealed class Operands(int seed, DType? dtype = null)
    {
        private readonly DType _dtype = dtype ?? DType.F32;
        private int _salt = 1;

        public OpTensors Tensors { get; } = new();

        public Operands In(string name, TensorShape shape)
        {
            Tensors.Inputs[name] = Seeded.Of(shape, _dtype, seed, _salt++);
            return this;
        }

        public Operands Out(string name, TensorShape shape)
        {
            Tensors.Outputs[name] = Seeded.Zeros(shape, _dtype);
            return this;
        }
    }
}
