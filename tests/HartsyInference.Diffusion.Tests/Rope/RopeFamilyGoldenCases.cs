using System.Security.Cryptography;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Tests.Rope;

/// <summary>The table-building and rotation calls of every Diffusion RoPE family on the shared Core tables, each reduced to
/// SHA-256 digests of its output bits. Rotations run on a fixed pseudo-random Q/K; device paths run on the CPU backend, whose
/// rope ops are the reference the GPU kernels are tested against. Only public (or test-visible) calls that exist both before
/// and after the Core RoPE migration are used, so the same cases produce the golden fixture from the pre-migration source.</summary>
internal static unsafe class RopeFamilyGoldenCases
{
    private const int Heads = 2;
    private const int KvHeads = 1;
    private const int GridH = 8;
    private const int GridW = 6;
    private const int TextLength = 7;

    /// <summary>Every case, in fixture order.</summary>
    public static IReadOnlyList<string> Names { get; } =
    [
        "qwen_apply_image", "qwen_apply_text", "qwen_apply_joint_refs", "qwen_apply_joint_unroped_text",
        "qwen_device_joint_refs", "qwen_device_joint_slice",
        "qwen21_device_text", "qwen21_device_image",
        "hunyuan_apply_image", "hunyuan_apply_video_joint", "hunyuan_gpu", "hunyuan_video_gpu_with_text",
        "zimage_forward", "zimage_gpu", "zimage_gpu_gqa",
        "kandinsky5_apply_3d", "kandinsky5_gpu_3d", "kandinsky5_apply_1d",
        "moss_joint", "moss_text",
        "stableaudio_tables",
        "omnigen2_host_tables", "omnigen2_staged_tables", "omnigen2_positions_expanded", "omnigen2_apply_joint",
        "lens_staged_joint", "lens_apply_joint",
        "anima_freqs", "anima_device_tables",
        "flux_forward", "flux_gpu", "flux_gpu_gqa_kontext",
        "hidream_gpu",
        "wan_shared", "wan_offsets", "wan_frame_indices", "wan_sigma_theta", "wan_explicit_split", "wan_temporal_only",
        "wan_apply_shared", "wan_apply_per_head",
        "ernie_freqs_batch2", "ernie_gpu",
        "ideogram4_tables", "ideogram4_sections_overrun", "ideogram4_apply",
        "mrope3d_tables", "mrope3d_apply",
    ];

    /// <summary>Runs <paramref name="name"/> and returns its digests as space-separated <c>label=hex</c> pairs.</summary>
    public static string Run(string name)
    {
        using CpuBackend cpu = new();
        int qwenStart = QwenImageRope.ComputeTextPositionStart(GridH, GridW);
        (int H, int W)[] refGrids = [(4, 4)];
        switch (name)
        {
            case "qwen_apply_image":
                return Rotate(128, GridH * GridW, (q, k) => new QwenImageRope().ApplyImage(q, k, 1, Heads, GridH, GridW));
            case "qwen_apply_text":
                return Rotate(128, TextLength, (q, k) => new QwenImageRope().ApplyText(q, k, 1, Heads, TextLength, qwenStart));
            case "qwen_apply_joint_refs":
                return Rotate(128, TextLength + GridH * GridW + 16,
                    (q, k) => new QwenImageRope().ApplyJoint(q, k, 1, Heads, GridH, GridW, TextLength, qwenStart, refGrids));
            case "qwen_apply_joint_unroped_text":
                return Rotate(128, TextLength + GridH * GridW,
                    (q, k) => new QwenImageRope(ropeText: false).ApplyJoint(q, k, 1, Heads, GridH, GridW, TextLength, qwenStart));
            case "qwen_device_joint_refs":
            {
                QwenImageRope rope = new();
                (Tensor cos, Tensor sin) = rope.GetOrBuildJointTables(cpu, GridH, GridW, TextLength, qwenStart, refGrids);
                return DigestsAndRelease(rope, ("cos", cos), ("sin", sin));
            }
            case "qwen_device_joint_slice":
            {
                QwenImageRope rope = new();
                (Tensor cos, Tensor sin) = rope.GetOrBuildJointTables(cpu, GridH, GridW, TextLength, qwenStart, default,
                    imgTokenStart: 12, imgTokenCount: 18);
                return DigestsAndRelease(rope, ("cos", cos), ("sin", sin));
            }
            case "qwen21_device_text":
            {
                QwenImage21Rope rope = new();
                (Tensor cos, Tensor sin) = rope.GetOrBuildTextTables(cpu, TextLength);
                return DigestsAndRelease(rope, ("cos", cos), ("sin", sin));
            }
            case "qwen21_device_image":
            {
                QwenImage21Rope rope = new();
                (Tensor cos, Tensor sin) = rope.GetOrBuildImageTables(cpu, TextLength, GridH, GridW);
                return DigestsAndRelease(rope, ("cos", cos), ("sin", sin));
            }
            case "hunyuan_apply_image":
                return Rotate(128, GridH * GridW, (q, k) => new HunyuanImageRope().ApplyImage(q, k, 1, Heads, GridH, GridW));
            case "hunyuan_apply_video_joint":
                return Rotate(128, 2 * GridH * GridW,
                    (q, k) => new HunyuanImageRope([16, 56, 56], 256f).ApplyJoint(q, k, 1, Heads, [2, GridH, GridW]));
            case "hunyuan_gpu":
                return RotatePrePermute(128, GridH * GridW, Heads,
                    (q, k) => new HunyuanImageRope().ApplyGpu(cpu, q, k, Heads, [GridH, GridW]));
            case "hunyuan_video_gpu_with_text":
                return RotatePrePermute(128, 2 * GridH * GridW + TextLength, Heads,
                    (q, k) => new HunyuanImageRope([16, 56, 56], 256f).ApplyGpu(cpu, q, k, Heads, [2, GridH, GridW]));
            case "zimage_forward":
            {
                ZImageRope rope = PrecomputedZImage();
                return Rotate(128, 32 + 48, (q, k) => rope.Forward(q, k, 1, Heads, 32 + 48));
            }
            case "zimage_gpu":
            {
                ZImageRope rope = PrecomputedZImage();
                return RotatePrePermute(128, 32 + 48, Heads, (q, k) => rope.ApplyGpu(cpu, q, k, Heads));
            }
            case "zimage_gpu_gqa":
            {
                ZImageRope rope = PrecomputedZImage();
                return RotatePrePermuteGqa(128, 32 + 48, (q, k) => rope.ApplyGpuGqa(cpu, q, k, Heads, KvHeads));
            }
            case "kandinsky5_apply_3d":
            {
                Kandinsky5Rope rope = new(128);
                rope.Precompute3D([16, 56, 56], 1, GridH, GridW, 0, 1f, 2f, 2f);
                return Rotate(128, GridH * GridW, (q, k) => rope.Apply(q, k, 1, Heads, GridH * GridW));
            }
            case "kandinsky5_gpu_3d":
            {
                Kandinsky5Rope rope = new(128);
                rope.Precompute3D([16, 56, 56], 2, GridH, GridW, 3, 1.5f, 2f, 2f);
                return RotatePrePermute(128, 2 * GridH * GridW, Heads, (q, k) => rope.ApplyGpu(cpu, q, k, Heads, 2 * GridH * GridW));
            }
            case "kandinsky5_apply_1d":
            {
                Kandinsky5Rope rope = new(128);
                rope.Precompute1D([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
                return Rotate(128, 11, (q, k) => rope.Apply(q, k, 1, Heads, 11));
            }
            case "moss_joint":
            {
                (Tensor cos, Tensor sin) = MossImageRope.Joint(GridH, GridW, TextLength, 128, 10000f);
                return DigestsAndDispose(cos, sin);
            }
            case "moss_text":
            {
                (Tensor cos, Tensor sin) = MossImageRope.Text(TextLength, 128, 10000f);
                return DigestsAndDispose(cos, sin);
            }
            case "stableaudio_tables":
            {
                float[] invFreq = new float[16];
                for (int i = 0; i < invFreq.Length; i++) invFreq[i] = 1f / MathF.Pow(10000f, i / 16f);
                Tensor cos = StableAudioRope.BuildCosTable(invFreq, 21, 64);
                Tensor sin = StableAudioRope.BuildSinTable(invFreq, 21, 64);
                return DigestsAndDispose(cos, sin);
            }
            case "omnigen2_host_tables":
            {
                OmniGen2Rope rope = new();
                (float[] textCos, float[] textSin) = rope.BuildTextTable(TextLength);
                (float[] imageCos, float[] imageSin) = rope.BuildImageTable(GridH, GridW, TextLength);
                (float[] jointCos, float[] jointSin) = rope.BuildJointTable(TextLength, GridH, GridW);
                return string.Join(' ', Hex("text_cos", textCos), Hex("text_sin", textSin), Hex("image_cos", imageCos),
                    Hex("image_sin", imageSin), Hex("joint_cos", jointCos), Hex("joint_sin", jointSin));
            }
            case "omnigen2_staged_tables":
            {
                using OmniGen2Rope rope = new();
                (Tensor textCos, Tensor textSin) = rope.GetOrBuildTextTables(TextLength);
                (Tensor imageCos, Tensor imageSin) = rope.GetOrBuildImageTables(GridH, GridW, TextLength);
                (Tensor jointCos, Tensor jointSin) = rope.GetOrBuildJointTables(TextLength, GridH, GridW);
                return Digests(("text_cos", textCos), ("text_sin", textSin), ("image_cos", imageCos), ("image_sin", imageSin),
                    ("joint_cos", jointCos), ("joint_sin", jointSin));
            }
            case "omnigen2_positions_expanded":
            {
                OmniGen2Rope rope = new([4, 6, 6], 256);
                int[] t = new int[11], h = new int[11], w = new int[11];
                for (int s = 0; s < 11; s++) { t[s] = s; h[s] = (s * 3) % 5; w[s] = (s * 7) % 11; }
                (float[] cos, float[] sin) = rope.BuildTableFromPositions(t, h, w);
                (Tensor cosRows, Tensor sinRows) = rope.ExpandToDeviceTables(cos, sin);
                return string.Join(' ', Hex("cos", cos), Hex("sin", sin), DigestsAndDispose(cosRows, sinRows));
            }
            case "omnigen2_apply_joint":
                return Rotate(120, TextLength + GridH * GridW, (q, k) =>
                    new OmniGen2Rope().ApplyJoint(q, k, 1, Heads, Heads, TextLength, GridH, GridW));
            case "lens_staged_joint":
            {
                LensRope rope = new();
                (Tensor cos, Tensor sin) = rope.GetOrBuildJointTables(GridH, GridW, TextLength,
                    LensRope.ComputeTextPositionStart(GridH, GridW));
                return DigestsAndRelease(rope, ("cos", cos), ("sin", sin));
            }
            case "lens_apply_joint":
                return Rotate(64, GridH * GridW + TextLength, (q, k) => new LensRope().ApplyJoint(q, k, 1, Heads, GridH, GridW,
                    TextLength, LensRope.ComputeTextPositionStart(GridH, GridW)));
            case "anima_freqs":
            {
                AnimaRope rope = new(128, 10000f, (1f, 4f, 4f));
                (Tensor cos, Tensor sin) = rope.BuildFreqs(2, GridH, GridW);
                return DigestsAndDispose(cos, sin);
            }
            case "anima_device_tables":
            {
                AnimaRope rope = new(128, 10000f, (1f, 4f, 4f));
                (Tensor cos, Tensor sin) = rope.GetOrCreateTables(cpu, 2, GridH, GridW);
                string digests = Digests(("cos", cos), ("sin", sin));
                rope.ReleaseGpuTables(cpu);
                return digests;
            }
            case "flux_forward":
            {
                FluxRope rope = PrecomputedFlux(kontext: false);
                return Rotate(128, TextLength + GridH * GridW, (q, k) => rope.Forward(q, k, 1, Heads, TextLength + GridH * GridW));
            }
            case "flux_gpu":
            {
                FluxRope rope = PrecomputedFlux(kontext: false);
                return RotatePrePermute(128, TextLength + GridH * GridW, Heads, (q, k) => rope.ApplyGpu(cpu, q, k, Heads));
            }
            case "flux_gpu_gqa_kontext":
            {
                FluxRope rope = PrecomputedFlux(kontext: true);
                return RotatePrePermuteGqa(128, TextLength + GridH * GridW + 4 * 4,
                    (q, k) => rope.ApplyGpuGqa(cpu, q, k, Heads, KvHeads));
            }
            case "hidream_gpu":
            {
                HiDreamRope rope = new();
                Tensor ids = HiDreamRope.BuildPositionIds(GridH * GridW, GridH, GridW, TextLength);
                rope.Precompute(ids);
                ids.Dispose();
                return RotatePrePermute(128, GridH * GridW + TextLength, Heads, (q, k) => rope.ApplyGpu(cpu, q, k, Heads));
            }
            case "wan_shared":
            {
                (Tensor cos, Tensor sin) = new WanRope(128, 10000, 1024).BuildCosSin(2, GridH, GridW);
                return DigestsAndDispose(cos, sin);
            }
            case "wan_offsets":
            {
                (Tensor cos, Tensor sin) = new WanRope(128, 10000, 1024).BuildCosSin(2, GridH, GridW, 1, 0, GridW);
                return DigestsAndDispose(cos, sin);
            }
            case "wan_frame_indices":
            {
                (Tensor cos, Tensor sin) = new WanRope(128, 10000, 1024).BuildCosSin([0, 3, 7, 21], 4, 4, 2, 5);
                return DigestsAndDispose(cos, sin);
            }
            case "wan_sigma_theta":
            {
                (Tensor cos, Tensor sin) = new WanRope(128, 10000, 1024, sigmaTheta: 0.1, numHeads: 4).BuildCosSin(2, 4, 4);
                return DigestsAndDispose(cos, sin);
            }
            case "wan_explicit_split":
            {
                (Tensor cos, Tensor sin) = new WanRope(8, 28, 28, 256).BuildCosSin(3, 4, 4);
                return DigestsAndDispose(cos, sin);
            }
            case "wan_temporal_only":
            {
                (Tensor cos, Tensor sin) = new WanRope(64, 0, 0, 10000).BuildCosSin(9, 1, 1);
                return DigestsAndDispose(cos, sin);
            }
            case "wan_apply_shared":
            {
                WanRope rope = new(128, 10000, 1024);
                (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 4, 4);
                string digests = RotateShapes(new TensorShape(32, Heads, 128), new TensorShape(32, Heads, 128), (q, k) =>
                {
                    rope.ApplyRotary(q, cos, sin, Heads);
                    rope.ApplyRotary(k, cos, sin, Heads);
                });
                cos.Dispose();
                sin.Dispose();
                return digests;
            }
            case "wan_apply_per_head":
            {
                WanRope rope = new(128, 10000, 1024, sigmaTheta: 0.1, numHeads: Heads);
                (Tensor cos, Tensor sin) = rope.BuildCosSin(2, 4, 4);
                string digests = RotateShapes(new TensorShape(32, Heads, 128), new TensorShape(32, Heads, 128), (q, k) =>
                {
                    rope.ApplyRotary(q, cos, sin, Heads);
                    rope.ApplyRotary(k, cos, sin, Heads);
                });
                cos.Dispose();
                sin.Dispose();
                return digests;
            }
            case "ernie_freqs_batch2":
            {
                Tensor ids = ErnieImageRope.BuildPositionIds(2, [5, 7], GridH, GridW, 9);
                Tensor freqs = new ErnieImageRope().BuildFreqs(ids);
                ids.Dispose();
                string digests = Digests(("freqs", freqs));
                freqs.Dispose();
                return digests;
            }
            case "ernie_gpu":
            {
                ErnieImageRope rope = new();
                Tensor ids = ErnieImageRope.BuildPositionIds(1, [7], GridH, GridW, 9);
                Tensor freqs = rope.BuildFreqs(ids);
                ids.Dispose();
                string digests = RotatePrePermute(128, GridH * GridW + 9, Heads, (q, k) => rope.ApplyRotaryEmbGpu(cpu, q, k, freqs));
                freqs.Dispose();
                return digests;
            }
            case "ideogram4_tables":
            {
                Tensor ids = SeededPositions(2, 11);
                (Tensor cos, Tensor sin) = new Ideogram4Mrope(128, 5_000_000, (24, 20, 20)).BuildCosSin(ids);
                ids.Dispose();
                return DigestsAndDispose(cos, sin);
            }
            case "ideogram4_sections_overrun":
            {
                Tensor ids = SeededPositions(1, 9);
                (Tensor cos, Tensor sin) = new Ideogram4Mrope(16, 10000, (2, 4, 4)).BuildCosSin(ids);
                ids.Dispose();
                return DigestsAndDispose(cos, sin);
            }
            case "ideogram4_apply":
            {
                Ideogram4Mrope rope = new(128, 5_000_000, (24, 20, 20));
                Tensor ids = SeededPositions(1, 11);
                (Tensor cos, Tensor sin) = rope.BuildCosSin(ids);
                ids.Dispose();
                string digests = RotateShapes(new TensorShape(1, 11, Heads, 128), new TensorShape(1, 11, Heads, 128),
                    (q, k) => rope.ApplyRotary(q, k, cos, sin));
                cos.Dispose();
                sin.Dispose();
                return digests;
            }
            case "mrope3d_tables":
            {
                Tensor ids = SeededPositions(13, 1, rank2: true);
                (Tensor cos, Tensor sin) = new Multimodal3DRope(128, 1_000_000f, (16, 24, 24)).BuildCosSin(ids);
                ids.Dispose();
                return DigestsAndDispose(cos, sin);
            }
            case "mrope3d_apply":
            {
                Multimodal3DRope rope = new(128, 1_000_000f, (16, 24, 24));
                Tensor ids = SeededPositions(13, 1, rank2: true);
                (Tensor cos, Tensor sin) = rope.BuildCosSin(ids);
                ids.Dispose();
                string digests = RotateShapes(new TensorShape(13, Heads, 128), new TensorShape(13, Heads, 128), (q, k) =>
                {
                    rope.ApplyRotary(q, cos, sin);
                    rope.ApplyRotary(k, cos, sin);
                });
                cos.Dispose();
                sin.Dispose();
                return digests;
            }
            default:
                throw new ArgumentException($"Unknown RoPE golden case '{name}'.", nameof(name));
        }
    }

    /// <summary>Fixed <c>(t, h, w)</c> positions, <c>[batch, seq, 3]</c> (or <c>[seq, 3]</c>), mixing integers, fractions and
    /// negatives the way M-RoPE callers produce them.</summary>
    private static Tensor SeededPositions(int a, int b, bool rank2 = false)
    {
        int tokens = a * b;
        Tensor ids = rank2 ? new(new TensorShape(tokens, 3), DType.F32) : new(new TensorShape(a, b, 3), DType.F32);
        float* data = (float*)ids.DataPointer;
        for (int t = 0; t < tokens; t++)
        {
            data[3 * t] = t;
            data[3 * t + 1] = (t % 5) - 2.5f;
            data[3 * t + 2] = (t * 7 % 11) * 0.5f;
        }
        return ids;
    }

    private static ZImageRope PrecomputedZImage()
    {
        ZImageRope rope = new();
        Tensor ids = ZImageRope.BuildPositionIds(32, GridH, GridW, 48);
        rope.Precompute(ids);
        ids.Dispose();
        return rope;
    }

    private static FluxRope PrecomputedFlux(bool kontext)
    {
        FluxRope rope = new([16, 56, 56], 10000);
        Tensor ids = kontext
            ? FluxRope.BuildPositionIdsKontext(TextLength, GridH, GridW, 4, 4)
            : FluxRope.BuildPositionIds(TextLength, GridH, GridW);
        rope.Precompute(ids);
        ids.Dispose();
        return rope;
    }

    /// <summary>Rotates a fixed head-major <c>[1, Heads, seq, headDim]</c> Q/K pair in place and digests both.</summary>
    private static string Rotate(int headDim, int seq, Action<Tensor, Tensor> apply) =>
        RotateShapes(new TensorShape(1, Heads, seq, headDim), new TensorShape(1, Heads, seq, headDim), apply);

    /// <summary>Rotates a fixed pre-permute <c>[1, seq, heads, headDim]</c> Q/K pair in place and digests both.</summary>
    private static string RotatePrePermute(int headDim, int seq, int heads, Action<Tensor, Tensor> apply) =>
        RotateShapes(new TensorShape(1, seq, heads, headDim), new TensorShape(1, seq, heads, headDim), apply);

    /// <summary>Pre-permute rotation with fewer K heads than Q heads.</summary>
    private static string RotatePrePermuteGqa(int headDim, int seq, Action<Tensor, Tensor> apply) =>
        RotateShapes(new TensorShape(1, seq, Heads, headDim), new TensorShape(1, seq, KvHeads, headDim), apply);

    private static string RotateShapes(TensorShape qShape, TensorShape kShape, Action<Tensor, Tensor> apply)
    {
        using Tensor q = Seeded(qShape, 0x9E3779B9u);
        using Tensor k = Seeded(kShape, 0x7F4A7C15u);
        apply(q, k);
        return Digests(("q", q), ("k", k));
    }

    private static Tensor Seeded(TensorShape shape, uint seed)
    {
        Tensor tensor = new(shape, DType.F32);
        float* data = (float*)tensor.DataPointer;
        uint state = seed;
        for (long i = 0; i < tensor.ElementCount; i++)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = ((state >> 8) & 0xFFFF) / 32768f - 1f;
        }
        return tensor;
    }

    /// <summary>Digests tables a family owns, then releases the family. Some families gained <see cref="IDisposable"/> in the
    /// migration, so the release goes through <see cref="object"/> to compile against both.</summary>
    private static string DigestsAndRelease(object owner, params (string Label, Tensor Tensor)[] tensors)
    {
        string digests = Digests(tensors);
        (owner as IDisposable)?.Dispose();
        return digests;
    }

    private static string DigestsAndDispose(Tensor cos, Tensor sin)
    {
        string digests = Digests(("cos", cos), ("sin", sin));
        cos.Dispose();
        sin.Dispose();
        return digests;
    }

    private static string Digests(params (string Label, Tensor Tensor)[] tensors) =>
        string.Join(' ', tensors.Select(entry => Hex(entry.Label,
            new ReadOnlySpan<byte>(entry.Tensor.DataPointer, checked((int)(entry.Tensor.ElementCount * sizeof(float)))))));

    private static string Hex(string label, float[] values) =>
        Hex(label, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()));

    private static string Hex(string label, ReadOnlySpan<byte> bytes) => $"{label}={Convert.ToHexStringLower(SHA256.HashData(bytes))}";
}
