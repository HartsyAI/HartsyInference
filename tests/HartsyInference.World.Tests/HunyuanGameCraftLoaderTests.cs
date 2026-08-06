using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Registry;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.World.Models;
using HartsyInference.World.Pipelines;
using Xunit;

namespace HartsyInference.World.Tests;

/// <summary>Code-only verification of the Hunyuan-GameCraft checkpoint loader (<see cref="HunyuanGameCraftPipeline.LoadFromPath"/>
/// + <see cref="WorldService"/> wiring) added to make <c>hunyuan-gamecraft</c> loadable. No real checkpoint is
/// available on this box (the ~51GB set exceeds the ~31GB free at the time this was written — see
/// <c>docs/Checklists/MODEL_STATUS_WORLD.md</c>), so these tests prove the two things that don't need real weight
/// bytes: (1) the two-stage checkpoint conversion (<see cref="HunyuanGameCraftCheckpointConverter"/>'s coarse prefix
/// router, chained into <c>HunyuanVideoCheckpointConverter</c>'s Tencent-raw→hybrid-naming remap) produces the exact
/// keys <c>HunyuanVideoDit.LoadWeights</c>/<c>GameCraftCameraNet.LoadWeights</c> index, on a synthetic dict shaped
/// like the real <c>mp_rank_00_model_states.pt</c>; and (2) every new code path fails with a clean, specific
/// exception (not a crash) when pointed at paths that don't exist.</summary>
public sealed unsafe class HunyuanGameCraftLoaderTests
{
    [Fact]
    public void Converter_ChainsRawTencentDump_ToHybridDitNaming_AndRoutesCameraNet()
    {
        const int h = 8, patchVec = 33 * 1 * 2 * 2, outVec = 16 * 1 * 2 * 2;
        Dictionary<string, Tensor> raw = new()
        {
            // Head, in raw Tencent naming (img_in.proj, not img_in.weight — MapOriginal reshapes the Conv3d).
            ["img_in.proj.weight"] = T(h, patchVec), ["img_in.proj.bias"] = T(h),
            ["final_layer.linear.weight"] = T(outVec, h), ["final_layer.linear.bias"] = T(outVec),
            ["final_layer.adaLN_modulation.1.weight"] = T(2 * h, h), ["final_layer.adaLN_modulation.1.bias"] = T(2 * h),
            // One double block, fused QKV under the raw `_attn_qkv` name (triggers HunyuanVideoCheckpointConverter's
            // tencentRaw NormalizeTencentRaw path: "double_blocks.0.img_attn_qkv.weight" is its own detector key).
            ["double_blocks.0.img_attn_qkv.weight"] = T(3 * h, h), ["double_blocks.0.img_attn_qkv.bias"] = T(3 * h),
            // GameCraft's camera_in.* — kept prefixed by the router, never seen by HunyuanVideoCheckpointConverter.
            ["camera_in.encode_first.0.weight"] = T(192, 384), ["camera_in.scale"] = T(1),
        };

        HunyuanGameCraftCheckpointConverter.ConvertedWeights routed = HunyuanGameCraftCheckpointConverter.Convert(raw);

        Assert.NotEmpty(routed.Dit);
        Assert.NotEmpty(routed.CameraNet);
        Assert.Empty(routed.Vae);
        Assert.Empty(routed.Llava);
        Assert.Empty(routed.Clip);
        Assert.All(routed.CameraNet.Keys, k => Assert.StartsWith("camera_in.", k, StringComparison.Ordinal));
        Assert.DoesNotContain(routed.Dit.Keys, k => k.StartsWith("camera_in.", StringComparison.Ordinal));

        Dictionary<string, Tensor> hybrid = HunyuanVideoCheckpointConverter.Convert(routed.Dit);

        // These are the exact keys HunyuanVideoDit.LoadWeights/GameCraftCameraNet.LoadWeights index (see
        // HunyuanVideoDit.cs LoadWeights and the double-block attn.to_q lookup inside HunyuanImageBlock) — proving
        // the chain (GameCraft router → HunyuanVideo hybrid remap) lands on the names the model classes actually read.
        Assert.True(hybrid.ContainsKey("img_in.weight"), "img_in.proj.* must reshape to img_in.weight.");
        Assert.True(hybrid.ContainsKey("final_layer.mod.weight"), "final_layer.adaLN_modulation.1.* must rename to final_layer.mod.weight.");
        Assert.True(hybrid.ContainsKey("final_layer.proj.weight"), "final_layer.linear.* must rename to final_layer.proj.weight.");
        Assert.True(hybrid.ContainsKey("double_blocks.0.attn.to_q.weight"), "fused img_attn_qkv must split into attn.to_q/to_k/to_v.");
        Assert.True(hybrid.ContainsKey("double_blocks.0.attn.to_k.weight"));
        Assert.True(hybrid.ContainsKey("double_blocks.0.attn.to_v.weight"));
    }

    [Fact]
    public void CameraNet_LoadsWeights_RoutedThroughTheRealConverter()
    {
        // Same shapes GameCraftPartsTests hand-builds, but routed through Convert() itself (not hand-prefixed) —
        // proves the coupling between the router's output and GameCraftCameraNet.LoadWeights' default prefix.
        int hidden = 8;
        Dictionary<string, Tensor> raw = new()
        {
            ["camera_in.encode_first.0.weight"] = T(192, 384, 1, 1), ["camera_in.encode_first.0.bias"] = T(192),
            ["camera_in.encode_first.1.weight"] = Ones(192), ["camera_in.encode_first.1.bias"] = T(192),
            ["camera_in.encode_second.0.weight"] = T(96, 192, 1, 1), ["camera_in.encode_second.0.bias"] = T(96),
            ["camera_in.encode_second.1.weight"] = Ones(96), ["camera_in.encode_second.1.bias"] = T(96),
            ["camera_in.final_proj.weight"] = T(16, 96, 1, 1), ["camera_in.final_proj.bias"] = T(16),
            ["camera_in.camera_in.proj.weight"] = T(hidden, 16 * 2 * 2), ["camera_in.camera_in.proj.bias"] = T(hidden),
            ["camera_in.scale"] = Ones(1),
            // Noise: a DiT-looking key that must NOT end up routed into CameraNet.
            ["img_in.proj.weight"] = T(hidden, 4),
        };

        HunyuanGameCraftCheckpointConverter.ConvertedWeights routed = HunyuanGameCraftCheckpointConverter.Convert(raw);
        GameCraftCameraNet net = new(hiddenSize: hidden, downscale: 8, outChannels: 16, patchH: 2, patchW: 2);
        net.LoadWeights(routed.CameraNet); // default prefix "camera_in" — throws KeyNotFoundException if the router mis-routed.

        Assert.NotEmpty(net.EnumerateWeights().ToList());
    }

    [Fact]
    public void LoadFromPath_NonexistentDitPath_ThrowsFileNotFoundException_NotACrash()
    {
        // Exercises the assembler directly, bypassing WorldService.LoadHunyuanGameCraft's "no VAE encoder yet"
        // gate below — proves LoadFromPath itself is real and fails cleanly on missing files, which is exactly
        // what would run once a VAE encoder lands and that gate is lifted.
        using IBackend cpu = new CpuBackend();
        FileNotFoundException ex = Assert.Throws<FileNotFoundException>(() =>
            HunyuanGameCraftPipeline.LoadFromPath(cpu, "/nonexistent/mp_rank_00_model_states.pt", "/nonexistent/vae.safetensors"));
        Assert.Contains("nonexistent", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void WorldService_MissingAuxKeys_ThrowsArgumentException_BeforeTouchingDisk()
    {
        using InferenceEngine engine = new InferenceEngine("cpu");
        CatalogEntry catalog = ModelCatalog.Find("hunyuan-gamecraft") ?? throw new InvalidOperationException("hunyuan-gamecraft must be catalogued.");
        ModelSpec spec = new ModelSpec { Requested = "hunyuan-gamecraft", Modality = Modality.World, Catalog = catalog, LocalPath = "/nonexistent/mp_rank_00_model_states.pt" };
        WorldRequest request = new WorldRequest { InitImage = new ImageData { Rgb = new byte[3], Width = 1, Height = 1 } };

        ArgumentException ex = Assert.Throws<ArgumentException>(() => engine.World.Open(spec, request));
        Assert.Contains(WorldService.VaeAuxKey, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WorldService_ValidAuxKeys_ThrowsNotSupportedException_BeforeTouchingDisk()
    {
        // Every aux key is present (unlike WorldService_MissingAuxKeys_...) — this proves the NEW, specific
        // "no VAE encoder" gate fires (HunyuanGameCraftPipeline.LoadFromPathBuildsVaeEncoder), not the old blanket
        // "catalogued but not loadable" message the hunyuan-gamecraft case used to throw. It must fire BEFORE any
        // file I/O: the checkpoint paths below don't exist, so a FileNotFoundException would mean the loader ran
        // (and, on a real checkpoint, would have loaded ~51GB of weights for a session that could never succeed).
        using InferenceEngine engine = new InferenceEngine("cpu");
        CatalogEntry catalog = ModelCatalog.Find("hunyuan-gamecraft") ?? throw new InvalidOperationException("hunyuan-gamecraft must be catalogued.");
        ModelSpec spec = new ModelSpec
        {
            Requested = "hunyuan-gamecraft",
            Modality = Modality.World,
            Catalog = catalog,
            LocalPath = "/nonexistent/mp_rank_00_model_states.pt",
            Aux = new Dictionary<string, string>
            {
                [WorldService.VaeAuxKey] = "/nonexistent/vae.safetensors",
                [WorldService.LlavaAuxKey] = "/nonexistent/llava.safetensors",
                [WorldService.ClipAuxKey] = "/nonexistent/clip.safetensors",
            },
        };
        WorldRequest request = new WorldRequest { InitImage = new ImageData { Rgb = new byte[3], Width = 1, Height = 1 } };

        NotSupportedException ex = Assert.Throws<NotSupportedException>(() => engine.World.Open(spec, request));
        Assert.Contains("VAE encoder", ex.Message, StringComparison.Ordinal);
    }


    private static Tensor T(params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = 0.01f;
        return t;
    }

    private static Tensor Ones(long n)
    {
        Tensor t = new(new TensorShape(n), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < n; i++) p[i] = 1f;
        return t;
    }
}
