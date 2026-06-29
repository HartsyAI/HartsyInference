using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Interactive.ActionEncoders;
using HartsyInference.Interactive.Models;
using HartsyInference.Interactive.Pipelines;
using HartsyInference.Tests.Common;
using Xunit;

namespace HartsyInference.Interactive.Tests;

/// <summary>CPU end-to-end structural test for the Hunyuan-GameCraft denoise integration: builds the composite
/// (noisy + history + mask), camera tokens (action → Plücker → CameraNet), and runs the flow-match denoise loop
/// with CFG through the reused HunyuanVideo DiT — asserting a finite latent of the right shape. VAE decode is the
/// reused, separately-tested HunyuanVideo VAE and is exercised in the env-gated real-checkpoint test.</summary>
public sealed unsafe class HunyuanGameCraftPipelineTests
{
    [Fact]
    public void DenoiseChunk_ProducesFiniteLatent()
    {
        using IBackend cpu = new CpuBackend();
        HunyuanVideoConfig cfg = HunyuanVideoSyntheticWeights.TinyConfig; // hidden 32, InCh 33, OutCh 16, patch(1,2,2)

        HunyuanVideoDit dit = new(cfg);
        dit.LoadWeights(HunyuanVideoSyntheticWeights.BuildDit(cfg));

        GameCraftCameraNet cameraNet = new(hiddenSize: cfg.HiddenSize, downscale: 8, outChannels: 16, patchH: 2, patchW: 2, temporalCompression: 4);
        cameraNet.LoadWeights(BuildCameraWeights(cfg.HiddenSize));

        // Hlat=Wlat=2 → full image 16×16; tLat=2 → T_pix=5 → camera tokens align to the DiT image grid (S_img=2).
        GameCraftActionEncoder actionEnc = new(fx: 8, fy: 8, cx: 8, cy: 8, height: 16, width: 16, framesPerChunk: 25);
        HunyuanVideoVaeDecoder vaeDec = new(); // unloaded; DenoiseChunk does not use it

        using HunyuanGameCraftPipeline pipeline = new(cpu, dit, vaeDec, cameraNet, actionEnc, cfg,
            vaeEnc: null, spatialCompression: 8, temporalCompression: 4, flowShift: 5f);

        int tLat = 2, hLat = 2, wLat = 2;
        using Tensor history = Filled(0.0f, 1, cfg.OutChannels, tLat, hLat, wLat);
        using Tensor prompt = Filled(0.03f, 1, 3, cfg.TextEmbedDim);
        using Tensor pooled = Filled(0.02f, 1, cfg.PooledEmbedDim);
        using Tensor negPrompt = Filled(0.0f, 1, 3, cfg.TextEmbedDim);
        using Tensor negPooled = Filled(0.0f, 1, cfg.PooledEmbedDim);
        float[] mask = [1f, 0f];

        byte[] payload = new byte[GameCraftActionEncoder.PayloadBytes];
        GameCraftActionEncoder.PackPayload(w: true, a: false, s: false, d: false, speed: 1f, yawDelta: 0f, pitchDelta: 0f, payload);

        using Tensor latent = pipeline.DenoiseChunk(prompt, pooled, negPrompt, negPooled, history, mask, payload, steps: 2, guidance: 2f, seed: 7);
        Assert.Equal(5, latent.Shape.Rank);
        Assert.Equal(cfg.OutChannels, (int)latent.Shape[1]);
        Assert.Equal(tLat, (int)latent.Shape[2]);
        float* p = (float*)latent.DataPointer;
        for (long i = 0; i < latent.ElementCount; i++) Assert.True(float.IsFinite(p[i]));
    }

    private static Dictionary<string, Tensor> BuildCameraWeights(int hidden)
    {
        Random r = new(9);
        return new()
        {
            ["camera_in.encode_first.0.weight"] = T(r, 192, 384, 1, 1), ["camera_in.encode_first.0.bias"] = T(r, 192),
            ["camera_in.encode_first.1.weight"] = Ones(192), ["camera_in.encode_first.1.bias"] = Zeros(192),
            ["camera_in.encode_second.0.weight"] = T(r, 96, 192, 1, 1), ["camera_in.encode_second.0.bias"] = T(r, 96),
            ["camera_in.encode_second.1.weight"] = Ones(96), ["camera_in.encode_second.1.bias"] = Zeros(96),
            ["camera_in.final_proj.weight"] = T(r, 16, 96, 1, 1), ["camera_in.final_proj.bias"] = T(r, 16),
            ["camera_in.camera_in.proj.weight"] = T(r, hidden, 16 * 2 * 2), ["camera_in.camera_in.proj.bias"] = T(r, hidden),
            ["camera_in.scale"] = Ones(1),
        };
    }

    private static Tensor T(Random r, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = (float)(r.NextDouble() * 0.2 - 0.1);
        return t;
    }

    private static Tensor Ones(long n) { Tensor t = new(new TensorShape(n), DType.F32); float* p = (float*)t.DataPointer; for (long i = 0; i < n; i++) p[i] = 1f; return t; }
    private static Tensor Zeros(long n) => new(new TensorShape(n), DType.F32);

    private static Tensor Filled(float v, params long[] dims)
    {
        Tensor t = new(new TensorShape(dims), DType.F32);
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++) p[i] = v;
        return t;
    }
}
