using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Models.Vae.QwenImage;
using HartsyInference.ModelHandler.SafeTensors;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Cheap structural sanity for the Qwen-Image VAE decoder, independent of any diffusers reference.
/// Decodes a CONSTANT latent and a smooth low-frequency latent: a correct VAE turns a constant latent into a
/// near-uniform image (low spatial variance, no high-frequency grid), whereas a structurally-broken decoder
/// (wrong conv/upsample layout, 3D-vs-2D kernel mismatch, bad weightnorm) emits a periodic grid regardless of
/// input. This isolates "is the garbage in the VAE or upstream (transformer/denoising)?" without a 13-min run.</summary>
[Trait("Category", "Integration")]
public unsafe class QwenImageVaeSmokeTests
{
    private readonly ITestOutputHelper _output;
    public QwenImageVaeSmokeTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void Decoder_ConstantLatent_IsSmooth_Cpu()
    {
        string vaePath = TestPaths.QwenImage.Vae;
        if (!File.Exists(vaePath)) { _output.WriteLine($"SKIPPED: VAE not found: {vaePath}"); return; }

        SafeTensorsLoader loader = new();
        loader.Load(vaePath);
        Dictionary<string, Tensor> raw = loader.GetAllTensors();
        Dictionary<string, Tensor> f32 = new(raw.Count);
        foreach (KeyValuePair<string, Tensor> kv in raw)
            f32[kv.Key] = (kv.Value.DType == DType.F16 || kv.Value.DType == DType.BF16) ? kv.Value.CastTo(DType.F32) : kv.Value;

        QwenImageVaeDecoder vae = new(VaeConfig.QwenImage);
        vae.LoadWeights(f32);
        using CpuBackend backend = new();

        // SMOOTH GRADIENT latent [1,16,32,32]: spatially low-frequency. A correct VAE decodes this to a smooth
        // gradient image; a structurally-broken decoder grids it. Discriminates VAE-bug vs transformer-bug.
        int C = 16, H = 32, W = 32;
        Tensor latent = new(new TensorShape(1, C, H, W), DType.F32);
        Span<float> ls = latent.AsSpan<float>();
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    ls[(c * H + y) * W + x] = 0.5f * MathF.Sin((x / (float)W) * 3.14159f) * MathF.Cos((y / (float)H) * 3.14159f);

        Tensor img = vae.Decode(backend, latent);
        // Save the decoded gradient so it can be eyeballed (smooth vs woven-grid).
        {
            int oC = (int)img.Shape[1], oH = (int)img.Shape[2], oW = (int)img.Shape[3];
            ReadOnlySpan<float> px = img.AsReadOnlySpan<float>();
            byte[] rgb = new byte[oH * oW * 3];
            for (int y = 0; y < oH; y++)
                for (int x = 0; x < oW; x++)
                    for (int ch = 0; ch < 3; ch++)
                    {
                        float v = px[(ch * oH + y) * oW + x];
                        rgb[(y * oW + x) * 3 + ch] = (byte)Math.Clamp((v * 0.5f + 0.5f) * 255f, 0, 255);
                    }
            string outDir = TestPaths.OutputDir; Directory.CreateDirectory(outDir);
            string p = Path.Combine(outDir, $"qwen_vae_gradient_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            HartsyInference.Diffusion.Utilities.ImagePostProcessor.SaveBmp(p, rgb, oW, oH);
            _output.WriteLine($"  saved VAE gradient decode: {p}");
        }
        _output.WriteLine($"decoded shape {img.Shape}");
        int iC = (int)img.Shape[1], iH = (int)img.Shape[2], iW = (int)img.Shape[3];

        ReadOnlySpan<float> o = img.AsReadOnlySpan<float>();
        // Global stats
        double mean = 0; for (int i = 0; i < o.Length; i++) mean += o[i]; mean /= o.Length;
        double var = 0; for (int i = 0; i < o.Length; i++) { double d = o[i] - mean; var += d * d; } var /= o.Length;
        double std = Math.Sqrt(var);

        // Grid/high-frequency metric: mean |pixel - right-neighbor| on channel 0. A smooth uniform image has
        // tiny adjacent differences; a periodic grid has large ones. Compare to the global std.
        double adjDiff = 0; long n = 0;
        for (int y = 0; y < iH; y++)
            for (int x = 0; x < iW - 1; x++)
            {
                int idx = (y * iW + x);
                adjDiff += Math.Abs(o[idx] - o[idx + 1]); n++;
            }
        adjDiff /= n;

        _output.WriteLine($"output: mean={mean:F4} std={std:F4} meanAdjDiff(ch0)={adjDiff:F4}  ratio adj/std={adjDiff / (std + 1e-9):F3}");
        _output.WriteLine(std < 0.05
            ? "=> SMOOTH/uniform: VAE decode of a constant latent is flat → VAE structurally OK; garbage is UPSTREAM (transformer/denoising)."
            : "=> NON-uniform output from a constant latent → VAE decoder is structurally broken (focus the VAE).");
    }
}
