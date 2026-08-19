using HartsyInference.Cuda;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.ModelAssets.CheckpointConverters.Utils;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Validates the bespoke <see cref="HunyuanImageVaeEncoder"/> against the already-verified
/// <see cref="HunyuanImageVaeDecoder"/> by encode→decode roundtrip on the real 2.1 VAE weights. The encoder is a
/// hand-mirrored architecture (residual pixel-shuffle downsamplers, channel-group-average skips) — if any piece of
/// that mirror is wrong (skip structure, [r1, r2, c] channel order, conv-vs-shuffle ordering), the roundtrip
/// reconstruction collapses to noise, so a high pixel correlation is a strong architectural check even though a
/// 32× VAE is far from lossless.</summary>
public sealed class HunyuanImageVaeEncoderRealWeightTests
{
    private const string VaePath = "/home/hartsy/Desktop/HartsyInference/Models/VAE/hunyuan_image_2.1_vae_fp16.safetensors";

    [Fact]
    public void EncodeDecodeRoundtrip_ReconstructsTheImage()
    {
        if (!File.Exists(VaePath))
        {
            return; // VAE not downloaded on this machine — nothing to validate against.
        }
        string ptxDir = Path.Combine(AppContext.BaseDirectory, "Ptx");
        if (!Directory.Exists(ptxDir))
        {
            return;
        }

        using SafeTensorsLoader loader = new SafeTensorsLoader();
        loader.Load(VaePath);
        Dictionary<string, Tensor> weights = new Dictionary<string, Tensor>();
        foreach (KeyValuePair<string, Tensor> kv in loader.GetAllTensors())
        {
            string mapped = CheckpointConvertUtils.ConvertVaeKey(kv.Key, numUpLevels: 6, reverseUpIndices: false) ?? kv.Key;
            weights[mapped] = kv.Value;
        }

        using CudaBackend backend = new CudaBackend(deviceOrdinal: 0, ptxDir: ptxDir);
        HunyuanImageVaeEncoder encoder = new HunyuanImageVaeEncoder(VaeConfig.HunyuanImage);
        encoder.LoadWeights(weights);
        HunyuanImageVaeDecoder decoder = new HunyuanImageVaeDecoder(VaeConfig.HunyuanImage);
        decoder.LoadWeights(weights);

        // Structured test image in [-1, 1]: smooth gradients + a block, so both low- and mid-frequency content
        // must survive the roundtrip. 256×256 → latent 8×8×64.
        const int size = 256;
        Tensor image = new Tensor(new TensorShape(1, 3, size, size), DType.F32);
        unsafe
        {
            float* p = (float*)image.DataPointer;
            for (int y = 0; y < size; y++)
            {
                for (int x = 0; x < size; x++)
                {
                    int i = y * size + x;
                    p[i] = (2f * x / size) - 1f;                                     // R: horizontal ramp
                    p[size * size + i] = (2f * y / size) - 1f;                       // G: vertical ramp
                    p[2 * size * size + i] = (x is > 64 and < 192 && y is > 64 and < 192) ? 0.8f : -0.8f; // B: block
                }
            }
        }

        Tensor latent = encoder.Encode(backend, image);
        Assert.True(latent.Shape.Equals(new TensorShape(1, 64, size / 32, size / 32)), $"latent shape {latent.Shape}");

        Tensor decoded = decoder.Decode(backend, latent);
        latent.Dispose();
        Assert.True(decoded.Shape.Equals(new TensorShape(1, 3, size, size)), $"decoded shape {decoded.Shape}");

        // Pearson correlation between input and reconstruction. A wrong architecture lands near 0; the real
        // roundtrip of a working VAE sits well above 0.9 on structured content.
        double corr;
        unsafe
        {
            float* a = (float*)image.DataPointer;
            float* b = (float*)decoded.DataPointer;
            long n = image.Shape.ElementCount;
            double ma = 0, mb = 0;
            for (long i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
            ma /= n; mb /= n;
            double num = 0, da = 0, db = 0;
            for (long i = 0; i < n; i++)
            {
                double xa = a[i] - ma, xb = b[i] - mb;
                num += xa * xb; da += xa * xa; db += xb * xb;
            }
            corr = num / Math.Sqrt(Math.Max(da * db, 1e-12));
        }
        image.Dispose();
        decoded.Dispose();

        Assert.True(corr > 0.90, $"Encode→decode roundtrip correlation {corr:F4} — the encoder mirror is architecturally wrong if this is low.");
    }
}
