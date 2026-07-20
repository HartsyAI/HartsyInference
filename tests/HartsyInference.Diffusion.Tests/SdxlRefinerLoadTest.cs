using System;
using System.IO;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.ModelAssets.CheckpointConverters;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Loads the real SDXL refiner checkpoint end-to-end through the converter + UNet weight
/// mapping, catching the "missing key" class of converter bugs (e.g. the doubled-conv upsampler
/// key that made the refiner un-loadable). CPU-only — weight mapping doesn't touch the GPU.</summary>
public class SdxlRefinerLoadTest
{
    private const string RefinerPath =
        "/home/hartsy/Desktop/Swarm/SwarmUI.not too old/Models/Stable-Diffusion/SDXL/sd_xl_refiner_1.0.safetensors";

    [Fact]
    public void SdxlRefiner_ConvertsAndLoadsUNet()
    {
        if (!File.Exists(RefinerPath))
        {
            return; // skip when the checkpoint isn't present
        }

        (SdxlRefinerCheckpointConverter.ConvertedWeights converted, HartsyInference.ModelAssets.SafeTensors.SafeTensorsLoader loader) =
            SdxlRefinerCheckpointConverter.LoadAndConvert(RefinerPath);
        try
        {
            Assert.True(converted.UNet.Count > 0, "no UNet keys converted");
            Assert.True(converted.ClipG.Count > 0, "no CLIP-G keys converted");
            Assert.True(converted.Vae.Count > 0, "no VAE keys converted");
            // The bug: this exact key was doubled to ...conv.conv.weight and thus absent.
            Assert.True(converted.UNet.ContainsKey("up_blocks.0.upsamplers.0.conv.weight"),
                "up_blocks.0.upsamplers.0.conv.weight missing — deepest-level upsampler not mapped");
            Assert.DoesNotContain(converted.UNet.Keys, k => k.Contains("conv.conv."));

            // The real test: build the refiner UNet and load — throws KeyNotFoundException on any gap.
            UNet refinerUnet = new UNet(UNetConfig.SdxlRefiner);
            refinerUnet.LoadWeights(converted.UNet);
        }
        finally
        {
            loader.Dispose();
        }
    }
}
