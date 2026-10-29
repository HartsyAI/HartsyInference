using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The standalone SDXL VAE a checkpoint without its own VAE falls back to. The pin is the Hugging Face LFS hash of
/// stabilityai/sdxl-vae's sdxl_vae.safetensors, so a changed constant here is a changed file and must be read again.</summary>
public sealed class SdxlVaeSideModelTests
{
    [Fact]
    public void TheSdxlVaeIsThePinnedStabilityFile()
    {
        Assert.Equal("stabilityai/sdxl-vae", SideModels.SdxlVae.Repo);
        Assert.Equal("sdxl_vae.safetensors", SideModels.SdxlVae.RepoPath);
        Assert.Equal("vae", SideModels.SdxlVae.Role);
        Assert.Equal("63aeecb90ff7bc1c115395962d3e803571385b61938377bc7089b36e81e92e2e", SideModels.SdxlVae.Sha256);
        Assert.Matches("^[0-9a-f]{64}$", SideModels.SdxlVae.Sha256);
    }
}
