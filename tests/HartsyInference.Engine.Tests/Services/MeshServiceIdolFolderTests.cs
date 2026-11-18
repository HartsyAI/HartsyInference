using HartsyInference.Engine.Services;

namespace HartsyInference.Engine.Tests.Services;

public sealed class MeshServiceIdolFolderTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "idol-folder-" + Guid.NewGuid().ToString("N"));

    public MeshServiceIdolFolderTests() => Directory.CreateDirectory(_root);

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private void Touch(string relative)
    {
        string path = Path.Combine(_root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, [1]);
    }

    [Theory]
    [InlineData("idol/diffusion_models/idol_fp16.safetensors")]
    [InlineData("idol/diffusion_models/budget3d_idol_turbo.safetensors")]
    [InlineData("idol/diffusion_models")]
    [InlineData("idol")]
    public void IdolFolder_IsTheFolderHoldingDiffusionModels_WhateverPartOfTheInstallTheResolverReturned(string given)
    {
        Touch("idol/diffusion_models/idol_fp16.safetensors");
        Touch("idol/diffusion_models/budget3d_idol_turbo.safetensors");
        Touch("idol/clip_vision/dino_v3_vit_h_bf16.safetensors");

        Assert.Equal(Path.GetFullPath(Path.Combine(_root, "idol")), MeshService.IdolFolder(Path.Combine(_root, given)));
    }

    [Fact]
    public void IdolFolder_FindsNoInstall_ReturnsWhatItWasGiven()
    {
        Touch("elsewhere/model.safetensors");

        Assert.Equal(Path.GetFullPath(Path.Combine(_root, "elsewhere")), MeshService.IdolFolder(Path.Combine(_root, "elsewhere", "model.safetensors")));
    }
}
