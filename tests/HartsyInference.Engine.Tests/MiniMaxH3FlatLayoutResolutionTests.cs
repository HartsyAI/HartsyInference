using HartsyInference.Engine.Recipes.Image;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;
using Xunit;

namespace HartsyInference.Engine.Tests;

/// <summary>Component lookup for the flat (Comfy-Org repack) layout that the video and image recipes share. The video
/// resolver must keep choosing its own VAE when the image VAE sits beside it, and the image resolver must honour the
/// request's component overrides.</summary>
public sealed class MiniMaxH3FlatLayoutResolutionTests
{
    [Fact]
    public void VideoResolverPicksItsOwnVaeAndTheSharedTextEncoderNotTheImageVae()
    {
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "minimax_h3_fl2va_pruned_bf16.safetensors"));
            WriteFile(Path.Combine(root, "vae", "minimax_h3_video_vae_fp16.safetensors"));
            WriteFile(Path.Combine(root, "vae", "minimax_h3_image_vae_fp16.safetensors"));
            WriteFile(Path.Combine(root, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"));

            MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(dit);

            Assert.False(assets.IsFolderLayout);
            Assert.EndsWith("minimax_h3_video_vae_fp16.safetensors", assets.VideoVae);
            Assert.EndsWith("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", assets.TextEncoder);
        });
    }

    [Fact]
    public void ImageResolverPrefersTheNvfp4EncoderWhenBothQuantizedBuildsArePresent()
    {
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "minimax_h3_image_pruned_int8_convrot.safetensors"));
            WriteFile(Path.Combine(root, "vae", "minimax_h3_image_vae_fp16.safetensors"));
            WriteFile(Path.Combine(root, "text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"));
            WriteFile(Path.Combine(root, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"));

            MiniMaxH3ImageAssets assets = MiniMaxH3ImageAssets.Resolve(dit, components: null);

            Assert.EndsWith("minimax_h3_image_vae_fp16.safetensors", assets.ImageVae);
            Assert.EndsWith("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", assets.TextEncoder);
        });
    }

    [Fact]
    public void ImageResolverHonoursTheRequestsVaeAndQwenOverrides()
    {
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "minimax_h3_image_pruned_int8_convrot.safetensors"));
            string customVae = WriteFile(Path.Combine(root, "elsewhere", "my_vae.safetensors"));
            string customQwen = WriteFile(Path.Combine(root, "elsewhere", "my_qwen.safetensors"));

            MiniMaxH3ImageAssets assets = MiniMaxH3ImageAssets.Resolve(dit,
                new ComponentOverrides { Vae = customVae, Qwen = customQwen });

            Assert.Equal(customVae, assets.ImageVae);
            Assert.Equal(customQwen, assets.TextEncoder);
        });
    }

    [Fact]
    public void ImageResolverFindsTheVaeAndEncoderInTheModelsRootLayout()
    {
        WithTempRoot(root =>
        {
            string dit = WriteFile(Path.Combine(root, "diffusion_models", "MiniMaxH3Image",
                "minimax_h3_image_pruned_int8_convrot.safetensors"));
            string vae = WriteFile(Path.Combine(root, "VAE", "MiniMaxH3", "minimax_h3_image_vae_fp16.safetensors"));
            string encoder = WriteFile(Path.Combine(root, "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"));

            MiniMaxH3ImageAssets assets = MiniMaxH3ImageAssets.Resolve(dit, components: null);

            Assert.Equal(vae, assets.ImageVae);
            Assert.Equal(encoder, assets.TextEncoder);
        });
    }

    private static void WithTempRoot(Action<string> test)
    {
        string root = Path.Combine(Path.GetTempPath(), "h3-flat-layout-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            test(root);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string WriteFile(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, Array.Empty<byte>());
        return path;
    }
}
