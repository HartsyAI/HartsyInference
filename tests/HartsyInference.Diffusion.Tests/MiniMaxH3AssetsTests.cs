using Xunit;
using HartsyInference.Engine.Recipes.Video;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Diffusion.Tests;

/// <summary>MiniMax-H3 ships in two incompatible layouts — the vendor folder tree and Comfy-Org's flat repack, which
/// is what SwarmUI's own H3 support downloads. Resolution is path-only (no headers read), so these use empty files.</summary>
public sealed class MiniMaxH3AssetsTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "h3assets-" + Guid.NewGuid().ToString("N"));

    private string Touch(params string[] parts)
    {
        string path = Path.Combine([_root, .. parts]);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllBytes(path, []);
        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(_root)) { Directory.Delete(_root, recursive: true); }
    }

    private void BuildFolderLayout(string variant)
    {
        Touch(variant, "transformer", "dit.safetensors");
        Touch(variant, "video_vae", "model.safetensors");
        Touch(variant, "audio_vae", "model.safetensors");
        Touch(variant, "text_encoder", "model.safetensors");
    }

    [Fact]
    public void FolderLayout_ResolvesEveryComponent()
    {
        BuildFolderLayout("FL2VA");
        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(Path.Combine(_root, "FL2VA"));
        Assert.True(assets.IsFolderLayout);
        Assert.EndsWith(Path.Combine("transformer", "dit.safetensors"), assets.Dit);
        Assert.Contains("video_vae", assets.VideoVae);
        Assert.Contains("audio_vae", assets.AudioVae);
        Assert.Contains("text_encoder", assets.TextEncoder);
    }

    /// <summary>Pointing at the model folder rather than the variant folder must still find FL2VA/Ref2VA.</summary>
    [Fact]
    public void FolderLayout_DescendsIntoTheVariantFolder()
    {
        BuildFolderLayout("Ref2VA");
        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(_root);
        Assert.True(assets.IsFolderLayout);
        Assert.Contains("Ref2VA", assets.Dit);
    }

    [Fact]
    public void FolderLayout_AcceptsTheDitFileItself()
    {
        BuildFolderLayout("FL2VA");
        string dit = Path.Combine(_root, "FL2VA", "transformer", "dit.safetensors");
        Assert.True(MiniMaxH3Assets.Resolve(dit).IsFolderLayout);
    }

    private string BuildFlatLayout()
    {
        string dit = Touch("Models", "diffusion_models", "minimax_h3_fl2va_bf16.safetensors");
        Touch("Models", "vae", "MiniMaxH3", "minimax_h3_video_vae_fp16.safetensors");
        Touch("Models", "vae", "MiniMaxH3", "minimax_h3_audio_vae_fp32.safetensors");
        Touch("Models", "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors");
        return dit;
    }

    [Fact]
    public void FlatLayout_FindsComponentsBesideTheDit()
    {
        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(BuildFlatLayout());
        Assert.False(assets.IsFolderLayout);
        Assert.Contains("video_vae", assets.VideoVae);
        Assert.Contains("audio_vae", assets.AudioVae);
        Assert.Contains("qwen3vl", assets.TextEncoder);
        // The flat repack ships no tokenizer files; the recipe falls back to the embedded Qwen BPE.
        Assert.Null(assets.TokenizerDir);
    }

    /// <summary>Comfy stages several variants of a component together. The now-supported resident
    /// <c>int8_convrot</c> build should beat BF16, with file size breaking ties between supported quant formats.</summary>
    [Fact]
    public void FlatLayout_PrefersSupportedInt8OverBf16()
    {
        string dit = BuildFlatLayout();
        File.Delete(Path.Combine(_root, "Models", "text_encoders", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"));
        File.WriteAllBytes(Path.Combine(_root, "Models", "text_encoders", "qwen3vl_32b_minimax_h3_bf16.safetensors"),
            new byte[64]);
        File.WriteAllBytes(Path.Combine(_root, "Models", "text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
            new byte[8]);

        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(dit);
        Assert.Contains("int8_convrot", assets.TextEncoder);
        Assert.DoesNotContain("bf16", assets.TextEncoder);
    }

    /// <summary>With every variant staged, the quantized-but-supported build wins. This is the preference the recipe
    /// actually relies on — nvfp4 is the Qwen3-VL format verified against real weights, and at 16 GB vs 51 GB it is
    /// the only one that leaves room for the DiT.</summary>
    [Fact]
    public void FlatLayout_PrefersNvfp4OverBf16()
    {
        string dit = BuildFlatLayout();
        string dir = Path.Combine(_root, "Models", "text_encoders");
        // Larger on disk than the nvfp4 file, so a size-only rule would still pick nvfp4 — make bf16 the smaller one
        // to prove the ranking, not the size, decides.
        File.WriteAllBytes(Path.Combine(dir, "qwen3vl_32b_minimax_h3_bf16.safetensors"), []);
        File.WriteAllBytes(Path.Combine(dir, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"), new byte[64]);

        Assert.Contains("nvfp4", MiniMaxH3Assets.Resolve(dit).TextEncoder);
    }

    /// <summary>SwarmUI's model root has a real <c>Video/</c> folder for video assets. Searching it for VAEs would
    /// let an unrelated file with a colliding name resolve as the VAE.</summary>
    [Fact]
    public void FlatLayout_IgnoresSwarmsVideoAssetFolder()
    {
        string dit = BuildFlatLayout();
        File.Delete(Path.Combine(_root, "Models", "vae", "MiniMaxH3", "minimax_h3_video_vae_fp16.safetensors"));
        Touch("Models", "Video", "minimax_h3_video_vae_decoy.safetensors");
        Assert.Throws<FileNotFoundException>(() => MiniMaxH3Assets.Resolve(dit));
    }

    /// <summary>An audio VAE is optional — video still decodes, just without its soundtrack.</summary>
    [Fact]
    public void FlatLayout_ToleratesAMissingAudioVae()
    {
        string dit = BuildFlatLayout();
        File.Delete(Path.Combine(_root, "Models", "vae", "MiniMaxH3", "minimax_h3_audio_vae_fp32.safetensors"));
        Assert.Null(MiniMaxH3Assets.Resolve(dit).AudioVae);
    }

    [Fact]
    public void FlatLayout_ExplicitComponentOverridesReplaceMissingDefaults()
    {
        string dit = BuildFlatLayout();
        string defaultVideo = Path.Combine(_root, "Models", "vae", "MiniMaxH3",
            "minimax_h3_video_vae_fp16.safetensors");
        File.Delete(defaultVideo);
        string video = Touch("overrides", "video.safetensors");
        string audio = Touch("overrides", "audio.safetensors");
        string text = Touch("overrides", "qwen.safetensors");
        ComponentOverrides components = new ComponentOverrides
        {
            VideoVae = video,
            AudioVae = audio,
            Qwen = text,
        };

        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(dit, components);

        Assert.Equal(video, assets.VideoVae);
        Assert.Equal(audio, assets.AudioVae);
        Assert.Equal(text, assets.TextEncoder);
    }

    [Fact]
    public void FlatLayout_MissingVideoVae_ThrowsWithAnActionableMessage()
    {
        string dit = BuildFlatLayout();
        File.Delete(Path.Combine(_root, "Models", "vae", "MiniMaxH3", "minimax_h3_video_vae_fp16.safetensors"));
        FileNotFoundException ex = Assert.Throws<FileNotFoundException>(() => MiniMaxH3Assets.Resolve(dit));
        Assert.Contains("video VAE", ex.Message);
    }

    [Fact]
    public void MissingPath_Throws() =>
        Assert.Throws<FileNotFoundException>(() => MiniMaxH3Assets.Resolve(Path.Combine(_root, "nope")));
}
