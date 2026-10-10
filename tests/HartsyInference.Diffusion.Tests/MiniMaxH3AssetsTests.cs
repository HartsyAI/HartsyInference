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

    /// <summary>The int8 video VAE remains behind a real-weight parity gate. Merely staging it next to the proven
    /// FP16 file must not make every ordinary dense H3 request select a component that planning then rejects.</summary>
    [Fact]
    public void FlatLayout_DoesNotAutoSelectValidationPendingInt8VideoVae()
    {
        string dit = BuildFlatLayout();
        string dir = Path.Combine(_root, "Models", "vae", "MiniMaxH3");
        File.WriteAllBytes(Path.Combine(dir, "minimax_h3_video_vae_int8_convrot.safetensors"), []);
        File.WriteAllBytes(Path.Combine(dir, "minimax_h3_video_vae_fp16.safetensors"), new byte[64]);

        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(dit);

        Assert.Contains("fp16", assets.VideoVae);
        Assert.DoesNotContain("int8", assets.VideoVae);
    }

    /// <summary>The same gate, in the container the quantized builds actually ship in. A GGUF names its quant rather
    /// than a ComfyUI marker, so reading only <c>fp8</c>/<c>int8</c>/<c>nvfp4</c> put <c>-Q4_K.gguf</c> in the dense
    /// class beside the proven FP16 file — and the size tie-break then chose it.</summary>
    [Fact]
    public void FlatLayout_DoesNotAutoSelectAQuantizedGgufVideoVae()
    {
        string dit = BuildFlatLayout();
        string dir = Path.Combine(_root, "Models", "vae", "MiniMaxH3");
        File.WriteAllBytes(Path.Combine(dir, "minimax_h3_video_vae-Q4_K.gguf"), []);
        File.WriteAllBytes(Path.Combine(dir, "minimax_h3_video_vae_fp16.safetensors"), new byte[64]);

        MiniMaxH3Assets assets = MiniMaxH3Assets.Resolve(dit);

        Assert.Contains("fp16", assets.VideoVae);
        Assert.DoesNotContain("Q4_K", assets.VideoVae);
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
}
