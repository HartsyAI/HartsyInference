namespace HartsyInference.Audio.Models.LongCat;

/// <summary>The audio-feature layout one LongCat-Video-Avatar version feeds its <c>AudioProjModel</c>: how many encoder
/// feature "blocks" of how many channels, at what feature rate, and how a video frame's window and a latent frame's group
/// are cut from them. Both presets share the 5-frame window and the 4x temporal VAE scale.</summary>
public sealed record LongCatAudioLayout
{
    /// <summary>Feature blocks per frame: 5 pooled Whisper blocks, or the 12 transformer-layer states of wav2vec2.</summary>
    public required int Blocks { get; init; }

    /// <summary>Channels per block: 1280 for Whisper-large-v3, 768 for chinese-wav2vec2-base.</summary>
    public required int Channels { get; init; }

    /// <summary>Video frames per second the clip is rendered at.</summary>
    public required int VideoFps { get; init; }

    /// <summary>Feature frames consumed per video frame: 1 for Whisper features resampled to the video rate, 2 for the
    /// 1.0 wav2vec2 features, which stay at twice the video rate.</summary>
    public required int AudioStride { get; init; }

    /// <summary>Feature frames in one video frame's window (<c>audio_window</c>), centred on the frame; odd.</summary>
    public int Window { get; init; } = 5;

    /// <summary>Video frames per latent frame after the first (<c>vae_scale</c>).</summary>
    public int VaeScale { get; init; } = 4;

    /// <summary>Feature frames per second the encoder output is resampled to (<c>VideoFps * AudioStride</c>).</summary>
    public int FeatureFps => VideoFps * AudioStride;

    /// <summary>Window slices one later latent frame contributes: the first video frame's left half and centre, the middle
    /// frames' centres, the last frame's centre and right half (8 for a 5-window and a 4x scale).</summary>
    public int GroupWindow => Window + VaeScale - 1;

    /// <summary>Values in one window, <c>Window * Blocks * Channels</c>: <c>proj1</c>'s input width.</summary>
    public int WindowValues => Window * Blocks * Channels;

    /// <summary>Values in one later group, <c>GroupWindow * Blocks * Channels</c>: <c>proj1_vf</c>'s input width.</summary>
    public int GroupValues => GroupWindow * Blocks * Channels;

    /// <summary>Avatar 1.5 and 1.6: Whisper-large-v3, 5 blocks of 1280, 25 fps. <c>proj1</c> reads 32000, <c>proj1_vf</c> 51200.</summary>
    public static LongCatAudioLayout Whisper => new() { Blocks = 5, Channels = 1280, VideoFps = 25, AudioStride = 1 };

    /// <summary>Avatar 1.0: chinese-wav2vec2-base, 12 blocks of 768, 16 fps with features at 32 Hz.
    /// <c>proj1</c> reads 46080, <c>proj1_vf</c> 73728.</summary>
    public static LongCatAudioLayout Wav2Vec2 => new() { Blocks = 12, Channels = 768, VideoFps = 16, AudioStride = 2 };
}
