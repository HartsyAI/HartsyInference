namespace HartsyInference.ModelAssets.Metadata;

/// <summary>The publishing identity of every family the engine can repack, keyed by engine model id.
///
/// <para>This information used to live in three places and none of them was the source of truth: the backend
/// extension's <c>ModelSupport</c> table, the classes that extension registers itself, and a JSON file beside a
/// Python tool. A conversion site could reach none of them, which is why every file the engine has ever written
/// carries no identity at all.</para>
///
/// <para><b>Stamping is data, not registration.</b> Writing an id into a header is not
/// <c>T2IModelClassSorter.Register</c>, so the engine holding this table cannot collide with core or an extension
/// over class ownership. The only failure mode is drift: an id renamed upstream and not here. A published
/// artifact is checked against a live scan before upload, which is where drift shows up.</para>
///
/// <para><b>Licenses are a declaration, not a lookup.</b> The value here is what a bundle will claim; confirm it
/// against the upstream model card before publishing, and use <c>"other"</c> — with the terms spelled out in the
/// bundle README — for anything that is not a real HuggingFace license id.</para></summary>
public static class ModelIdentityCatalog
{
    private static readonly Dictionary<string, ArtifactIdentity> _byEngineId = Build();

    /// <summary>Every known identity, keyed by engine model id.</summary>
    public static IReadOnlyDictionary<string, ArtifactIdentity> All => _byEngineId;

    /// <summary>Looks up an identity by engine model id, case-insensitively. Null when the family has none yet —
    /// a caller that needs to stamp must ask the user rather than guess a class id.</summary>
    public static ArtifactIdentity? Find(string engineId) =>
        string.IsNullOrWhiteSpace(engineId) ? null : _byEngineId.GetValueOrDefault(engineId);

    private static ArtifactIdentity Image(string engineId, string classId, string name, string author,
        string license, string resolution, string? repo = null) =>
        new()
        {
            EngineId = engineId, SwarmClassId = classId, DisplayName = name, Author = author, License = license,
            StandardResolution = resolution, UpstreamRepo = repo, Tags = ["image", engineId],
        };

    private static ArtifactIdentity Video(string engineId, string classId, string name, string author,
        string license, string resolution, string? repo = null) =>
        new()
        {
            EngineId = engineId, SwarmClassId = classId, DisplayName = name, Author = author, License = license,
            StandardResolution = resolution, UpstreamRepo = repo, Tags = ["video", engineId],
        };

    // No resolution, ever: an audio class is registered 0x0, so any stamped value is a mismatch and SwarmUI
    // substitutes a clone carrying a standard size nobody declared.
    private static ArtifactIdentity Audio(string engineId, string classId, string name, string author,
        string license, string category, string? repo = null) =>
        new()
        {
            EngineId = engineId, SwarmClassId = classId, DisplayName = name, Author = author, License = license,
            UpstreamRepo = repo, Tags = [category, engineId], ProviderId = classId,
        };

    private static Dictionary<string, ArtifactIdentity> Build()
    {
        ArtifactIdentity[] rows =
        [
            // ── Image ───────────────────────────────────────────────────────────────────────────────────
            Image("sd15", "stable-diffusion-v1", "Stable Diffusion 1.5", "Stability AI", "creativeml-openrail-m", "512x512"),
            Image("sdxl", "stable-diffusion-xl-v1-base", "Stable Diffusion XL 1.0", "Stability AI", "openrail++", "1024x1024"),
            Image("sd3", "stable-diffusion-v3-medium", "Stable Diffusion 3 Medium", "Stability AI", "other", "1024x1024"),
            Image("flux1", "Flux.1-dev", "Flux.1 Dev", "Black Forest Labs", "other", "1024x1024", "black-forest-labs/FLUX.1-dev"),
            Image("flux2", "flux.2-dev", "Flux.2 Dev", "Black Forest Labs", "other", "1024x1024"),
            Image("chroma", "chroma", "Chroma", "lodestones", "apache-2.0", "1024x1024"),
            Image("chroma-radiance", "chroma-radiance", "Chroma Radiance", "lodestones", "apache-2.0", "1024x1024"),
            Image("qwen-image", "qwen-image", "Qwen-Image", "Alibaba Qwen", "apache-2.0", "1328x1328"),
            // 1024, not Qwen-Image v1's 1328 — the 2.1 class declares its own standard and a stamped 1328 would
            // match the id but disagree on size, which is exactly the case that disables the matcher.
            Image("qwen-image-2.1", "qwen-image-2.1", "Qwen-Image 2.1", "Alibaba Qwen", "apache-2.0", "1024x1024"),
            Image("hunyuan-image", "hunyuan-image-2_1", "HunyuanImage 2.1", "Tencent", "other", "2048x2048"),
            Image("hidream", "hidream-i1", "HiDream i1", "HiDream AI", "mit", "1024x1024"),
            Image("auraflow", "auraflow-v1", "AuraFlow", "Fal AI", "apache-2.0", "1024x1024"),
            Image("lumina2", "lumina-2", "Lumina 2.0", "Alpha VLLM", "apache-2.0", "1024x1024"),
            Image("ernie-image", "ernie-image", "ERNIE-Image", "Baidu", "apache-2.0", "1024x1024"),
            Image("omnigen2", "omnigen-2", "OmniGen 2", "VectorSpaceLab", "apache-2.0", "1024x1024"),
            Image("ideogram4", "ideogram-4", "Ideogram 4", "Ideogram", "other", "1024x1024"),
            Image("zimage", "z-image", "Z-Image Turbo", "Tongyi Lab", "apache-2.0", "1024x1024"),
            Image("anima", "anima", "Anima", "Anima", "other", "1024x1024"),
            Image("zeta-chroma", "zeta-chroma", "Zeta-Chroma", "lodestones", "apache-2.0", "1024x1024"),
            Image("boogu", "boogu", "Boogu Image", "Boogu", "other", "1024x1024"),
            Image("lens", "lens", "Lens", "Lens", "other", "1440x1440"),
            Image("krea2", "krea-2", "Krea 2", "Krea AI", "other", "1024x1024"),
            Image("mage-flow", "mage-flow", "Mage-Flow", "Mage", "other", "1024x1024"),
            Image("kandinsky5", "kandinsky5-image-lite", "Kandinsky 5 Image Lite", "Sber AI", "apache-2.0", "1024x1024"),
            Image("f-lite", "f-lite", "F-Lite", "Freepik", "apache-2.0", "1024x1024"),
            Image("lance-image", "lance-t2i", "Lance (Image)", "Lance", "other", "1024x1024"),

            // ── Video ───────────────────────────────────────────────────────────────────────────────────
            Video("ltx-video", "lightricks-ltx-video", "LTX-Video", "Lightricks", "other", "768x512"),
            Video("ltx-2", "lightricks-ltx-video-2-3", "LTX-2.3", "Lightricks", "other", "960x960"),
            Video("ltx-2.5", "lightricks-ltx-video-2-5", "LTX-2.5", "Lightricks", "other", "960x960"),
            Video("ltx-2.5-distilled", "lightricks-ltx-video-2-5", "LTX-2.5 Distilled", "Lightricks", "other", "960x960"),
            Video("wan", "wan-2_2-ti2v-5b", "Wan 2.2 TI2V 5B", "Alibaba Wan", "apache-2.0", "960x960"),
            Video("hunyuan-video", "hunyuan-video", "HunyuanVideo", "Tencent", "other", "720x720"),
            Video("minimax-h3", "minimax-h3", "MiniMax-H3", "MiniMax", "other", "960x960"),
            Video("kandinsky5-video", "kandinsky5-video-lite", "Kandinsky 5 Video Lite", "Sber AI", "apache-2.0", "640x640"),
            Video("lance-video", "lance-t2v", "Lance (Video)", "Lance", "other", "640x640"),

            // ── Music — these three classes are registered by SwarmUI core, not by AudioLab ──────────────
            Audio("acestep", "ace-step-1_5", "ACE-Step 1.5", "ACE Studio", "mit", "music", "ACE-Step/Ace-Step1.5") with
            {
                ProviderId = "acestep_music",
                VariantClassIds = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["turbo"] = "acestep_music_turbo", ["turbo-shift1"] = "acestep_music_turbo",
                    ["turbo-shift3"] = "acestep_music_turbo", ["turbo-continuous"] = "acestep_music_turbo",
                    ["xl-turbo"] = "acestep_music_turbo",
                },
            },
            Audio("minimaxmusic3", "minimax-music-3", "MiniMax Music 3", "MiniMax", "other", "music") with { ProviderId = "minimax_music3" },
            Audio("yue2", "yue-2", "YuE2", "m-a-p", "cc-by-nc-4.0", "music", "m-a-p/YuE2-3B") with { ProviderId = "yue2_music" },
            Audio("yue", "yue_music", "YuE", "m-a-p", "apache-2.0", "music"),
            Audio("heartmula", "heartlib_music", "HeartMuLa oss-3B", "HeartMuLa", "apache-2.0", "music"),
            Audio("musicgen", "musicgen_music", "MusicGen", "Meta", "cc-by-nc-4.0", "music"),
            Audio("audiogen", "audiogen_sfx", "AudioGen", "Meta", "cc-by-nc-4.0", "music"),
            Audio("stableaudio", "stableaudio_music", "Stable Audio Open Small", "Stability AI", "other", "music"),

            // ── Speech-to-text ──────────────────────────────────────────────────────────────────────────
            Audio("whisper", "whisper_stt", "Whisper", "OpenAI", "apache-2.0", "stt"),
            Audio("whisperstreaming", "whisperstreaming_stt", "Whisper Streaming", "OpenAI", "apache-2.0", "stt"),
            Audio("distilwhisper", "distilwhisper_stt", "Distil-Whisper", "Hugging Face", "mit", "stt"),
            Audio("moonshine", "moonshine_stt", "Moonshine", "Useful Sensors", "mit", "stt"),
            Audio("moonshinestreaming", "moonshinestreaming_stt", "Moonshine Streaming", "Useful Sensors", "mit", "stt"),
            Audio("kyutaistt", "kyutaistt_stt", "Kyutai STT", "Kyutai", "cc-by-4.0", "stt"),
            // Ships inside the YuE2 release, under its license.
            Audio("sheetsage2", "sheetsage2", "SheetSage2", "m-a-p", "cc-by-nc-4.0", "stt", "Comfy-Org/YuE2") with { ProviderId = "sheetsage2_transcribe" },

            // ── Text-to-speech ──────────────────────────────────────────────────────────────────────────
            Audio("kokoro", "kokoro_tts", "Kokoro-82M", "hexgrad", "apache-2.0", "tts", "hexgrad/Kokoro-82M"),
            Audio("bark", "bark_tts", "Bark", "Suno", "mit", "tts"),
            Audio("dia", "dia_tts", "Dia 1.6B", "Nari Labs", "apache-2.0", "tts"),
            Audio("orpheus", "orpheus_tts", "Orpheus", "Canopy Labs", "apache-2.0", "tts"),
            Audio("csm", "csm_tts", "CSM 1B", "Sesame", "apache-2.0", "tts"),
            Audio("vibevoice", "vibevoice_tts", "VibeVoice", "Microsoft", "mit", "tts"),
            Audio("cosyvoice", "cosyvoice_tts", "CosyVoice 2", "Alibaba FunAudioLLM", "apache-2.0", "tts"),
            Audio("chatterbox", "chatterbox_tts", "Chatterbox", "Resemble AI", "mit", "tts"),
            Audio("kyutaitts", "kyutaitts_tts", "Kyutai TTS", "Kyutai", "cc-by-4.0", "tts"),
            Audio("piper", "piper_tts", "Piper", "Rhasspy", "mit", "tts"),
            Audio("melotts", "melotts_tts", "MeloTTS", "MyShell", "mit", "tts"),
            Audio("pockettts", "pockettts_tts", "PocketTTS", "Kyutai", "cc-by-4.0", "tts"),
            Audio("styletts2", "styletts2_tts", "StyleTTS2", "yl4579", "mit", "tts"),
            Audio("zonos", "zonos_tts", "Zonos v0.1", "Zyphra", "apache-2.0", "tts"),
            Audio("gptsovits", "gptsovits_clone", "GPT-SoVITS v2", "RVC-Boss", "mit", "tts"),
            Audio("qwen3tts", "qwen3_tts", "Qwen3-TTS", "Alibaba Qwen", "apache-2.0", "tts") with
            {
                ProviderId = "qwen3_tts",
                VariantClassIds = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                {
                    ["0.6B-Base"] = "qwen3_tts_clone", ["1.7B-Base"] = "qwen3_tts_clone",
                    ["0.6B-CustomVoice"] = "qwen3_tts_custom", ["1.7B-CustomVoice"] = "qwen3_tts_custom",
                    ["1.7B-VoiceDesign"] = "qwen3_tts_design",
                },
            },
            Audio("zipvoice", "zipvoice_tts", "ZipVoice", "k2-fsa", "apache-2.0", "tts"),
            Audio("f5", "f5_tts", "F5-TTS", "SWivid", "cc-by-nc-4.0", "tts"),
            Audio("sparktts", "sparktts_tts", "Spark-TTS", "SparkAudio", "cc-by-nc-sa-4.0", "tts"),
            Audio("fishspeech", "fishspeech_tts", "Fish-Speech 1.5", "Fish Audio", "cc-by-nc-sa-4.0", "tts", "fishaudio/fish-speech-1.5"),
            Audio("neutts", "neutts_tts", "NeuTTS Air", "Neuphonic", "apache-2.0", "tts", "neuphonic/neutts-air"),

            // ── Voice conversion and effects ────────────────────────────────────────────────────────────
            Audio("rvc", "rvc_clone", "RVC v2", "RVC-Project", "mit", "clone"),
            Audio("openvoice", "openvoice_clone", "OpenVoice V2", "MyShell", "mit", "clone"),
            Audio("demucs", "demucs_fx", "Demucs", "Meta", "mit", "fx"),
            Audio("resemble-enhance", "resemble_enhance_fx", "Resemble-Enhance", "Resemble AI", "mit", "fx"),
        ];
        Dictionary<string, ArtifactIdentity> byId = new(rows.Length, StringComparer.OrdinalIgnoreCase);
        foreach (ArtifactIdentity row in rows)
        {
            if (!byId.TryAdd(row.EngineId, row))
            {
                throw new InvalidOperationException(
                    $"Duplicate identity for engine id '{row.EngineId}'. One row per family — a second would make "
                    + "which class an artifact is stamped with depend on declaration order.");
            }
        }
        return byId;
    }
}
