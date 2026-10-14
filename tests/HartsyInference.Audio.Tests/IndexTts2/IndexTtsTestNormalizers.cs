using HartsyInference.Audio.Frontends.WeText;
using HartsyInference.Audio.Pipelines;

namespace HartsyInference.Audio.Tests.IndexTts2;

/// <summary>The text normalizers real-weight IndexTTS tests load with: wetext 0.1.8 over the grammars in
/// <c>WETEXT_FST_DIR</c> (else the reference venv's wetext wheel), as the engine does, or the interim normalizer
/// the engine also falls back to when no grammars are present.</summary>
internal static class IndexTtsTestNormalizers
{
    public static string FstDirectory =>
        Environment.GetEnvironmentVariable("WETEXT_FST_DIR") is { Length: > 0 } dir
            ? dir
            : "/mnt/model-storage/scratch/tts-ref/fe-venv/lib/python3.12/site-packages/wetext/fsts";

    public static IndexTtsFstNormalizers Resolve() =>
        File.Exists(Path.Combine(FstDirectory, "en", "tn", "tagger.fst")) && File.Exists(Path.Combine(FstDirectory, "zh", "tn", "tagger.fst"))
            ? IndexTtsFstNormalizers.FromWeText(WeTextNormalizer.Load(FstDirectory, WeTextLanguage.Auto, removeErhua: false, WeTextVersion.V0_1_8))
            : IndexTtsFstNormalizers.Interim;
}
