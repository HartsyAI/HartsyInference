using HartsyInference.Audio.Frontends;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>The dictionaries Kokoro's Mandarin front-end reads, fetched once from their upstream repositories at the
/// versions misaki pins (jieba 0.42.1 and pypinyin 0.55.0, both MIT) into <c>{audio}/kokoro_zh/</c>, each checked
/// against its SHA-256. That folder is also what <c>KOKORO_ZH_DATA_DIR</c> points the parity test at.</summary>
internal static class KokoroMandarinAssets
{
    private const string JiebaUrl = "https://raw.githubusercontent.com/fxsjy/jieba/v0.42.1/jieba/";
    private const string PypinyinUrl = "https://raw.githubusercontent.com/mozillazg/python-pinyin/v0.55.0/pypinyin/";

    /// <summary>The folder under the audio root the dictionaries land in.</summary>
    internal const string Folder = "kokoro_zh";

    // Local name, source URL and SHA-256 of each file.
    private static readonly (string Name, string Url, string Sha256)[] Files =
    [
        ("dict.txt", JiebaUrl + "dict.txt", "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"),
        ("prob_emit.py", JiebaUrl + "finalseg/prob_emit.py",
            "27d46b1c9efe4dd148fde8be042a21be40e3562d0c7f1273f9de7abae12ebb8d"),
        ("phrases_dict.json", PypinyinUrl + "phrases_dict.json",
            "a45ff140a6b631ca9c82127b280a2f414e0aba6bb2824a0e9d1e77fff359c665"),
        ("pinyin_dict.json", PypinyinUrl + "pinyin_dict.json",
            "5f294c01e6c6c0a1c8e329c79335a3f8e0b27d06bf1de7a99244b765892d1e5b"),
    ];

    /// <summary>Fetches any missing dictionary and builds the front-end from them.</summary>
    public static async Task<KokoroMandarinG2P> LoadAsync(CancellationToken cancel)
    {
        string[] paths = new string[Files.Length];
        for (int i = 0; i < Files.Length; i++)
        {
            paths[i] = AudioModelRoot.SharedFile(Path.Combine(Folder, Files[i].Name));
            if (File.Exists(paths[i])) continue;
            Logs.Info($"[Audio][Kokoro] Downloading the Mandarin dictionary {Files[i].Name}...");
            await AudioFileFetcher.EnsureAsync(Files[i].Url, paths[i], Files[i].Sha256, cancel).ConfigureAwait(false);
        }
        return await Task.Run(() => KokoroMandarinG2P.FromFiles(paths[0], paths[1], paths[2], paths[3]), cancel)
            .ConfigureAwait(false);
    }
}
