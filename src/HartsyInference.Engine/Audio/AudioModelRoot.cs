using HartsyInference.Audio.Cache;
using HartsyInference.Core.IO;

namespace HartsyInference.Engine.Audio;

/// <summary>The on-disk layout for user-placed audio checkpoints: <c>{models}/audio/{category}/{prefix}</c>, the Engine-native replacement for the extension's <c>AudioConfiguration.ModelRoot</c> + provider model prefix. A folder spelled in another case (AudioLab's <c>music/YuE</c> for <c>music/yue</c>) is matched the same way <see cref="ModelDownloader.TargetPath"/> matches it, so a download and its loader agree on one folder.</summary>
internal static class AudioModelRoot
{
    /// <summary>The audio models root, <c>{models}/audio</c>. Shared assets (cmudict, contentvec) live here.</summary>
    internal static string Root()
    {
        string root = Location();
        AudioStandIns.EnsureSynced(root);
        return root;
    }

    /// <summary>The audio models root without first linking converted stand-ins into it.</summary>
    internal static string Location() => CaseInsensitivePath.ResolveDirectory(RepoPaths.ModelsRoot(), "audio");

    /// <summary>The directory a category/prefix pair's weights live in.</summary>
    internal static string WeightsDirectory(string category, string prefix) =>
        CaseInsensitivePath.ResolveDirectory(Root(), Path.Combine(category, prefix));

    /// <summary>A shared file directly under the audio root (e.g. <c>cmudict.dict</c>).</summary>
    internal static string SharedFile(string fileName) => Path.Combine(Root(), fileName);
}
