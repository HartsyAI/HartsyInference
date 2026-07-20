namespace HartsyInference.Engine.Audio;

/// <summary>The on-disk layout for user-placed audio checkpoints: <c>{models}/audio/{category}/{prefix}</c>, the
/// Engine-native replacement for the extension's <c>AudioConfiguration.ModelRoot</c> + provider model prefix.</summary>
internal static class AudioModelRoot
{
    /// <summary>The audio models root, <c>{models}/audio</c>. Shared assets (cmudict, contentvec) live here.</summary>
    internal static string Root() => Path.Combine(RepoPaths.ModelsRoot(), "audio");

    /// <summary>The directory a category/prefix pair's weights live in.</summary>
    internal static string WeightsDirectory(string category, string prefix) => Path.Combine(Root(), category, prefix);

    /// <summary>A shared file directly under the audio root (e.g. <c>cmudict.dict</c>).</summary>
    internal static string SharedFile(string fileName) => Path.Combine(Root(), fileName);
}
