using HartsyInference.Engine.HuggingFace;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Uses the engine downloader; cache hits are verified, never trusted by filename or size alone.</summary>
public static class Assets
{
    public static string PathFor(string cache, AssetDefinition asset) => Path.Combine(cache, asset.Sha256, asset.File);
    public static void Verify(string cache, AssetDefinition asset)
    {
        string file = PathFor(cache, asset);
        if (!File.Exists(file) || new FileInfo(file).Length != asset.Bytes || Hashes.FileHash(file) != asset.Sha256)
            throw new InvalidDataException("Missing or corrupt pinned model: " + asset.File + ". Run fetch first.");
    }

    public static async Task FetchAsync(SuiteDefinition suite, string cache, CancellationToken cancel)
    {
        using HuggingFaceClient client = new();
        foreach (AssetDefinition asset in suite.Cases.Select(c => c.Asset).DistinctBy(a => a.Sha256))
        {
            string file = PathFor(cache, asset);
            if (!File.Exists(file))
                await client.DownloadRevisionAsync(asset.Repository, asset.File, asset.Revision, file, null, asset.Sha256, cancel);
            Verify(cache, asset);
        }
    }
}
