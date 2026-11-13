using System.Net;
using System.Security.Cryptography;
using HartsyInference.Core.Configuration;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.Engine.Registry;
using Xunit;

namespace HartsyInference.Engine.Tests.Registry;

/// <summary>The mirror-first download of a pickle asset's repack, the URL fetch of a release asset and the presence rules around them, over a temp models root and a stand-in network: nothing is downloaded.</summary>
[Collection("ModelsRootKnob")]
public sealed class WeightsMirrorDownloadTests : IDisposable
{
    private const string Mirror = "https://mirror.example/weights";
    private static readonly byte[] Pickle = [1, 2, 3, 4, 5];
    private static readonly byte[] Repack = [9, 8, 7, 6];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-mirror-" + Guid.NewGuid().ToString("N"));

    public WeightsMirrorDownloadTests()
    {
        Directory.CreateDirectory(_root);
        KnobStore.Set(EngineKnobs.ModelsRoot, _root);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        Directory.Delete(_root, recursive: true);
    }

    private static string Sha(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static ModelAsset PickleAsset(bool withUrl = false) => new()
    {
        Repo = "acme/model", RepoPath = "unet/weights.bin", TargetSubdir = "3D/mirror-test", TargetName = "unet/weights.bin", Role = "transformer",
        Sha256 = Sha(Pickle),
        Url = withUrl ? "https://releases.example/acme/weights.bin" : null,
        Repack = new AssetRepack { MirrorPath = "mirror-test/unet/weights.safetensors", Sha256 = Sha(Repack) },
    };

    private static HttpClient Client(StandInNetwork network) => new(network);

    [Fact]
    public async Task AMirrorRepackIsFetchedFirstAndSavedBesideThePickleWithoutTouchingUpstream()
    {
        ModelAsset asset = PickleAsset();
        StandInNetwork network = new(request => request.RequestUri!.AbsoluteUri == $"{Mirror}/mirror-test/unet/weights.safetensors" ? Ok(Repack) : throw new InvalidOperationException("unexpected " + request.RequestUri));
        using HuggingFaceClient hf = new("hf_secret", new HttpClient(new StandInNetwork(_ => throw new InvalidOperationException("Hugging Face must not be asked"))));

        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(network), Mirror + "/", CancellationToken.None);

        string repacked = ModelDownloader.RepackedTargetPath(asset);
        Assert.Equal(Path.Combine(_root, "3D", "mirror-test", "unet", "weights.safetensors"), Path.GetFullPath(repacked));
        Assert.Equal(Repack, File.ReadAllBytes(repacked));
        Assert.False(File.Exists(ModelDownloader.TargetPath(asset)));
        Assert.Single(network.Requests);
        Assert.Empty(ModelDownloader.MissingAssets(new CatalogEntry { Id = "t", Modality = Modality.Mesh, DisplayName = "t", Architecture = "t", Status = ModelStatus.Structural, Assets = [asset] }));
    }

    [Fact]
    public async Task AnInstallThatHoldsOnlyTheRepackOfThePrimaryResolvesToTheRepackNotToAPathThatDoesNotExist()
    {
        ModelAsset asset = PickleAsset();
        CatalogEntry entry = new() { Id = "t", Modality = Modality.Mesh, DisplayName = "t", Architecture = "t", Status = ModelStatus.Structural, Assets = [asset] };
        Assert.Equal(ModelDownloader.TargetPath(asset), ModelDownloader.PrimaryLocalPath(entry));
        using HuggingFaceClient hf = new(null, new HttpClient(new StandInNetwork(_ => throw new InvalidOperationException("Hugging Face must not be asked"))));

        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(new StandInNetwork(_ => Ok(Repack))), Mirror, CancellationToken.None);

        string primary = ModelDownloader.PrimaryLocalPath(entry)!;
        Assert.True(File.Exists(primary));
        Assert.EndsWith("weights.safetensors", primary, StringComparison.Ordinal);
    }

    [Fact]
    public void TheResolverPointsAtTheRepackOfTheCatalogPrimaryWhenOnlyTheRepackIsInstalled()
    {
        CatalogEntry entry = ModelCatalog.Find("hunyuan3d-paint21")!;
        ModelAsset primary = ModelDownloader.PrimaryAsset(entry)!;
        string repack = ModelDownloader.RepackedTargetPath(primary);
        Directory.CreateDirectory(Path.GetDirectoryName(repack)!);
        File.WriteAllBytes(repack, Repack);

        ModelSpec spec = ModelResolver.Resolve("hunyuan3d-paint21", null, Modality.Mesh);

        Assert.Equal(Path.GetFullPath(repack), Path.GetFullPath(spec.LocalPath!));
        Assert.Equal(Path.GetFullPath(repack), Path.GetFullPath(ModelDownloader.PrimaryLocalPath(entry)!));
    }

    [Fact]
    public async Task AMirrorThatServesTheWrongBytesFallsBackToUpstreamAndLeavesNoPartialFile()
    {
        ModelAsset asset = PickleAsset(withUrl: true);
        StandInNetwork network = new(request => request.RequestUri!.Host == "mirror.example" ? Ok([0, 0, 0]) : Ok(Pickle));
        using HuggingFaceClient hf = new(null, new HttpClient(new StandInNetwork(_ => throw new InvalidOperationException("not used for a URL asset"))));

        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(network), Mirror, CancellationToken.None);

        Assert.Equal(Pickle, File.ReadAllBytes(ModelDownloader.TargetPath(asset)));
        Assert.False(File.Exists(ModelDownloader.RepackedTargetPath(asset)));
        Assert.Empty(Directory.EnumerateFiles(_root, "*.tmp", SearchOption.AllDirectories));
        Assert.Equal(["mirror.example", "releases.example"], network.Requests.Select(r => r.Host));
    }

    [Fact]
    public async Task AMirrorThatDoesNotAnswerFallsBackToHuggingFaceWithItsToken()
    {
        ModelAsset asset = PickleAsset();
        StandInNetwork mirror = new(_ => new HttpResponseMessage(HttpStatusCode.NotFound));
        StandInNetwork huggingFace = new(request => request.RequestUri!.AbsolutePath == "/acme/model/resolve/main/unet/weights.bin" ? Ok(Pickle) : new HttpResponseMessage(HttpStatusCode.NotFound));
        using HuggingFaceClient hf = new("hf_secret", new HttpClient(huggingFace));

        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(mirror), Mirror, CancellationToken.None);

        Assert.Equal(Pickle, File.ReadAllBytes(ModelDownloader.TargetPath(asset)));
        Assert.Equal("Bearer", huggingFace.Authorizations.Single()?.Scheme);
        Assert.Null(mirror.Authorizations.Single());
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("http://mirror.example/weights")]
    [InlineData("not a url")]
    public async Task WithoutAnHttpsMirrorTheMirrorIsNeverAsked(string? mirrorBase)
    {
        ModelAsset asset = PickleAsset();
        StandInNetwork mirror = new(_ => throw new InvalidOperationException("the mirror must not be asked"));
        StandInNetwork huggingFace = new(_ => Ok(Pickle));
        using HuggingFaceClient hf = new(null, new HttpClient(huggingFace));

        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(mirror), mirrorBase, CancellationToken.None);

        Assert.Empty(mirror.Requests);
        Assert.Equal(Pickle, File.ReadAllBytes(ModelDownloader.TargetPath(asset)));
    }

    [Fact]
    public async Task ARepackAlreadyOnDiskSatisfiesTheAssetAndNothingIsFetched()
    {
        ModelAsset asset = PickleAsset();
        Directory.CreateDirectory(Path.GetDirectoryName(ModelDownloader.RepackedTargetPath(asset))!);
        File.WriteAllBytes(ModelDownloader.RepackedTargetPath(asset), Repack);
        StandInNetwork network = new(_ => throw new InvalidOperationException("nothing may be fetched"));
        using HuggingFaceClient hf = new(null, new HttpClient(network));

        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(network), Mirror, CancellationToken.None);

        Assert.Empty(network.Requests);
        Assert.Equal(ModelDownloader.RepackedTargetPath(asset), ModelDownloader.FindRepacked(asset));
    }

    [Fact]
    public async Task AReleaseAssetIsFetchedFromItsUrlWithoutTheHuggingFaceTokenAndCheckedAgainstItsHash()
    {
        ModelAsset asset = PickleAsset(withUrl: true) with { Repack = null };
        StandInNetwork wrong = new(_ => Ok([0, 0]));
        using HuggingFaceClient hf = new("hf_secret", new HttpClient(new StandInNetwork(_ => throw new InvalidOperationException("Hugging Face must not be asked"))));

        InvalidOperationException mismatch = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(wrong), null, CancellationToken.None));

        Assert.Contains("SHA-256 mismatch", mismatch.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(ModelDownloader.TargetPath(asset)));
        Assert.Empty(Directory.EnumerateFiles(_root, "*.tmp", SearchOption.AllDirectories));

        StandInNetwork right = new(_ => Ok(Pickle));
        await ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(right), null, CancellationToken.None);

        Assert.Equal(Pickle, File.ReadAllBytes(ModelDownloader.TargetPath(asset)));
        Assert.Null(right.Authorizations.Single());
    }

    [Fact]
    public async Task AUrlDownloadWithoutAPinnedHashIsRefused()
    {
        ModelAsset asset = PickleAsset(withUrl: true) with { Repack = null, Sha256 = null };
        using HuggingFaceClient hf = new(null, new HttpClient(new StandInNetwork(_ => Ok(Pickle))));

        await Assert.ThrowsAsync<ArgumentException>(() =>
            ModelDownloader.EnsureAsync(hf, asset, allowDownload: true, null, Client(new StandInNetwork(_ => Ok(Pickle))), null, CancellationToken.None));
        Assert.False(File.Exists(ModelDownloader.TargetPath(asset)));
    }

    [Fact]
    public async Task WithDownloadsOffAMissingAssetIsAnErrorEvenWhenAMirrorIsConfigured()
    {
        ModelAsset asset = PickleAsset();
        StandInNetwork network = new(_ => throw new InvalidOperationException("no network"));
        using HuggingFaceClient hf = new(null, new HttpClient(network));

        await Assert.ThrowsAsync<FileNotFoundException>(() =>
            ModelDownloader.EnsureAsync(hf, asset, allowDownload: false, null, Client(network), Mirror, CancellationToken.None));
        Assert.Empty(network.Requests);
    }

    [Theory]
    [InlineData("https://mirror.example/weights", "mirror-test/unet/weights.safetensors", "https://mirror.example/weights/mirror-test/unet/weights.safetensors")]
    [InlineData("https://mirror.example/weights/", "a b/c.safetensors", "https://mirror.example/weights/a%20b/c.safetensors")]
    [InlineData("https://mirror.example", "x.safetensors", "https://mirror.example/x.safetensors")]
    [InlineData("http://mirror.example", "x.safetensors", null)]
    [InlineData("https://mirror.example/w?sig=abc", "x.safetensors", null)]
    [InlineData("https://mirror.example/w#frag", "x.safetensors", null)]
    [InlineData("https://mirror.example", "../x.safetensors", null)]
    [InlineData("https://mirror.example", "a//x.safetensors", null)]
    [InlineData("https://mirror.example", "a\\x.safetensors", null)]
    [InlineData("  ", "x.safetensors", null)]
    public void TheMirrorUrlIsBuiltOnlyFromAnHttpsBaseAndAPathThatStaysInsideIt(string baseUrl, string mirrorPath, string? expected)
    {
        ModelAsset asset = PickleAsset() with { Repack = new AssetRepack { MirrorPath = mirrorPath, Sha256 = Sha(Repack) } };

        Assert.Equal(expected, AssetUrlDownload.MirrorUrl(asset, baseUrl));
    }

    [Fact]
    public void AnAssetWithNoRepackHasNoMirrorUrlAndTheKnobIsOffByDefault()
    {
        Assert.Null(AssetUrlDownload.MirrorUrl(PickleAsset() with { Repack = null }, Mirror));
        Assert.True(string.IsNullOrEmpty(EngineKnobs.WeightsMirrorUrl.Default));
    }

    [Fact]
    public void EveryCatalogUrlIsHttpsAndPinnedAndEveryRepackIsAHashedSafetensorsBesideAPickle()
    {
        System.Text.RegularExpressions.Regex hash = new("^[0-9a-f]{64}$");
        string[] pickles = [".bin", ".pth", ".pt", ".ckpt"];
        ModelAsset[] assets = [.. ModelCatalog.All.SelectMany(e => e.Assets.Select(a => a))];

        Assert.All(assets.Where(a => a.Url is not null), a =>
        {
            Assert.StartsWith("https://", a.Url, StringComparison.Ordinal);
            Assert.Matches(hash, a.Sha256 ?? string.Empty);
        });
        IEnumerable<ModelAsset> repacked = assets.Where(a => a.Repack is not null);
        Assert.NotEmpty(repacked);
        Assert.All(repacked, a =>
        {
            Assert.Contains(Path.GetExtension(a.FileName), pickles);
            Assert.Matches(hash, a.Repack!.Sha256);
            Assert.Matches(hash, a.Sha256 ?? string.Empty);
            Assert.NotEqual(a.Sha256, a.Repack.Sha256);
            // The mirror path is <catalog folder name>/<file name with .safetensors>, the layout the staging tool writes.
            string folder = a.TargetSubdir[(a.TargetSubdir.LastIndexOf('/') + 1)..];
            Assert.Equal($"{folder}/{Path.ChangeExtension(a.FileName, ".safetensors")}".Replace('\\', '/'), a.Repack.MirrorPath);
            Assert.NotNull(AssetUrlDownload.MirrorUrl(a, "https://mirror.example"));
        });
    }

    private static HttpResponseMessage Ok(byte[] body) => new(HttpStatusCode.OK) { Content = new ByteArrayContent(body) };

    /// <summary>A network that answers from a function and remembers who asked, with which credentials.</summary>
    private sealed class StandInNetwork(Func<HttpRequestMessage, HttpResponseMessage> answer) : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = [];

        public List<System.Net.Http.Headers.AuthenticationHeaderValue?> Authorizations { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request.RequestUri!);
            Authorizations.Add(request.Headers.Authorization);
            return Task.FromResult(answer(request));
        }
    }
}
