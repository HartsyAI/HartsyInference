using System.Net;
using HartsyInference.Audio.Cache;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Exceptions;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.HuggingFace;

namespace HartsyInference.Engine.Tests;

/// <summary>Keeps the process-wide refusal of these tests away from tests that download through a fake server.</summary>
[CollectionDefinition("DownloadPermission", DisableParallelization = true)]
public sealed class DownloadPermissionCollection;

/// <summary>A refused download never reaches the network, whichever path the engine fetches through, and leaves nothing on disk.</summary>
[Collection("DownloadPermission")]
public sealed class DownloadPermissionTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-download-" + Guid.NewGuid().ToString("N"));
    private readonly CountingHandler _network = new();
    private readonly HttpClient _http;

    public DownloadPermissionTests() => _http = new HttpClient(_network);

    public void Dispose()
    {
        _http.Dispose();
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public async Task ARefusal_StopsEveryDownloadPath_BeforeTheNetwork()
    {
        string target = Path.Combine(_dir, "sub", "file.bin");
        string audioRepo = $"hartsy-tests/none-{Guid.NewGuid():N}";
        using HuggingFaceClient hub = new(token: null, _http);
        using IDisposable refused = DownloadPermission.Refuse("a test says so");

        List<DownloadRefusedException> errors =
        [
            await Assert.ThrowsAsync<DownloadRefusedException>(() => hub.DownloadFileAsync("some/repo", "weights.safetensors", target, null, null, default)),
            await Assert.ThrowsAsync<DownloadRefusedException>(
                () => AssetUrlDownload.DownloadAsync(_http, "https://example.com/weights.bin", "weights", target, null, new string('0', 64), default)),
            await Assert.ThrowsAsync<DownloadRefusedException>(() => AudioFileFetcher.EnsureAsync(_http, "https://example.com/lexicon.json", target, null, default)),
            await Assert.ThrowsAsync<DownloadRefusedException>(() => AudioModelCache.GetAsync(audioRepo, "model.safetensors", "tts")),
        ];

        Assert.All(errors, e => Assert.Contains("a test says so", e.Message));
        Assert.Contains("some/repo/weights.safetensors", errors[0].Message);
        Assert.Contains($"{audioRepo}/model.safetensors", errors[3].Message);
        Assert.Equal(0, _network.Requests);
        Assert.False(Directory.Exists(_dir));
        // Resolving the repo's folder creates it, so it exists either way; an empty one is all a refusal may leave.
        string audioDirectory = AudioModelCache.GetRepoDirectory(audioRepo, "tts");
        Assert.Empty(Directory.EnumerateFileSystemEntries(audioDirectory));
        Directory.Delete(audioDirectory);
    }

    [Fact]
    public async Task WithoutARefusal_TheDownloadIsMade()
    {
        using HuggingFaceClient hub = new(token: null, _http);

        await Assert.ThrowsAnyAsync<Exception>(() => hub.DownloadFileAsync("some/repo", "weights.safetensors", Path.Combine(_dir, "f"), null, null, default));

        Assert.Equal(1, _network.Requests);
    }

    [Fact]
    public void ScopesNest_AndAGrantEndsWithItsScope()
    {
        using (DownloadPermission.Refuse("outer"))
        {
            Assert.Throws<DownloadRefusedException>(() => DownloadPermission.Demand("a"));
            using (DownloadPermission.Grant())
                DownloadPermission.Demand("b");
            Assert.Contains("outer", Assert.Throws<DownloadRefusedException>(() => DownloadPermission.Demand("c")).Message);
            using (DownloadPermission.Refuse("inner"))
                Assert.Contains("inner", Assert.Throws<DownloadRefusedException>(() => DownloadPermission.Demand("d")).Message);
        }
        DownloadPermission.Demand("e");
    }

    [Fact]
    public async Task TheRefusal_HoldsOnOtherThreads()
    {
        using IDisposable refused = DownloadPermission.Refuse("everywhere");
        Task<Exception?> other;
        using (ExecutionContext.SuppressFlow())
            other = Task.Run(static Exception? () => Record.Exception(static () => DownloadPermission.Demand("x")));

        Assert.IsType<DownloadRefusedException>(await other);
    }

    [Fact]
    public void TheSetting_IsTheAnswer_EitherWay()
    {
        using (KnobProfileScope.Push(KnobProfile.Create("no").With(EngineKnobs.AllowDownloads, false)))
        using (DownloadPermission.Grant())
            Assert.Contains("paths.allowDownloads is off", Assert.Throws<DownloadRefusedException>(() => DownloadPermission.Demand("x")).Message);

        using (KnobProfileScope.Push(KnobProfile.Create("yes").With(EngineKnobs.AllowDownloads, true)))
        using (DownloadPermission.Refuse("no terminal"))
            DownloadPermission.Demand("x");
    }

    private sealed class CountingHandler : HttpMessageHandler
    {
        public int Requests { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests++;
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.NotFound));
        }
    }
}
