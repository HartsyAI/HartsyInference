using System.Net;
using System.Security.Cryptography;
using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Engine.Tests.Audio;

/// <summary>RVC's two encoders come from their Hartsy card first, checked against a pinned hash, and anything short of the exact
/// bytes leaves nothing behind so the upstream pickle is converted instead. Over a stand-in network: nothing is downloaded.</summary>
public sealed class RvcHartsyFetchTests : IDisposable
{
    private static readonly byte[] Weights = [4, 2, 4, 2, 4, 2];
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "hartsy-rvc-" + Guid.NewGuid().ToString("N"));

    public RvcHartsyFetchTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private static string Sha(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private sealed class StandInNetwork(Func<HttpRequestMessage, HttpResponseMessage> answer) : HttpMessageHandler
    {
        public List<HttpRequestMessage> Requests { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request);
            return Task.FromResult(answer(request));
        }
    }

    private static HttpResponseMessage Ok(byte[] body) => new(HttpStatusCode.OK) { Content = new ByteArrayContent(body) };

    [Fact]
    public async Task TheCardsFileIsSavedWhenItsHashMatches()
    {
        StandInNetwork network = new(_ => Ok(Weights));
        string destination = Path.Combine(_dir, "contentvec.safetensors");

        bool fetched = await VcCatalog.TryFromHartsyAsync(new HttpClient(network), VcCatalog.ContentVecHartsyUrl, Sha(Weights), destination, "contentvec.safetensors", CancellationToken.None);

        Assert.True(fetched);
        Assert.Equal(Weights, File.ReadAllBytes(destination));
        HttpRequestMessage request = Assert.Single(network.Requests);
        Assert.Equal(VcCatalog.ContentVecHartsyUrl, request.RequestUri!.AbsoluteUri);
        Assert.Null(request.Headers.Authorization);
    }

    [Fact]
    public async Task OtherBytesAreDiscardedAndTheCallerFallsBack()
    {
        string destination = Path.Combine(_dir, "rmvpe.safetensors");

        bool fetched = await VcCatalog.TryFromHartsyAsync(new HttpClient(new StandInNetwork(_ => Ok([9, 9, 9]))), VcCatalog.RmvpeHartsyUrl, Sha(Weights), destination, "rmvpe.safetensors", CancellationToken.None);

        Assert.False(fetched);
        Assert.Empty(Directory.GetFiles(_dir));
    }

    [Theory]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.ServiceUnavailable)]
    public async Task AnErrorFromTheSiteFallsBack(HttpStatusCode status)
    {
        string destination = Path.Combine(_dir, "rmvpe.safetensors");

        bool fetched = await VcCatalog.TryFromHartsyAsync(new HttpClient(new StandInNetwork(_ => new HttpResponseMessage(status))), VcCatalog.RmvpeHartsyUrl, Sha(Weights), destination, "rmvpe.safetensors", CancellationToken.None);

        Assert.False(fetched);
        Assert.False(File.Exists(destination));
    }

    [Fact]
    public async Task ACancelledRunIsNotAFallback()
    {
        using CancellationTokenSource cancel = new();
        cancel.Cancel();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => VcCatalog.TryFromHartsyAsync(
            new HttpClient(new StandInNetwork(_ => Ok(Weights))), VcCatalog.ContentVecHartsyUrl, Sha(Weights), Path.Combine(_dir, "x.safetensors"), "x", cancel.Token));
    }

    /// <summary>The pins are the card's: the files on Hartsy's RVC v2 card hash to these.</summary>
    [Fact]
    public void ThePinsAreFullSha256s()
    {
        Assert.Matches("^[0-9a-f]{64}$", VcCatalog.ContentVecHartsySha256);
        Assert.Matches("^[0-9a-f]{64}$", VcCatalog.RmvpeHartsySha256);
        Assert.StartsWith("https://hartsy.ai/download/", VcCatalog.ContentVecHartsyUrl);
        Assert.StartsWith("https://hartsy.ai/download/", VcCatalog.RmvpeHartsyUrl);
    }
}
