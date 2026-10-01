using System.Net;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Engine.Audio;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>The SHA-256 gate on non-HuggingFace downloads. A check that always passes looks exactly like one that
/// works until a corrupted or substituted file is installed, so the rejection path is what these pin.</summary>
public sealed class AudioFileFetcherTests : IDisposable
{
    private static readonly byte[] Payload = Encoding.UTF8.GetBytes("rnnoise fixture");
    private readonly string _dir = Directory.CreateTempSubdirectory("audio-fetch-").FullName;

    [Fact]
    public async Task EnsureAsync_DiscardsADownloadWhoseSha256DoesNotMatch()
    {
        string target = Path.Combine(_dir, "model.tar.gz");
        using HttpClient client = new(new FixedResponse(Payload));

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            AudioFileFetcher.EnsureAsync(client, "https://example.invalid/model.tar.gz", target, new string('0', 64),
                CancellationToken.None));

        Assert.False(File.Exists(target));
        Assert.False(File.Exists(target + ".tmp"));
    }

    [Fact]
    public async Task EnsureAsync_KeepsADownloadWhoseSha256Matches_InEitherCase()
    {
        string target = Path.Combine(_dir, "model.tar.gz");
        using HttpClient client = new(new FixedResponse(Payload));
        string sha = Convert.ToHexString(SHA256.HashData(Payload));

        await AudioFileFetcher.EnsureAsync(client, "https://example.invalid/model.tar.gz", target, sha, CancellationToken.None);

        Assert.Equal(Payload, File.ReadAllBytes(target));
        Assert.False(File.Exists(target + ".tmp"));
    }

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    private sealed class FixedResponse(byte[] body) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken) =>
            Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(body) });
    }
}
