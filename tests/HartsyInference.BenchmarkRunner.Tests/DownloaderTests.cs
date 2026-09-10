using System.Net;
using System.Text;
using HartsyInference.Engine.HuggingFace;
using HartsyInference.BenchmarkRunner.Serialization;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;
/// <summary>Pinned URLs and atomic hash verification are checked without reaching the network.</summary>
public sealed class DownloaderTests
{
    [Fact]
    public async Task DownloadUsesPinnedRevisionAndRejectsWrongHash()
    {
        string root = Path.Combine(Path.GetTempPath(), "hartsy-download-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            RecordingHandler handler = new();
            using HttpClient http = new(handler);
            using HuggingFaceClient client = new(httpClient: http);
            string file = Path.Combine(root, "weights.bin"), revision = new string ('a', 40);
            await client.DownloadRevisionAsync("test/model", "weights.bin", revision, file, null, Hashes.Text("fixture"), CancellationToken
                .None);
            Assert.Equal($"https://huggingface.co/test/model/resolve/{revision}/weights.bin", handler.Url);
            Assert.Equal("fixture", File.ReadAllText(file));
            string corrupt = Path.Combine(root, "corrupt.bin");
            await Assert.ThrowsAnyAsync<Exception>(() => client.DownloadRevisionAsync("test/model", "weights.bin", revision, corrupt, null,
                new string ('0', 64), CancellationToken.None));
            Assert.False(File.Exists(corrupt));
            Assert.False(File.Exists(corrupt + ".tmp"));
            await Assert.ThrowsAsync<ArgumentException>(() => client.DownloadRevisionAsync("test/model", "weights.bin", "../main", file,
                null, null, CancellationToken.None));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private sealed class RecordingHandler : HttpMessageHandler
    {
        public string? Url { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Url = request.RequestUri!.ToString();
            return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(Encoding.UTF8.GetBytes(
                "fixture")) });
        }
    }
}
