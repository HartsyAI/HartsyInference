using System.Net;
using System.Text.RegularExpressions;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Evidence;
/// <summary>Small repository records refer to immutable GitHub Release bundles; only allowlisted hosts are downloaded.</summary>
public static class Submissions
{
    public const string ArchivePrefix = "https://github.com/HartsyAI/HartsyInference/releases/download/benchmark-evidence/";
    public static void Create(string root, string bundle, string output, string stagingUrl)
    {
        string campaignHash = Hashes.FileHash(Path.Combine(root, "campaign.json"));
        string bundleHash = Hashes.FileHash(bundle);
        string id = campaignHash;
        SubmissionRecord submission = new()
        {
            Id = id,
            CampaignSha256 = campaignHash,
            BundleSha256 = bundleHash,
            BundleBytes = new FileInfo(bundle).Length,
            StagingUrl = stagingUrl,
            ArchiveUrl = ArchivePrefix + id + ".zip"
        };
        Check(submission);
        string directory = Path.Combine(output, id);
        if (Directory.Exists(directory))
            throw new IOException("Submission already exists; records are immutable.");
        Directory.CreateDirectory(directory);
        File.Copy(Path.Combine(root, "campaign.json"), Path.Combine(directory, "campaign.json"));
        BenchJson.Write(Path.Combine(directory, "submission.json"), submission, BenchJson.Default.SubmissionRecord);
        ValidateMetadata(directory);
    }

    public static void Check(SubmissionRecord submission)
    {
        if (submission.SchemaVersion != 1 || !Hashes.IsHash(submission.Id) || submission.Id != submission.CampaignSha256 || !Hashes.IsHash(
            submission.BundleSha256) || submission.BundleBytes is < 1 or > Bundle.MaximumBytes || submission.ArchiveUrl != ArchivePrefix
            + submission.Id + ".zip" || !ReleaseUrl(submission.StagingUrl))
            throw new InvalidDataException("Invalid submission identity or Release URL.");
    }

    public static bool ReleaseUrl(string value) => value is not null && Regex.IsMatch(value,
        @"^https://github\.com/[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+/releases/download/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\.zip$");
    public static SubmissionRecord ValidateMetadata(string directory)
    {
        string[] files = Directory.GetFiles(directory, "*", SearchOption.AllDirectories);
        if (files.Length != 2 || files.Sum(p => new FileInfo(p).Length) > 2 * 1024 * 1024)
            throw new InvalidDataException("Submission must contain only campaign.json and submission.json, at most 2 MiB combined.");
        SubmissionRecord submission = BenchJson.Read(Hashes.SafePath(directory, "submission.json"), BenchJson.Default.SubmissionRecord);
        Check(submission);
        if (Path.GetFileName(Path.TrimEndingDirectorySeparator(directory)) != submission.Id || Hashes.FileHash(Hashes.SafePath(directory,
            "campaign.json")) != submission.CampaignSha256)
            throw new InvalidDataException("Submission directory or campaign hash mismatch.");
        return submission;
    }

    public static async Task FetchAsync(string directory, string output, bool staging, CancellationToken cancel)
    {
        SubmissionRecord submission = ValidateMetadata(directory);
        await DownloadAsync(staging ? submission.StagingUrl : submission.ArchiveUrl, output, submission.BundleBytes, cancel);
        if (Hashes.FileHash(output) != submission.BundleSha256)
            throw new InvalidDataException("Downloaded bundle hash mismatch.");
    }

    public static async Task DownloadAsync(string url, string output, long expectedBytes, CancellationToken cancel)
    {
        if (!ReleaseUrl(url))
            throw new InvalidDataException("Only GitHub Release evidence URLs are accepted.");
        using HttpClient client = new(new HttpClientHandler { AllowAutoRedirect = false })
        {
            Timeout = TimeSpan.FromMinutes(10)
        };
        Uri current = new(url);
        for (int hop = 0; hop < 5; hop++)
        {
            using HttpResponseMessage response = await client.GetAsync(current, HttpCompletionOption.ResponseHeadersRead, cancel);
            if ((int)response.StatusCode is >= 300 and < 400)
            {
                current = response.Headers.Location is Uri location ? new Uri(current, location) : throw new IOException("Missing redirect.");
                if (current.Scheme != "https" || !current.IsDefaultPort || current.UserInfo.Length != 0 || current.Host is not (
                    "github.com" or "release-assets.githubusercontent.com" or "objects.githubusercontent.com"))
                    throw new InvalidDataException("Untrusted evidence redirect.");
                continue;
            }

            response.EnsureSuccessStatusCode();
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
            await using Stream input = await response.Content.ReadAsStreamAsync(cancel);
            await using FileStream file = new(output, FileMode.CreateNew);
            byte[] buffer = new byte[65536];
            long count = 0;
            int read;
            while ((read = await input.ReadAsync(buffer, cancel)) > 0)
            {
                count += read;
                if (count > expectedBytes || count > Bundle.MaximumBytes)
                    throw new InvalidDataException("Oversized evidence download.");
                await file.WriteAsync(buffer.AsMemory(0, read), cancel);
            }

            if (count != expectedBytes)
                throw new InvalidDataException("Downloaded size mismatch.");
            return;
        }

        throw new IOException("Too many evidence redirects.");
    }
}
