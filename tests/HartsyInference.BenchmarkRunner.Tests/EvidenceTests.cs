using System.Buffers.Binary;
using System.IO.Compression;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Evidence;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Publication;
using HartsyInference.BenchmarkRunner.Serialization;
using HartsyInference.Engine;
using Xunit;

namespace HartsyInference.BenchmarkRunner.Tests;
/// <summary>Adversarial evidence and aggregation tests; synthetic data never leaves the temporary directory.</summary>
public sealed class EvidenceTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-bench-tests-" + Guid.NewGuid().ToString("N"));
    public EvidenceTests() => Directory.CreateDirectory(_root);
    [Theory]
    [InlineData("../escape")]
    [InlineData("/absolute")]
    [InlineData("foo\\bar")]
    [InlineData("foo/../bar")]
    [InlineData("C:/payload")]
    [InlineData("foo//bar")]
    [InlineData("foo:bar")]
    public void UnsafePathsAreRejected(string path) => Assert.Throws<InvalidDataException>(() => Hashes.SafePath(_root, path));
    [Fact]
    public void SymlinkEscapeIsRejected()
    {
        if (OperatingSystem.IsWindows())
            return; // Windows requires symlink privilege; ZIP link test runs on both systems.
        string outside = Path.Combine(_root, "outside");
        Directory.CreateDirectory(outside);
        Directory.CreateSymbolicLink(Path.Combine(_root, "link"), outside);
        Assert.Throws<InvalidDataException>(() => Hashes.SafePath(_root, "link/payload.txt"));
    }

    [Theory]
    [InlineData("{\"a\":\"1\",\"a\":\"2\"}")]
    [InlineData("{\"a\":\"1\",\"A\":\"2\"}")]
    public void DuplicateJsonKeysAreRejected(string json)
    {
        string path = Path.Combine(_root, "duplicate.json");
        File.WriteAllText(path, json);
        Assert.Throws<InvalidDataException>(() => BenchJson.Read(path, BenchJson.Default.SortedDictionaryStringString));
    }

    [Fact]
    public void ValidBundleRoundTripsAndCorruptionFails()
    {
        string campaign = Fixture("quick-v1");
        Assert.True(Validator.Validate(campaign).Valid);
        Assert.False(Validator.Validate(campaign).HeadlineEligible);
        string zip = Path.Combine(_root, "evidence.zip"), extracted = Path.Combine(_root, "extracted");
        Bundle.Export(campaign, zip);
        Bundle.Extract(zip, extracted);
        Assert.True(Validator.Validate(extracted).Valid);
        CampaignRecord record = Read(extracted);
        File.AppendAllText(Path.Combine(extracted, record.Sessions[0].Measurements[0].Output), "tampered");
        Assert.False(Validator.Validate(extracted).Valid);
    }

    [Theory]
    [InlineData("../escape.txt", 0)]
    [InlineData("sessions/link.txt", 0xA000)]
    [InlineData("sessions/payload.exe", 0)]
    public void MaliciousZipEntriesAreRejected(string name, int unixType)
    {
        string zip = Path.Combine(_root, "evil.zip");
        using (ZipArchive archive = ZipFile.Open(zip, ZipArchiveMode.Create))
        {
            ZipArchiveEntry entry = archive.CreateEntry(name);
            entry.ExternalAttributes = unixType << 16;
            using (StreamWriter writer = new(entry.Open()))
                writer.Write("payload");
            using StreamWriter second = new(archive.CreateEntry("checksums.json").Open());
            second.Write("{}");
        }

        Assert.Throws<InvalidDataException>(() => Bundle.Extract(zip, Path.Combine(_root, "unpack")));
    }

    [Fact]
    public void ZipCaseAliasesAreRejected()
    {
        string zip = Path.Combine(_root, "alias.zip");
        using (ZipArchive archive = ZipFile.Open(zip, ZipArchiveMode.Create))
        {
            archive.CreateEntry("campaign.json");
            archive.CreateEntry("CAMPAIGN.json");
        }

        Assert.Throws<InvalidDataException>(() => Bundle.Extract(zip, Path.Combine(_root, "unpack")));
    }

    [Fact]
    public void MissingTrialsAndFabricatedRatesFailValidation()
    {
        string root = Fixture("quick-v1");
        CampaignRecord campaign = Read(root);
        SessionRecord session = campaign.Sessions[0];
        Save(root, campaign with { Sessions = [session with { Measurements = session.Measurements[..^1] }] });
        Assert.False(Validator.Validate(root).Valid);
        Measurement[] measurements = session.Measurements.ToArray();
        measurements[2] = measurements[2] with
        {
            DecodeTokensPerSecond = 9999999
        };
        Save(root, campaign with { Sessions = [session with { Measurements = measurements }] });
        Assert.False(Validator.Validate(root).Valid);
    }

    [Fact]
    public void WrongDeviceAndEditedSuiteCannotBecomeHeadlineResults()
    {
        string root = Fixture("standard-v1");
        CampaignRecord campaign = Read(root);
        Assert.True(Validator.Validate(root).HeadlineEligible);
        SessionRecord[] sessions = campaign.Sessions.ToArray();
        sessions[0] = sessions[0] with
        {
            ActualDevice = sessions[0].ActualDevice!with
            {
                Selector = "cpu"
            }
        };
        Save(root, campaign with { Sessions = sessions });
        Assert.False(Validator.Validate(root).Valid);
        Save(root, campaign with { SuiteSha256 = new string ('0', 64) });
        Assert.False(Validator.Validate(root).Valid);
    }

    [Fact]
    public void FailedAndSkippedAttemptsAreValidButNotHeadlineEligible()
    {
        string root = Fixture("standard-v1");
        CampaignRecord campaign = Read(root);
        Save(root, campaign with { Sessions = campaign.Sessions.Select(s => s with { Status = "timeout" }).ToArray() });
        ValidationReport report = Validator.Validate(root);
        Assert.True(report.Valid);
        Assert.False(report.HeadlineEligible);
    }

    [Fact]
    public void ImageDecoderBoundsExpansionAndRejectsConstantOutputs()
    {
        string file = Path.Combine(_root, "image.png");
        File.WriteAllBytes(file, PngEncoder.Encode(new byte[12], 2, 2));
        Assert.False(OutputChecks.Image(file, 2, 2));
        Assert.Throws<InvalidDataException>(() => OutputChecks.Image(file, 512, 512));
        byte[] pixels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
        File.WriteAllBytes(file, PngEncoder.Encode(pixels, 2, 2));
        Assert.True(OutputChecks.Image(file, 2, 2));
        byte[] corrupt = File.ReadAllBytes(file);
        corrupt[^1] ^= 1;
        File.WriteAllBytes(file, corrupt);
        Assert.Throws<InvalidDataException>(() => OutputChecks.Image(file, 2, 2));
    }

    [Theory]
    [InlineData("http://github.com/a/b/releases/download/t/x.zip")]
    [InlineData("https://localhost/private.zip")]
    [InlineData("https://github.com.evil/a/b/releases/download/t/x.zip")]
    [InlineData("https://github.com/a/b/releases/download/t/x.zip?token=secret")]
    [InlineData("https://github.com/a/b/releases/download/t/../../x.zip")]
    public void UntrustedEvidenceUrlsAreRejected(string url) => Assert.False(Submissions.ReleaseUrl(url));
    [Fact]
    public void AggregationDoesNotPickFastestUploadAndWithdrawalRemovesResult()
    {
        string submissions = Path.Combine(_root, "submissions"), receipts = Path.Combine(_root, "reviews"), evidence = Path.Combine(_root,
            "evidence");
        Directory.CreateDirectory(receipts);
        string first = Fixture("standard-v1"), second = Fixture("standard-v1");
        CampaignRecord original = Read(first);
        CampaignRecord later = Read(second)with
        {
            CreatedUtc = original.CreatedUtc.AddHours(1)
        };
        Save(second, later);
        foreach (string root in new[]
        {
            first,
            second
        }

        )
        {
            string zip = root + ".zip";
            Bundle.Export(root, zip);
            Submissions.Create(root, zip, submissions, "https://github.com/test/fork/releases/download/bench/evidence.zip");
            string id = Hashes.FileHash(Path.Combine(root, "campaign.json"));
            Bundle.Extract(zip, Path.Combine(evidence, id));
            BenchJson.Write(Path.Combine(receipts, id + ".json"), new ReviewRecord { SubmissionId = id, BundleSha256 = Hashes.FileHash(zip),
                CreatedUtc = DateTimeOffset.UtcNow, Status = "accepted", Reviewer = "test",
                PullRequest = "https://github.com/HartsyAI/HartsyInference/pull/1", VerifiedHead = new string ('a', 40),
                Reason = "Test fixture" }, BenchJson.Default.ReviewRecord);
        }

        SummaryRow[] rows = Publisher.Build(submissions, receipts, evidence, Path.Combine(_root, "site"));
        Assert.Single(rows);
        Assert.Equal(1, rows[0].Machines);
        Assert.Equal(3, rows[0].Sessions);
        Assert.Single(rows[0].Evidence);
        Assert.Contains(Hashes.FileHash(Path.Combine(first, "campaign.json")), rows[0].Evidence[0]);
        foreach (string file in Directory.GetFiles(receipts))
        {
            ReviewRecord review = BenchJson.Read(file, BenchJson.Default.ReviewRecord);
            BenchJson.Write(file + ".withdrawn.json", review with { Status = "withdrawn", CreatedUtc = review.CreatedUtc.AddDays(1) },
                BenchJson.Default.ReviewRecord);
        }

        Assert.Empty(Publisher.Build(submissions, receipts, evidence, Path.Combine(_root, "withdrawn")));
    }

    [Fact]
    public void SoftwareVulkanAndMissingSessionsAreNotGpuResults()
    {
        string root = Fixture("standard-v1");
        CampaignRecord campaign = Read(root);
        DeviceRecord software = campaign.Environment.Device with
        {
            Selector = "vulkan:0",
            HardwareKind = "cpu",
            Name = "llvmpipe"
        };
        Save(root, campaign with { Environment = campaign.Environment with { Device = software }, Sessions = campaign.Sessions.Select(
            s => s with { ActualDevice = s.ActualDevice is null ? null : software }).ToArray() });
        Assert.True(Validator.Validate(root).Valid);
        Assert.False(Validator.Validate(root).HeadlineEligible);
        Save(root, campaign with { Sessions = campaign.Sessions[..^1] });
        Assert.False(Validator.Validate(root).Valid);
    }

    [Fact]
    public void EmptyFailedOutputIsPreservedAsEvidence()
    {
        string root = Fixture("quick-v1");
        CampaignRecord campaign = Read(root);
        Measurement[] measurements = campaign.Sessions[0].Measurements.ToArray();
        string output = Path.Combine(root, measurements[0].Output);
        File.WriteAllText(output, "");
        measurements[0] = measurements[0] with
        {
            QualityPassed = false,
            OutputSha256 = Hashes.FileHash(output)
        };
        Save(root, campaign with { Sessions = [campaign.Sessions[0] with { Status = "quality-failed", Measurements = measurements }] });
        Assert.True(Validator.Validate(root).Valid);
        Bundle.Export(root, Path.Combine(_root, "failed-output.zip"));
    }

    [Fact]
    public void StatisticsAreStableAndDoNotPretendOneMachineHasPopulationUncertainty()
    {
        Assert.Equal(2.5, Statistics.Median([1, 2, 3, 4]));
        Assert.Equal((12.0, 12.0), Statistics.Interval([12]));
        Assert.Equal(Statistics.Interval([10, 20, 30]), Statistics.Interval([10, 20, 30]));
        Assert.Throws<ArgumentException>(() => Statistics.Median([double.NaN]));
    }

    [Fact]
    public void ValidPngCompressionVariantsDoNotRequireIdenticalEncoderBytes()
    {
        byte[] pixels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
        byte[] original = PngEncoder.Encode(pixels, 2, 2);
        byte[] scanlines = [0, 0, 10, 20, 30, 40, 50, 0, 60, 70, 80, 90, 100, 110];
        using MemoryStream compressed = new();
        using (ZLibStream compressor = new(compressed, CompressionLevel.NoCompression, true)) compressor.Write(scanlines);
        byte[] data = compressed.ToArray();
        byte[] png = new byte[33 + 12 + data.Length + 12];
        original.AsSpan(0, 33).CopyTo(png);
        BinaryPrimitives.WriteInt32BigEndian(png.AsSpan(33), data.Length);
        "IDAT"u8.CopyTo(png.AsSpan(37));
        data.CopyTo(png, 41);
        BinaryPrimitives.WriteUInt32BigEndian(png.AsSpan(41 + data.Length), PngEncoder.Crc32(png.AsSpan(37, 4 + data.Length)));
        original.AsSpan(original.Length - 12).CopyTo(png.AsSpan(png.Length - 12));
        string file = Path.Combine(_root, "alternate.png"); File.WriteAllBytes(file, png);
        Assert.True(OutputChecks.Image(file, 2, 2));
    }

    private string Fixture(string suiteId)
    {
        string root = Path.Combine(_root, Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        SuiteDefinition suite = Suites.Load(suiteId);
        DeviceRecord device = new()
        {
            Selector = "cuda:0",
            Name = "SYNTHETIC TEST DEVICE",
            Driver = "test",
            HardwareKind = "gpu",
            Identity = new string ('1', 64),
            MemoryBytes = 1024
        };
        List<SessionRecord> sessions = [];
        foreach (CaseDefinition definition in suite.Cases)
            for (int session = 0; session < suite.Sessions; session++)
            {
                List<Measurement> measurements = [];
                if (definition.Id == suite.Cases[0].Id)
                    for (int index = 0; index < suite.Warmups + definition.Inputs.Length; index++)
                    {
                        string lane = index == 0 ? "first" : index < suite.Warmups ? "warmup" : "warm";
                        int input = index < suite.Warmups ? index : index - suite.Warmups;
                        string output = $"sessions/{definition.Id}/{session}/1/{lane}-{input}.txt";
                        string file = Path.Combine(root, output);
                        Directory.CreateDirectory(Path.GetDirectoryName(file)!);
                        File.WriteAllText(file, "Synthetic test output. Never a measured engine result.");
                        measurements.Add(new Measurement { Input = input, Lane = lane, ElapsedMs = 1000, StopwatchFrequency = 1000,
                            StartedTicks = 1000, RequestStartedTicks = 1001, PrefillTicks = 1010, CompletedTicks = 2000,
                            TokenTimestamps = [1100, 1500], FirstTokenMs = 99, DecodeTokensPerSecond = 2.5, PromptTokens = 10,
                            CompletionTokens = 2, StopReason = "Stop", Output = output, OutputSha256 = Hashes.FileHash(file),
                            QualityPassed = true, QualityDetail = "Synthetic test fixture" });
                    }

                sessions.Add(new SessionRecord { CaseId = definition.Id, Session = session, Attempt = 1, Status = measurements
                    .Count > 0 ? "completed" : "budget-skipped", StartedUtc = DateTimeOffset.UtcNow, NativeLibraries = new(
                    ) { ["libcuda.test"] = new string ('4', 64) }, Measurements = measurements.ToArray(), ActualDevice = measurements
                    .Count > 0 ? device : null });
            }

        Save(root, new CampaignRecord { SuiteId = suiteId, SuiteSha256 = Hashes.FileHash(Suites.PathFor(suiteId)),
            CreatedUtc = DateTimeOffset.UtcNow, Sessions = sessions.ToArray(),
            Environment = new EnvironmentRecord { MachineId = new string ('2', 64), EngineRevision = new string ('a', 40),
            EngineVersion = "test", OperatingSystem = "test", Runtime = "test", Architecture = "x64", CpuCount = 1, Device = device,
            Binaries = new() { ["engine.dll"] = new string ('3', 64) }, Settings = new() { ["test"] = "true" }, } });
        return root;
    }

    private static CampaignRecord Read(string root) => BenchJson.Read(Path.Combine(root, "campaign.json"), BenchJson.Default.CampaignRecord);
    private static void Save(string root, CampaignRecord campaign) => BenchJson.Write(Path.Combine(root, "campaign.json"), campaign,
        BenchJson.Default.CampaignRecord);
    public void Dispose() => Directory.Delete(_root, recursive: true);
}
