using System.Globalization;
using System.Text.Json;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Evidence;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Publication;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner;
/// <summary>Portable community benchmark command entry point.</summary>
public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        using CancellationTokenSource cancellation = new();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;
            cancellation.Cancel();
        };
        try
        {
            if (args.Length == 3 && args[0] == "probe")
            {
                BenchJson.Write(args[2], Hardware.Probe(args[1]), BenchJson.Default.DeviceRecord);
                return 0;
            }

            if (args.Length == 8 && args[0] == "worker")
                return await Worker.RunAsync(args[1], args[2], args[3], args[4], args[5], int.Parse(args[6]), int.Parse(args[7]),
                    cancellation.Token);
            if (args.Length == 0 || args[0] is "help" or "--help")
            {
                Console.WriteLine("""
                    hartsy-bench — reproducible community evidence
                      list
                      doctor --device cuda:0
                      fetch --suite standard-v1 --cache <directory>
                      run|resume --suite standard-v1 --device cuda:0 --cache <directory> --output <campaign> [--minutes 30]
                      validate --input <campaign-or-extracted-bundle>
                      export --input <campaign> --bundle <new.zip>
                      extract --bundle <zip> --output <empty-directory>
                      submission --input <campaign> --bundle <zip> --output <submissions-root> --staging <github-release-url>
                      submit --input <campaign> --bundle <zip> --repository <checkout> --fork <owner/repository>
                      metadata --input <submission-directory>
                      download --input <submission-directory> --bundle <new.zip> [--source staging|archive]
                      review --input <submission-directory> --output <new.json> --status accepted|withdrawn
                             --reviewer <github-login> --pr <url> --head <sha> --reason <text>
                      publish --input <submissions-root> --reviews <trusted-receipts> --evidence <extracted-root> --output <site>
                    All model downloads happen in fetch. run never silently changes device, model, or workload.
                    """);
                return 0;
            }

            Dictionary<string, string> options = Parse(args);
            string Need(string key) => options.Remove(key, out string? value) ? value : throw new ArgumentException("Missing --" + key);
            string Get(string key, string fallback) => options.Remove(key, out string? value) ? value : fallback;
            void End()
            {
                if (options.Count != 0)
                    throw new ArgumentException("Unknown option --" + options.Keys.First());
            }

            switch (args[0])
            {
                case "schema":
                    string schemaDirectory = Need("output");
                    End();
                    Schemas.Write(schemaDirectory);
                    return 0;
                case "list":
                    End();
                    foreach (string id in Suites.AvailableIds)
                    {
                        SuiteDefinition suite = Suites.Load(id);
                        Console.WriteLine($"{id}: {suite.Cases.Length} cases, {suite.Sessions} sessions, publishable={suite.Publishable}");
                    }

                    return 0;
                case "doctor":
                {
                    string device = Need("device");
                    End();
                    string root = Path.Combine(Path.GetTempPath(), "hartsy-doctor-" + Guid.NewGuid().ToString("N"));
                    Directory.CreateDirectory(root);
                    int code = await ChildProcess.RunAsync(["probe", device, Path.Combine(root, "device.json")], Path.Combine(root,
                        "probe.log"), TimeSpan.FromMinutes(2), cancellation.Token);
                    Console.WriteLine(code == 0 ? File.ReadAllText(Path.Combine(root, "device.json")) : "Probe failed: " + Path.Combine(
                        root, "probe.log"));
                    return code;
                }

                case "fetch":
                {
                    string suite = Get("suite", "standard-v1"), cache = Path.GetFullPath(Need("cache"));
                    End();
                    await Assets.FetchAsync(Suites.Load(suite), cache, cancellation.Token);
                    return 0;
                }

                case "run":
                case "resume":
                {
                    string suite = Get("suite", "standard-v1"), device = Need("device"), cache = Path.GetFullPath(Need("cache"));
                    string output = Path.GetFullPath(Need("output"));
                    int minutes = int.Parse(Get("minutes", suite == "extended-v1" ? "120" : "30"), CultureInfo.InvariantCulture);
                    End();
                    if (minutes is < 1 or > 1440)
                        throw new ArgumentOutOfRangeException("minutes");
                    return await Campaign.RunAsync(output, cache, suite, device, minutes, args[0] == "resume", cancellation.Token);
                }

                case "validate":
                {
                    string input = Need("input");
                    End();
                    ValidationReport report = Validator.Validate(input);
                    Console.WriteLine(JsonSerializer.Serialize(report, BenchJson.Default.ValidationReport));
                    return report.Valid ? 0 : 1;
                }

                case "export":
                {
                    string input = Need("input"), bundle = Need("bundle");
                    End();
                    Bundle.Export(input, bundle);
                    return 0;
                }

                case "extract":
                {
                    string bundle = Need("bundle"), output = Need("output");
                    End();
                    Bundle.Extract(bundle, output);
                    return 0;
                }

                case "submission":
                {
                    string input = Need("input"), bundle = Need("bundle"), output = Need("output"), staging = Need("staging");
                    End();
                    Submissions.Create(input, bundle, output, staging);
                    return 0;
                }

                case "submit":
                {
                    string input = Need("input"), bundle = Need("bundle"), repository = Need("repository"), fork = Need("fork");
                    End();
                    await SubmitCommand.RunAsync(input, bundle, repository, fork, cancellation.Token);
                    return 0;
                }

                case "metadata":
                {
                    string input = Need("input");
                    End();
                    Submissions.ValidateMetadata(input);
                    return 0;
                }

                case "download":
                {
                    string input = Need("input"), bundle = Need("bundle"), source = Get("source", "archive");
                    End();
                    if (source is not ("archive" or "staging"))
                        throw new ArgumentException("Unknown evidence source.");
                    await Submissions.FetchAsync(input, bundle, source == "staging", cancellation.Token);
                    return 0;
                }

                case "review":
                {
                    string input = Need("input"), output = Need("output"), status = Need("status"), reviewer = Need("reviewer");
                    string pr = Need("pr"), head = Need("head"), reason = Need("reason");
                    End();
                    SubmissionRecord submission = Submissions.ValidateMetadata(input);
                    if (File.Exists(output) || status is not ("accepted" or "withdrawn") || reason.Length is < 1 or > 2000 || !System.Text
                        .RegularExpressions.Regex.IsMatch(head, "^[0-9a-f]{40}$") || !System.Text.RegularExpressions.Regex.IsMatch(reviewer,
                        "^[A-Za-z0-9-]{1,39}$") || !System.Text.RegularExpressions.Regex.IsMatch(pr,
                        @"^https://github\.com/HartsyAI/HartsyInference/pull/[0-9]+$"))
                        throw new InvalidDataException("Invalid review receipt.");
                    BenchJson.Write(output, new ReviewRecord { SubmissionId = submission.Id, BundleSha256 = submission.BundleSha256,
                        CreatedUtc = DateTimeOffset.UtcNow, Status = status, Reviewer = reviewer, PullRequest = pr, VerifiedHead = head,
                        Reason = reason }, BenchJson.Default.ReviewRecord);
                    return 0;
                }

                case "publish":
                {
                    string input = Need("input"), reviews = Need("reviews"), evidence = Need("evidence"), output = Need("output");
                    End();
                    SummaryRow[] rows = Publisher.Build(input, reviews, evidence, output);
                    Console.WriteLine($"Published {rows.Length} comparable cohorts.");
                    return 0;
                }

                default:
                    throw new ArgumentException("Unknown command; use --help.");
            }
        }
        catch (Exception error)
        {
            Console.Error.WriteLine(error.Message);
            return 1;
        }
    }

    private static Dictionary<string, string> Parse(string[] args)
    {
        Dictionary<string, string> result = new(StringComparer.Ordinal);
        for (int i = 1; i < args.Length; i += 2)
            if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length || !result.TryAdd(args[i][2..], args[i + 1]))
                throw new ArgumentException("Expected unique --option value pairs.");
        return result;
    }
}
