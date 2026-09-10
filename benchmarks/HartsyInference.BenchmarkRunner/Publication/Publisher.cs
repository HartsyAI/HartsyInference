using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Evidence;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Publication;

/// <summary>Deterministic derived artifacts from reviewed, intact evidence and compatible workloads.</summary>
public static class Publisher
{
    public static SummaryRow[] Build(string submissionsRoot, string reviewsRoot, string extractedRoot, string output)
    {
        List<AcceptedCase> accepted = [];
        List<string> excluded = [];
        ReviewRecord[] reviews = Directory.Exists(reviewsRoot)
            ? Directory.GetFiles(reviewsRoot, "*.json").Select(p => BenchJson.Read(p, BenchJson.Default.ReviewRecord)).ToArray()
            : [];
        IEnumerable<string> submissions = Directory.Exists(submissionsRoot)
            ? Directory.GetDirectories(submissionsRoot).Order(StringComparer.Ordinal) : [];
        foreach (string directory in submissions)
        {
            SubmissionRecord submission = Submissions.ValidateMetadata(directory);
            ReviewRecord? review = reviews.Where(r => r.SubmissionId == submission.Id)
                .OrderByDescending(r => r.CreatedUtc).ThenBy(r => r.Status, StringComparer.Ordinal).FirstOrDefault();
            if (review is null || review.Status != "accepted" || review.BundleSha256 != submission.BundleSha256)
            {
                excluded.Add(submission.Id + ": awaiting review or withdrawn");
                continue;
            }
            string evidence = Path.Combine(extractedRoot, submission.Id);
            ValidationReport report = Validator.Validate(evidence);
            if (!report.Valid || !report.HeadlineEligible
                || Hashes.FileHash(Path.Combine(evidence, "campaign.json")) != submission.CampaignSha256)
            {
                excluded.Add(submission.Id + ": incomplete, ineligible, or unavailable evidence");
                continue;
            }
            CampaignRecord campaign = BenchJson.Read(Path.Combine(evidence, "campaign.json"), BenchJson.Default.CampaignRecord);
            SuiteDefinition suite = Suites.Load(campaign.SuiteId);
            foreach (CaseDefinition definition in suite.Cases.Where(c => Validator.Complete(campaign, suite, c.Id)))
                accepted.Add(new AcceptedCase(submission, campaign, definition));
        }

        SummaryRow[] rows = accepted.GroupBy(CohortKey).OrderBy(g => g.Key, StringComparer.Ordinal)
            .Select(Summarize).OrderBy(r => r.CaseId, StringComparer.Ordinal)
            .ThenBy(r => r.EngineRevision, StringComparer.Ordinal).ThenBy(r => r.Gpu, StringComparer.Ordinal)
            .ThenBy(r => r.Backend, StringComparer.Ordinal).ThenBy(r => r.Driver, StringComparer.Ordinal)
            .ThenBy(r => r.Configuration, StringComparer.Ordinal).ThenBy(r => r.NativeConfiguration, StringComparer.Ordinal)
            .ToArray();
        Directory.CreateDirectory(output);
        BenchJson.Write(Path.Combine(output, "summary.json"), rows, BenchJson.Default.SummaryRowArray);
        File.WriteAllText(Path.Combine(output, "excluded.txt"), string.Join('\n', excluded.Order(StringComparer.Ordinal)));
        File.WriteAllText(Path.Combine(output, "overview.svg"), Svg(rows));
        foreach (string file in new[] { "index.html", "explorer.js" })
            File.Copy(Path.Combine(AppContext.BaseDirectory, "web", file), Path.Combine(output, file), true);
        return rows;
    }

    private static SummaryRow Summarize(IGrouping<string, AcceptedCase> group)
    {
        AcceptedCase first = group.First();
        // One vote per physical GPU. Later uploads cannot replace an earlier run just because they are faster.
        AcceptedCase[] machines = group.GroupBy(x => x.Campaign.Environment.Device.Identity)
            .Select(g => g.OrderBy(x => x.Campaign.CreatedUtc).ThenBy(x => x.Submission.Id, StringComparer.Ordinal).First())
            .OrderBy(x => x.Campaign.Environment.Device.Identity, StringComparer.Ordinal).ToArray();
        double[] medians = machines.Select(x => MachineMetric(x, m => m.ElapsedMs, "warm")!.Value).ToArray();
        (double low, double high) = Statistics.Interval(medians);
        EnvironmentRecord environment = first.Campaign.Environment;
        return new SummaryRow
        {
            CaseId = first.Case.Id, SuiteId = first.Campaign.SuiteId, EngineRevision = environment.EngineRevision,
            Gpu = environment.Device.Name, Backend = environment.Device.Selector.Split(':')[0], Driver = environment.Device.Driver,
            DeviceMemoryBytes = environment.Device.MemoryBytes, OperatingSystem = environment.OperatingSystem,
            Runtime = environment.Runtime, NativeConfiguration = NativeConfiguration(first), Configuration = Configuration(first.Campaign),
            MedianFirstRequestMs = Metric(machines, m => m.ElapsedMs, "first"),
            MedianFirstTokenMs = Metric(machines, m => m.FirstTokenMs),
            MedianDecodeTokensPerSecond = Metric(machines, m => m.DecodeTokensPerSecond),
            MedianOutputTokens = first.Case.Adapter == "text" ? Metric(machines, m => m.CompletionTokens) : null,
            MedianMs = Statistics.Median(medians), MeanMs = medians.Average(), IntervalLowMs = low, IntervalHighMs = high,
            Machines = machines.Length, Sessions = machines.Length * Suites.Load(first.Campaign.SuiteId).Sessions,
            Evidence = machines.Select(x => x.Submission.ArchiveUrl).Order(StringComparer.Ordinal).ToArray(),
        };
    }

    private static double? Metric(AcceptedCase[] machines, Func<Measurement, double?> selector, string lane = "warm")
    {
        double?[] values = machines.Select(m => MachineMetric(m, selector, lane)).ToArray();
        return values.Any(v => v is null) ? null : Statistics.Median(values.Select(v => v!.Value));
    }

    private static double? MachineMetric(AcceptedCase machine, Func<Measurement, double?> selector, string lane)
    {
        List<double> sessionValues = [];
        foreach (SessionRecord session in machine.Campaign.Sessions.Where(s => s.CaseId == machine.Case.Id && s.Status == "completed")
            .OrderBy(s => s.Session))
        {
            double?[] values = session.Measurements.Where(m => m.Lane == lane).Select(selector).ToArray();
            if (values.Length == 0 || values.Any(v => v is null)) return null;
            sessionValues.Add(Statistics.Median(values.Select(v => v!.Value)));
        }
        return Statistics.Median(sessionValues);
    }

    private static string CohortKey(AcceptedCase item)
    {
        EnvironmentRecord environment = item.Campaign.Environment;
        return Fingerprint(new SortedDictionary<string, string>(StringComparer.Ordinal)
        {
            ["suite"] = item.Campaign.SuiteSha256, ["case"] = item.Case.Id, ["engine"] = environment.EngineRevision,
            ["device"] = environment.Device.Name, ["capacity"] = environment.Device.MemoryBytes.ToString(CultureInfo.InvariantCulture),
            ["backend"] = environment.Device.Selector.Split(':')[0], ["driver"] = environment.Device.Driver,
            ["os"] = environment.OperatingSystem, ["runtime"] = environment.Runtime, ["power"] = environment.PowerProfile,
            ["configuration"] = Configuration(item.Campaign), ["native"] = NativeConfiguration(item),
        });
    }

    private static string NativeConfiguration(AcceptedCase item)
    {
        SortedDictionary<string, string> libraries = new(StringComparer.Ordinal);
        foreach (SessionRecord session in item.Campaign.Sessions.Where(s => s.CaseId == item.Case.Id && s.Status == "completed"))
            foreach (KeyValuePair<string, string> library in session.NativeLibraries)
                libraries[session.Session.ToString(CultureInfo.InvariantCulture) + "/" + library.Key] = library.Value;
        return Fingerprint(libraries);
    }

    private static string Configuration(CampaignRecord campaign)
    {
        SortedDictionary<string, string> fields = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, string> setting in campaign.Environment.Settings) fields["setting/" + setting.Key] = setting.Value;
        foreach (KeyValuePair<string, string> binary in campaign.Environment.Binaries) fields["binary/" + binary.Key] = binary.Value;
        fields["cpu-count"] = campaign.Environment.CpuCount.ToString(CultureInfo.InvariantCulture);
        fields["architecture"] = campaign.Environment.Architecture;
        return Fingerprint(fields);
    }

    private static string Fingerprint(SortedDictionary<string, string> fields) =>
        Hashes.Text(JsonSerializer.Serialize(fields, BenchJson.Default.SortedDictionaryStringString));

    private static string Svg(SummaryRow[] rows)
    {
        int height = 100 + rows.Length * 58;
        StringBuilder svg = new($"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1000\" height=\"{height}\" ");
        svg.Append("role=\"img\" aria-label=\"Reviewed community benchmark results\">");
        svg.Append("<rect width=\"100%\" height=\"100%\" fill=\"#101827\"/><g fill=\"#e5edf5\" font-family=\"sans-serif\">");
        svg.Append("<text x=\"24\" y=\"32\" font-size=\"20\">HartsyInference • reviewed community results</text>");
        if (rows.Length == 0)
            svg.Append("<text x=\"24\" y=\"70\">No accepted reproducible campaigns yet. Contribute a run to build this dataset.</text>");
        for (int i = 0; i < rows.Length; i++)
        {
            SummaryRow row = rows[i];
            string gpu = row.Gpu.Length > 42 ? row.Gpu[..39] + "…" : row.Gpu;
            string label = $"{row.CaseId} | {gpu} | {row.Backend} | {row.EngineRevision[..7]} | "
                + row.MedianMs.ToString("F1", CultureInfo.InvariantCulture) + $" ms | n={row.Machines}";
            double maximum = rows.Where(r => r.CaseId == row.CaseId && r.EngineRevision == row.EngineRevision).Max(r => r.MedianMs);
            string width = (900 * row.MedianMs / maximum).ToString("F1", CultureInfo.InvariantCulture);
            svg.Append($"<text x=\"24\" y=\"{72 + i * 58}\" font-size=\"13\">").Append(WebUtility.HtmlEncode(label)).Append("</text>");
            svg.Append($"<rect x=\"24\" y=\"{82 + i * 58}\" width=\"{width}\" height=\"6\" fill=\"#7dd3fc\"/>");
        }
        return svg.Append("</g></svg>\n").ToString();
    }
}
