using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Execution;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Evidence;
/// <summary>Validates protocol and evidence, not the honesty of externally supplied timing claims.</summary>
public static class Validator
{
    public static ValidationReport Validate(string root)
    {
        List<string> errors = [], notes = [];
        bool eligible = false;
        try
        {
            CampaignRecord campaign = BenchJson.Read(Hashes.SafePath(root, "campaign.json"), BenchJson.Default.CampaignRecord);
            SuiteDefinition suite = Suites.Load(campaign.SuiteId);
            Require(campaign.SchemaVersion == 1 && campaign.SuiteSha256 == Hashes.FileHash(Suites.PathFor(suite.Id)),
                "Suite identity mismatch.");
            Require(campaign.BudgetMinutes is >= 1 and <= 1440 && campaign.CreatedUtc <= DateTimeOffset.UtcNow.AddDays(1),
                "Invalid campaign metadata.");
            EnvironmentRecord environment = campaign.Environment;
            Require(Hashes.IsHash(environment.MachineId) && Hashes.IsHash(environment.Device.Identity), "Missing machine/device identity.");
            Require(environment.Binaries.Count is> 0 and <= 5000 && environment.Binaries.All(p => Hashes.IsHash(p.Value)),
                "Missing binary identities.");
            Require(environment.Settings.Count is> 0 and <= 1000 && environment.CpuCount > 0, "Missing runtime settings.");
            Require(environment.Device.HardwareKind is "gpu" or "cpu", "Missing hardware class.");
            Require(environment.Device.Error is null && environment.Device.Name.Length is> 0 and < 200 && environment.Device.Driver
                .Length is> 0 and < 100, "Invalid backend provenance.");
            Require(environment.Device.Selector == "cpu" || System.Text.RegularExpressions.Regex.IsMatch(environment.Device.Selector,
                @"^(cuda|vulkan):[0-9]{1,2}$"), "Backend must be explicit.");
            Require(campaign.Sessions.Length <= 200, "Too many attempts.");
            HashSet<string> attempts = new(StringComparer.Ordinal), outputs = new(StringComparer.Ordinal);
            foreach (SessionRecord session in campaign.Sessions)
            {
                CaseDefinition definition = suite.Cases.Single(c => c.Id == session.CaseId);
                Require(session.Session >= 0 && session.Session < suite.Sessions && session.Attempt is> 0 and <= 200, "Invalid session index.");
                Require(attempts.Add($"{session.CaseId}/{session.Session}/{session.Attempt}"), "Duplicate attempt.");
                Require(session
                    .Status is "completed" or "quality-failed" or "failed" or "oom"
                    or "unsupported" or "timeout" or "cancelled" or "budget-skipped" or "crashed", "Invalid session status.");
                Require(session.Measurements.Length <= suite.Warmups + definition.Inputs.Length, "Too many measurements.");
                if (session.Measurements.Length > 0)
                    Require(session.ActualDevice == environment.Device, "Actual backend differs from declared device.");
                for (int index = 0; index < session.Measurements.Length; index++)
                {
                    Measurement measurement = session.Measurements[index];
                    string lane = index == 0 ? "first" : index < suite.Warmups ? "warmup" : "warm";
                    int input = index < suite.Warmups ? index : index - suite.Warmups;
                    Require(measurement.Lane == lane && measurement.Input == input, "Missing, reordered, or duplicate trial.");
                    Require(double.IsFinite(measurement.ElapsedMs) && measurement.ElapsedMs > 0 && measurement.ElapsedMs <= definition
                        .TimeoutSeconds * 1000, "Invalid elapsed time.");
                    Require(measurement.StopwatchFrequency > 0 && measurement.StartedTicks > 0 && measurement.CompletedTicks > measurement
                        .StartedTicks && measurement.RequestStartedTicks >= measurement.StartedTicks && measurement
                        .RequestStartedTicks <= measurement.CompletedTicks && Math.Abs(measurement.ElapsedMs - (measurement
                        .CompletedTicks - measurement.StartedTicks) * 1000.0 / measurement.StopwatchFrequency) < 0.001,
                        "Timing does not match raw ticks.");
                    Require(measurement.TokenTimestamps.Length == measurement.CompletionTokens && measurement.TokenTimestamps
                        .Length <= definition.MaxTokens, "Native token trace count differs.");
                    long previousToken = measurement.RequestStartedTicks;
                    foreach (long timestamp in measurement.TokenTimestamps)
                    {
                        Require(timestamp >= previousToken && timestamp <= measurement.CompletedTicks, "Nonmonotonic token trace.");
                        previousToken = timestamp;
                    }

                    Require(measurement.HostPeakBytes >= 0 && measurement.SampledUsedDeviceBytes is null && measurement
                        .MemorySource == "unavailable", "Unsupported memory measurement semantics.");
                    string suffix = definition.Adapter == "text" ? ".txt" : ".png";
                    string expected = $"sessions/{session.CaseId}/{session.Session}/{session.Attempt}/{lane}-{input}{suffix}";
                    Require(measurement.Output == expected && outputs.Add(expected), "Output path does not match trial identity.");
                    string file = Hashes.SafePath(root, expected);
                    Require(File.Exists(file) && new FileInfo(file).Length is >= 0 and <= 8 * 1024 * 1024 && Hashes.FileHash(
                        file) == measurement.OutputSha256, "Missing or corrupt output evidence.");
                    bool quality;
                    if (definition.Adapter == "text")
                    {
                        Require(measurement.CompletionTokens >= 0 && measurement.CompletionTokens <= definition.MaxTokens && measurement
                            .PromptTokens >= 0, "Invalid native token count.");
                        Require(measurement.FirstTokenMs is null || double.IsFinite(measurement.FirstTokenMs.Value) && measurement
                            .FirstTokenMs > 0 && measurement.FirstTokenMs <= measurement.ElapsedMs, "Invalid first-token latency.");
                        Require(measurement.DecodeTokensPerSecond is null || double.IsFinite(measurement.DecodeTokensPerSecond.Value)
                            && measurement.DecodeTokensPerSecond > 0, "Invalid decode rate.");
                        if (measurement.TokenTimestamps.Length > 0)
                        {
                            Require(measurement.PrefillTicks >= measurement.RequestStartedTicks && measurement.PrefillTicks <= measurement
                                .TokenTimestamps[0], "Missing prefill boundary.");
                            double firstMs = (measurement.TokenTimestamps[0] - measurement.RequestStartedTicks) * 1000.0 / measurement
                                .StopwatchFrequency;
                            Require(measurement.FirstTokenMs is not null && Math.Abs(firstMs - measurement.FirstTokenMs.Value) < 0.001,
                                "First-token metric differs from raw trace.");
                        }

                        if (measurement.TokenTimestamps.Length > 1)
                        {
                            long span = measurement.TokenTimestamps[^1] - measurement.TokenTimestamps[0];
                            double rate = (measurement.TokenTimestamps.Length - 1) * (double)measurement.StopwatchFrequency / span;
                            Require(span > 0 && measurement.DecodeTokensPerSecond is not null && Math.Abs(rate - measurement
                                .DecodeTokensPerSecond.Value) < 0.001, "Decode rate differs from raw trace.");
                        }

                        quality = OutputChecks.Text(file) && measurement.PromptTokens > 0 && measurement.CompletionTokens > 0 && measurement
                            .StopReason is "Stop" or "Length" && measurement.FirstTokenMs is not null && (measurement.CompletionTokens < 2
                            || measurement.DecodeTokensPerSecond is not null);
                    }
                    else
                    {
                        Require(measurement.FirstTokenMs is null && measurement.DecodeTokensPerSecond is null && measurement
                            .CompletionTokens == 0 && measurement.PromptTokens == 0, "Image has text-only metrics.");
                        try
                        {
                            quality = OutputChecks.Image(file, definition.Width, definition.Height);
                        }
                        catch (InvalidDataException)when (!measurement.QualityPassed)
                        {
                            quality = false;
                        }
                    }

                    Require(!measurement.QualityPassed || quality, "Claimed quality check is contradicted by output evidence.");
                }

                if (session.Status == "completed" && environment.Device.Selector != "cpu")
                    Require(session.NativeLibraries.Count is> 0 and <= 100 && session.NativeLibraries.Values.All(Hashes.IsHash),
                        "Missing loaded native library identities.");
                if (session.Status == "completed")
                    Require(session.Measurements.Length == suite.Warmups + definition.Inputs.Length && session.Measurements.All(m => m
                        .QualityPassed), "Completed session lacks passing full protocol.");
            }

            Require(campaign.Sessions.Where(s => s.Status == "completed").GroupBy(s => (s.CaseId, s.Session))
                .All(g => g.Count() == 1), "More than one completed attempt for a planned session.");
            Require(suite.Cases.All(c => Enumerable.Range(0, suite.Sessions).All(i => campaign.Sessions.Any(s => s.CaseId == c.Id && s
                .Session == i))), "Every planned session needs an outcome, including skips.");
            bool completeCase = suite.Cases.Any(c => Complete(campaign, suite, c.Id));
            bool revision = System.Text.RegularExpressions.Regex.IsMatch(environment.EngineRevision, "^[0-9a-f]{40}$");
            eligible = suite.Publishable && revision && environment.Device.HardwareKind == "gpu" && environment.Device.Selector != "cpu"
                && completeCase;
            if (!suite.Publishable)
                notes.Add("Quick suite is diagnostic and never included in headline comparisons.");
            if (!revision)
                notes.Add("Development build: no immutable engine revision.");
            if (!completeCase)
                notes.Add("No case has all independent sessions. Partial results are retained but excluded.");
            notes.Add("Automated validation proves format/protocol consistency. Maintainer output review is still required.");
        }
        catch (Exception error)when (error is not OutOfMemoryException)
        {
            errors.Add(error is InvalidDataException ? error.Message : "Malformed or incomplete evidence: " + error.GetType().Name);
        }

        return new ValidationReport
        {
            Valid = errors.Count == 0,
            HeadlineEligible = errors.Count == 0 && eligible,
            Errors = errors.ToArray(),
            Notes = notes.ToArray()
        };
    }

    public static bool Complete(CampaignRecord campaign, SuiteDefinition suite, string caseId) => Enumerable.Range(0, suite.Sessions).All(
        i => campaign.Sessions.Any(s => s.CaseId == caseId && s.Session == i && s.Status == "completed"));
    private static void Require(bool condition, string message)
    {
        if (!condition)
            throw new InvalidDataException(message);
    }
}
