using System.Diagnostics;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Execution;
/// <summary>Serial campaign scheduler with immutable attempts and explicit incomplete outcomes.</summary>
public static class Campaign
{
    public static async Task<int> RunAsync(string root, string cache, string suiteId, string selector, int budgetMinutes, bool resume,
        CancellationToken cancel)
    {
        SuiteDefinition suite = Suites.Load(suiteId);
        string manifest = Path.Combine(root, "campaign.json");
        if (File.Exists(manifest) && !resume)
            throw new IOException("Campaign exists; use resume or a new output directory.");
        if (resume && !File.Exists(manifest))
            throw new IOException("No campaign to resume.");
        Directory.CreateDirectory(root);
        string devicePath = Path.Combine(root, "device.json");
        int probe = await ChildProcess.RunAsync(["probe", selector, devicePath], Path.Combine(root, "probe.log"), TimeSpan.FromMinutes(2),
            cancel);
        if (probe != 0)
            throw new IOException("Backend probe failed. Inspect probe.log; no CPU fallback was used.");
        DeviceRecord device = BenchJson.Read(devicePath, BenchJson.Default.DeviceRecord);
        EnvironmentRecord environment = Hardware.Capture(device);
        CampaignRecord campaign = resume ? BenchJson.Read(manifest, BenchJson.Default.CampaignRecord) : new CampaignRecord
        {
            SuiteId = suiteId,
            SuiteSha256 = Hashes.FileHash(Suites.PathFor(suiteId)),
            CreatedUtc = DateTimeOffset.UtcNow,
            Environment = environment,
            Sessions = [],
            BudgetMinutes = budgetMinutes
        };
        if (campaign.Environment.MachineId != environment.MachineId
            || campaign.Environment.OperatingSystem != environment.OperatingSystem
            || campaign.Environment.Runtime != environment.Runtime
            || campaign.Environment.Architecture != environment.Architecture
            || campaign.Environment.CpuCount != environment.CpuCount
            || campaign.SuiteId != suiteId || campaign.SuiteSha256 != Hashes.FileHash(Suites.PathFor(suiteId)) || campaign.Environment
            .Device != device || campaign.Environment.EngineRevision != environment.EngineRevision || !campaign.Environment.Binaries
            .SequenceEqual(environment.Binaries) || !campaign.Environment.Settings.SequenceEqual(environment.Settings))
            throw new InvalidDataException("Resume requires the same suite, binary hashes, device, settings, and engine revision.");
        foreach (AssetDefinition asset in suite.Cases.Select(c => c.Asset).DistinctBy(a => a.Sha256))
            Assets.Verify(cache, asset);
        BenchJson.Write(manifest, campaign, BenchJson.Default.CampaignRecord);
        List<SessionRecord> sessions = campaign.Sessions.ToList();
        if (resume)
        {
            string journals = Path.Combine(root, "sessions");
            foreach (string journal in Directory.Exists(journals) ? Directory.GetFiles(journals, "session.json", SearchOption
                .AllDirectories) : [])
            {
                SessionRecord orphan = BenchJson.Read(journal, BenchJson.Default.SessionRecord);
                if (sessions.Any(s => s.CaseId == orphan.CaseId && s.Session == orphan.Session && s.Attempt == orphan.Attempt))
                    continue;
                if (orphan.Status == "running")
                    orphan = orphan with
                    {
                        Status = "crashed",
                        Failure = "controller-interrupted"
                    };
                sessions.Add(orphan);
            }

            campaign = campaign with
            {
                Sessions = sessions.ToArray()
            };
            BenchJson.Write(manifest, campaign, BenchJson.Default.CampaignRecord);
        }

        if (sessions.Count + suite.Cases.Length * suite.Sessions > 200)
            throw new InvalidOperationException("Campaign attempt limit reached; start a new campaign.");
        long start = Stopwatch.GetTimestamp();
        foreach (CaseDefinition definition in suite.Cases)
            for (int session = 0; session < suite.Sessions; session++)
            {
                SessionRecord[] previous = sessions.Where(s => s.CaseId == definition.Id && s.Session == session).ToArray();
                if (previous.Any(s => s.Status == "completed"))
                    continue;
                // Recover an interrupted controller's journal without reusing its attempt directory.
                string sessionRoot = Path.Combine(root, "sessions", definition.Id, session.ToString());
                int diskAttempt = Directory.Exists(sessionRoot) ? Directory.GetDirectories(sessionRoot).Select(p => int.TryParse(Path
                    .GetFileName(p), out int n) ? n : 0).DefaultIfEmpty(0).Max() : 0;
                int attempt = Math.Max(previous.Select(s => s.Attempt).DefaultIfEmpty(0).Max(), diskAttempt) + 1;
                double remaining = budgetMinutes * 60 - Stopwatch.GetElapsedTime(start).TotalSeconds;
                string directory = Path.Combine(sessionRoot, attempt.ToString());
                Directory.CreateDirectory(directory);
                string journal = Path.Combine(directory, "session.json");
                SessionRecord record;
                if (remaining <= 0 || cancel.IsCancellationRequested)
                {
                    record = new SessionRecord
                    {
                        CaseId = definition.Id,
                        Session = session,
                        Attempt = attempt,
                        StartedUtc = DateTimeOffset.UtcNow,
                        Status = cancel.IsCancellationRequested ? "cancelled" : "budget-skipped",
                        Measurements = []
                    };
                }
                else
                {
                    Console.WriteLine($"{definition.Id}: session {session + 1}/{suite.Sessions}, attempt {attempt}");
                    int code = await ChildProcess.RunAsync(["worker", root, cache, suiteId, definition.Id, selector, session.ToString(),
                        attempt.ToString()], Path.Combine(directory, "worker.log"), TimeSpan.FromSeconds(Math.Min(remaining, definition
                        .TimeoutSeconds)), cancel);
                    record = File.Exists(journal) ? BenchJson.Read(journal, BenchJson.Default.SessionRecord) : new SessionRecord
                    {
                        CaseId = definition.Id,
                        Session = session,
                        Attempt = attempt,
                        StartedUtc = DateTimeOffset.UtcNow,
                        Status = "crashed",
                        Measurements = []
                    };
                    if (code != 0 || record.Status == "running")
                        record = record with
                        {
                            Status = code == 124 ? "timeout" : code == 130 ? "cancelled" : record.Status == "running" ? "crashed" : record
                                .Status,
                            Failure = record.Failure ?? "worker-exit-" + code,
                        };
                }

                BenchJson.Write(journal, record, BenchJson.Default.SessionRecord);
                sessions.Add(record);
                campaign = campaign with
                {
                    Sessions = sessions.ToArray()
                };
                BenchJson.Write(manifest, campaign, BenchJson.Default.CampaignRecord);
            }

        return suite.Cases.All(c => Enumerable.Range(0, suite.Sessions).All(s => sessions.Any(r => r.CaseId == c.Id && r.Session == s && r
            .Status == "completed"))) ? 0 : 1;
    }
}
