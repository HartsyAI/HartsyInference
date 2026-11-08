using System.Diagnostics;
using System.Globalization;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Configuration;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine;
using HartsyInference.OpParity;
using HartsyInference.OpParity.Cases;

// hartsy-opparity record|check|list --op <family|all> [--case <substring>] [--backend cpu|cuda[:n]|vulkan[:n]]
//                 --fixtures <dir> [--seed N] [--commit <sha>] [--require cpu,cuda,vulkan]
//
// record runs the CURRENT implementation on seeded inputs and writes the inputs, outputs and a manifest. It is run on
// the baseline commit before that implementation is modified. check re-runs the current implementation on the
// recorded inputs and judges it by the case's acceptance rule; a case the baseline backend did not serve but the
// current one does is judged against the CPU backend instead of a fixture; a required backend that cannot run fails.

Dictionary<string, string> args_ = ParseArgs(args);
if (args.Length == 0 || !args_.TryGetValue("command", out string? command))
{
    Console.Error.WriteLine("usage: hartsy-opparity record|check|list --op <family|all> --fixtures <dir> [--backend cpu] [--case <substring>] [--seed 1234] [--commit <sha>] [--require cpu,cuda]");
    return 2;
}
string family = args_.GetValueOrDefault("op", "all");
string? caseFilter = args_.GetValueOrDefault("case");
List<OpCase> selected = Registry.Cases().Where(c => (family == "all" || c.Family == family) && (caseFilter is null || c.Name.Contains(caseFilter, StringComparison.Ordinal))).ToList();
if (command == "list")
{
    foreach (OpCase c in selected)
        Console.WriteLine($"{c.Family}/{c.Name}\t{c.Acceptance.Kind}\trequired: {string.Join(",", c.RequiredBackends)}");
    return 0;
}
if (!args_.TryGetValue("fixtures", out string? root))
{
    Console.Error.WriteLine("--fixtures <dir> is required for record and check.");
    return 2;
}
string selector = args_.GetValueOrDefault("backend", "cpu");
int seed = int.Parse(args_.GetValueOrDefault("seed", "1234"), CultureInfo.InvariantCulture);
string kind = BackendFactory.Kind(selector);
string[] requireOverride = args_.TryGetValue("require", out string? req) ? req.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) : [];
using IBackend backend = BackendFactory.Create(selector);
string device = string.IsNullOrEmpty(backend.Capabilities.DeviceName) ? backend.Capabilities.Name : backend.Capabilities.DeviceName;
Console.WriteLine($"{command}: {selected.Count} case(s), backend {selector} ({device}), fixtures {root}");
// A run that selects nothing (an unknown op, or no case for this backend) verifies nothing, so it cannot pass.
if (selected.Count == 0)
{
    Console.WriteLine($"FAIL: no cases selected for --op {args_.GetValueOrDefault("op", "(all)")} on backend {selector}");
    return 1;
}

int failures = 0;
int compared = 0;
IBackend? host = null;
try
{
    foreach (OpCase c in selected)
    {
        string dir = FixtureStore.CaseDir(root, c.Family, c.Name, kind);
        bool required = (requireOverride.Length > 0 ? requireOverride : c.RequiredBackends).Contains(kind, StringComparer.Ordinal);
        if (command == "record")
        {
            Directory.CreateDirectory(dir);
            if (!c.SupportedOn(backend))
            {
                Manifest(c, "unsupported", new()).Save(FixtureStore.ManifestPath(dir));
                Console.WriteLine($"  {c.Family}/{c.Name}: unsupported on {kind} (recorded as such)");
                continue;
            }
            using OpTensors t = c.Build(seed);
            c.Run(backend, t);
            backend.Sync();
            Dictionary<string, string> meta = new(StringComparer.Ordinal) { ["family"] = c.Family, ["case"] = c.Name, ["backend"] = kind, ["seed"] = seed.ToString(CultureInfo.InvariantCulture) };
            Dictionary<string, string> hashes = new(StringComparer.Ordinal)
            {
                ["inputs.safetensors"] = FixtureStore.Save(Path.Combine(dir, "inputs.safetensors"), t.Inputs, meta),
                ["outputs.safetensors"] = FixtureStore.Save(Path.Combine(dir, "outputs.safetensors"), t.Outputs, meta),
            };
            FixtureManifest m = Manifest(c, "recorded", hashes);
            m.MayBeNonFinite.AddRange(t.MayBeNonFinite);
            m.Save(FixtureStore.ManifestPath(dir));
            Console.WriteLine($"  {c.Family}/{c.Name}: recorded ({t.Outputs.Count} output(s))");
            continue;
        }
        // check
        string manifestPath = FixtureStore.ManifestPath(dir);
        FixtureManifest? recorded = File.Exists(manifestPath) ? FixtureManifest.Load(manifestPath) : null;
        bool servedNow = c.SupportedOn(backend);
        if (!servedNow)
        {
            string status = required ? "FAIL (required backend does not serve this case)" : "skipped (unsupported)";
            if (required) failures++;
            Console.WriteLine($"  {c.Family}/{c.Name}: {status}");
            continue;
        }
        // A fixture check needs its recorded manifest. Only a case judged against the host by rule, or one the baseline
        // recorded as unsupported, may take the host path; a missing or unknown manifest must not pass silently.
        bool hostByRule = c.Acceptance.Kind == AcceptanceKind.HostReference;
        if (!hostByRule && recorded is null)
        {
            failures++;
            Console.WriteLine($"  {c.Family}/{c.Name}: FAIL (no recorded fixture; run record on the baseline first)");
            continue;
        }
        if (!hostByRule && recorded!.Status is not ("recorded" or "unsupported"))
        {
            failures++;
            Console.WriteLine($"  {c.Family}/{c.Name}: FAIL (fixture status '{recorded.Status}' is not a recorded result)");
            continue;
        }
        if (recorded is null || recorded.Status != "recorded" || hostByRule)
        {
            // Newly supported, or judged against the host by rule: the case's oracle, else the CPU backend, is the independent reference.
            using OpTensors mine = c.Build(seed);
            using OpTensors theirs = c.Build(seed);
            c.Run(backend, mine);
            backend.Sync();
            if (c.Oracle is { } oracle)
            {
                oracle(theirs);
            }
            else
            {
                host ??= BackendFactory.Create("cpu");
                c.Run(host, theirs);
                host.Sync();
            }
            string oracleName = c.Oracle is null ? "CPU backend" : "contract reference";
            Acceptance rule = hostByRule ? c.Acceptance : Acceptance.AgainstHost(c.Acceptance.MaxAbs, c.Acceptance.MaxRel, $"newly supported path, judged against the {oracleName}");
            string why = recorded is null ? "no fixture" : recorded.Status == "unsupported" ? "unsupported on the baseline" : "host-reference rule";
            failures += Report(c, mine, theirs.Outputs, rule, $"vs {oracleName} ({why})");
            compared++;
            continue;
        }
        if (recorded.KernelHashes.Count > 0)
        {
            Dictionary<string, string> now = KernelHashesFor(kind);
            IEnumerable<string> changed = recorded.KernelHashes.Where(kv => now.TryGetValue(kv.Key, out string? h) && h != kv.Value).Select(kv => kv.Key);
            string changedList = string.Join(", ", changed);
            if (changedList.Length > 0) Console.WriteLine($"  {c.Family}/{c.Name}: kernel artifacts changed since record: {changedList}");
        }
        Dictionary<string, Tensor> inputs = FixtureStore.Load(Path.Combine(dir, "inputs.safetensors"));
        Dictionary<string, Tensor> expected = FixtureStore.Load(Path.Combine(dir, "outputs.safetensors"));
        using OpTensors run = c.Build(seed);
        foreach ((string name, Tensor fixture) in inputs)
        {
            run.Inputs[name].Dispose();
            run.Inputs[name] = fixture;
        }
        foreach (Tensor o in run.Outputs.Values) o.AsSpan<byte>().Clear();
        c.Run(backend, run);
        backend.Sync();
        // The case's declared acceptance judges the check. The fixture's outputs stay the recorded ones; the tolerance a case declares
        // at the head, with its reason, is the one the check enforces. A manifest that recorded a different acceptance says so.
        Acceptance recordedAcceptance = new(Enum.Parse<AcceptanceKind>(recorded.AcceptanceKind), recorded.MaxAbs, recorded.MaxRel, recorded.Reason);
        if (recordedAcceptance.Kind != c.Acceptance.Kind || recordedAcceptance.MaxAbs != c.Acceptance.MaxAbs || recordedAcceptance.MaxRel != c.Acceptance.MaxRel)
            Console.WriteLine($"  {c.Family}/{c.Name}: acceptance changed since record: recorded {recordedAcceptance.Kind} ({recordedAcceptance.MaxAbs:E1} / {recordedAcceptance.MaxRel:E1}), judged {c.Acceptance.Kind} ({c.Acceptance.MaxAbs:E1} / {c.Acceptance.MaxRel:E1}): {c.Acceptance.Reason}");
        failures += Report(c, run, expected, c.Acceptance, "vs fixture", recorded.MayBeNonFinite);
        compared++;
        foreach (Tensor e in expected.Values) e.Dispose();
    }
}
finally
{
    host?.Dispose();
}
// A check that compared nothing (every selected case skipped as unsupported on this backend) verified nothing, so it cannot pass.
if (command == "check" && compared == 0)
{
    Console.WriteLine($"FAIL: no case was compared on backend {selector}; every selected case was skipped");
    return 1;
}
Console.WriteLine(failures == 0 ? "PASS" : $"FAIL: {failures} output(s) outside acceptance");
return failures == 0 ? 0 : 1;

static int Report(OpCase c, OpTensors run, IReadOnlyDictionary<string, Tensor> expected, Acceptance acceptance, string how, IReadOnlyCollection<string>? mayBeNonFinite = null)
{
    int failed = 0;
    foreach ((string name, Tensor actual) in run.Outputs)
    {
        bool allowNonFinite = run.MayBeNonFinite.Contains(name) || (mayBeNonFinite?.Contains(name) ?? false);
        CompareResult r = TensorCompare.Compare(name, expected[name], actual, acceptance, allowNonFinite);
        if (!r.Pass) failed++;
        Console.WriteLine($"  {c.Family}/{c.Name}.{name}: {(r.Pass ? "ok" : "FAIL")} {how}: {r.Detail}");
    }
    return failed;
}

FixtureManifest Manifest(OpCase c, string status, Dictionary<string, string> fileHashes) => new()
{
    Family = c.Family,
    Case = c.Name,
    Backend = kind,
    Device = device,
    Commit = args_.GetValueOrDefault("commit") ?? GitHead(),
    RecordedUtc = DateTime.UtcNow.ToString("O", CultureInfo.InvariantCulture),
    Status = status,
    SettingsFile = KnobFile.LoadedFrom,
    KernelHashes = KernelHashesFor(kind),
    FileHashes = fileHashes,
    AcceptanceKind = c.Acceptance.Kind.ToString(),
    MaxAbs = c.Acceptance.MaxAbs,
    MaxRel = c.Acceptance.MaxRel,
    Reason = c.Acceptance.Reason,
    RequiredBackends = [.. c.RequiredBackends],
};

static Dictionary<string, string> KernelHashesFor(string kind) => kind switch
{
    "cuda" => FixtureManifest.HashKernels(BackendFactory.KernelDir(BackendFactory.PtxDirName)),
    "vulkan" => FixtureManifest.HashKernels(BackendFactory.KernelDir(BackendFactory.SpirvDirName)),
    _ => new(StringComparer.Ordinal),
};

static string GitHead()
{
    try
    {
        using Process p = Process.Start(new ProcessStartInfo("git", "rev-parse HEAD") { RedirectStandardOutput = true, UseShellExecute = false })!;
        string head = p.StandardOutput.ReadToEnd().Trim();
        p.WaitForExit();
        return p.ExitCode == 0 && head.Length > 0 ? head : "unknown";
    }
    catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
    {
        return "unknown";
    }
}

static Dictionary<string, string> ParseArgs(string[] argv)
{
    Dictionary<string, string> parsed = new(StringComparer.Ordinal);
    for (int i = 0; i < argv.Length; i++)
    {
        if (argv[i].StartsWith("--", StringComparison.Ordinal))
        {
            string key = argv[i][2..];
            string value = i + 1 < argv.Length && !argv[i + 1].StartsWith("--", StringComparison.Ordinal) ? argv[++i] : "true";
            parsed[key] = value;
        }
        else if (!parsed.ContainsKey("command"))
        {
            parsed["command"] = argv[i];
        }
    }
    return parsed;
}

/// <summary>Every case set the tool knows; a new op family adds its class here.</summary>
file static class Registry
{
    public static IEnumerable<OpCase> Cases() => SmokeCases.Cases().Concat(AttentionCases.Cases()).Concat(LayoutCases.Cases()).Concat(ShuffleCases.Cases()).Concat(RotaryCases.Cases());
}
