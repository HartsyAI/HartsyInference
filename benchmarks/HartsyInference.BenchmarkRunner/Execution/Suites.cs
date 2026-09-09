using System.Text.RegularExpressions;
using HartsyInference.BenchmarkRunner.Contracts;
using HartsyInference.BenchmarkRunner.Serialization;

namespace HartsyInference.BenchmarkRunner.Execution;

/// <summary>Loads bundled data manifests; adding a model workload does not require changing runner dispatch.</summary>
public static class Suites
{
    public static IEnumerable<string> AvailableIds => Directory.GetFiles(Path.Combine(AppContext.BaseDirectory, "suites"), "*.json")
        .Select(Path.GetFileNameWithoutExtension).OfType<string>().Order(StringComparer.Ordinal);

    public static string PathFor(string id)
    {
        if (!Regex.IsMatch(id, "^[a-z0-9][a-z0-9-]{0,63}$")) throw new ArgumentException("Invalid suite identifier.");
        string path = Path.Combine(AppContext.BaseDirectory, "suites", id + ".json");
        if (!File.Exists(path)) throw new FileNotFoundException("Unknown suite; run list to see bundled workloads.");
        return path;
    }

    public static SuiteDefinition Load(string id)
    {
        SuiteDefinition suite = BenchJson.Read(PathFor(id), BenchJson.Default.SuiteDefinition);
        if (suite.Id != id || suite.SchemaVersion != 1 || suite.Cases.Length is < 1 or > 16
            || suite.Sessions is < 1 or > 3 || suite.Warmups != 2 || suite.Publishable && suite.Sessions != 3
            || suite.Cases.Select(c => c.Id).Distinct(StringComparer.Ordinal).Count() != suite.Cases.Length)
            throw new InvalidDataException("Invalid bundled suite protocol.");
        foreach (CaseDefinition definition in suite.Cases)
        {
            if (!Regex.IsMatch(definition.Id, "^[a-z0-9][a-z0-9-]{0,63}$") || definition.Inputs.Length != 5
                || definition.MaxTokens is < 1 or > 4096 || definition.PrefixRepeats is < 0 or > 1024
                || definition.Width is < 1 or > 4096 || definition.Height is < 1 or > 4096
                || definition.TimeoutSeconds is < 1 or > 86400 || !Hashes.IsHash(definition.Asset.Sha256)
                || !Regex.IsMatch(definition.Asset.Revision, "^[0-9a-f]{40}$") || definition.Asset.Bytes <= 0)
                throw new InvalidDataException("Invalid bundled case definition.");
        }
        return suite;
    }
}
