using System.Text.Json;
using System.Text.Json.Schema;
using System.Text.Json.Serialization.Metadata;

namespace HartsyInference.BenchmarkRunner.Serialization;

/// <summary>Documentation schemas generated from the same contracts used by the strict validator.</summary>
public static class Schemas
{
    public static void Write(string directory)
    {
        Directory.CreateDirectory(directory);
        (string Name, JsonTypeInfo Type)[] contracts =
        [
            ("campaign-v1", BenchJson.Default.CampaignRecord), ("submission-v1", BenchJson.Default.SubmissionRecord),
            ("review-v1", BenchJson.Default.ReviewRecord), ("suite-v1", BenchJson.Default.SuiteDefinition),
        ];
        foreach ((string name, JsonTypeInfo type) in contracts)
        {
            System.Text.Json.Nodes.JsonNode schema = type.GetJsonSchemaAsNode();
            schema["$schema"] = "https://json-schema.org/draft/2020-12/schema";
            File.WriteAllText(Path.Combine(directory, name + ".schema.json"), schema.ToJsonString(new JsonSerializerOptions
            {
                WriteIndented = true, NewLine = "\n", TypeInfoResolver = BenchJson.Default,
            }) + "\n");
        }
    }
}
