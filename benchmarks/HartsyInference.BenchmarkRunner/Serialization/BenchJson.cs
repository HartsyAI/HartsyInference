using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using HartsyInference.BenchmarkRunner.Contracts;

namespace HartsyInference.BenchmarkRunner.Serialization;
/// <summary>Strict, source-generated serialization shared by local execution and trusted validation.</summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, WriteIndented = true, NewLine = "\n",
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow)]
[JsonSerializable(typeof(AssetDefinition))]
[JsonSerializable(typeof(CaseDefinition))]
[JsonSerializable(typeof(SuiteDefinition))]
[JsonSerializable(typeof(DeviceRecord))]
[JsonSerializable(typeof(EnvironmentRecord))]
[JsonSerializable(typeof(Measurement))]
[JsonSerializable(typeof(SessionRecord))]
[JsonSerializable(typeof(CampaignRecord))]
[JsonSerializable(typeof(SubmissionRecord))]
[JsonSerializable(typeof(ReviewRecord))]
[JsonSerializable(typeof(ValidationReport))]
[JsonSerializable(typeof(SummaryRow))]
[JsonSerializable(typeof(DeviceRecord[]))]
[JsonSerializable(typeof(SummaryRow[]))]
[JsonSerializable(typeof(SortedDictionary<string, string>))]
public partial class BenchJson : JsonSerializerContext
{
    public static T Read<T>(string path, JsonTypeInfo<T> type, long limit = 2 * 1024 * 1024)
    {
        FileInfo file = new(path);
        if (!file.Exists || file.Length > limit)
            throw new InvalidDataException("Missing or oversized JSON record.");
        using JsonDocument document = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { MaxDepth = 32 });
        RejectDuplicates(document.RootElement);
        return document.RootElement.Deserialize(type) ?? throw new InvalidDataException("Null record.");
    }

    public static void Write<T>(string path, T value, JsonTypeInfo<T> type)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        string temporary = path + ".tmp";
        File.WriteAllText(temporary, JsonSerializer.Serialize(value, type));
        File.Move(temporary, path, true);
    }

    private static void RejectDuplicates(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            HashSet<string> names = new(StringComparer.OrdinalIgnoreCase);
            foreach (JsonProperty property in value.EnumerateObject())
            {
                if (!names.Add(property.Name))
                    throw new InvalidDataException("Duplicate JSON property.");
                RejectDuplicates(property.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (JsonElement element in value.EnumerateArray())
                RejectDuplicates(element);
    }
}
