using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HartsyInference.LLM.ChatTemplates;
using HartsyInference.LLM.Generation;
using HartsyInference.ModelAssets.Gguf;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.GenCapture;

/// <summary>Prepares an ordinary text workload using the production tokenizer and template without loading weights.</summary>
internal static class AttentionWorkloadFreeze
{
    /// <summary>Writes a candidate manifest; B0 must still prove token-limit completion and its paged execution path.</summary>
    public static int Run(Dictionary<string, string> options)
    {
        string model = Path.GetFullPath(options["model"]);
        string directory = Path.GetFullPath(options["out"]);
        if (Directory.Exists(directory)) throw new IOException("Workload directories are immutable; choose a fresh path.");
        using GgufLoader loader = new();
        loader.Load(model);
        Type production = typeof(GgufLanguageModel);
        ILlmTokenizer tokenizer = (ILlmTokenizer)production.GetMethod("BuildTokenizer", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [loader.Metadata])!;
        IChatTemplate template = (IChatTemplate)production.GetMethod("BuildTemplate", BindingFlags.NonPublic | BindingFlags.Static)!
            .Invoke(null, [loader.Metadata, tokenizer])!;
        string introduction = "Read the project notes below. Then write a detailed essay of at least two thousand words "
            + "explaining how a small engineering team should measure performance, check correctness, and review changes. "
            + "Discuss practical examples and tradeoffs in complete paragraphs.\n\nProject notes:\n";
        string paragraph = "The team maintains an inference service used by several applications. Each change needs a clear "
            + "owner, reproducible inputs, explicit numerical expectations, and measurements of the complete user request. "
            + "Engineers compare results on the same device, preserve the original data, and record failed experiments. "
            + "They schedule shared hardware carefully and review evidence before changing the next component.\n";
        string suffix = "\nNow write the essay. Begin with the team's measurement process and continue with its review process.\n";
        int Count(string content) => template.Encode(tokenizer, [ChatMessage.User(content)], true, false).Length;
        StringBuilder body = new(introduction);
        while (Count(body.ToString() + paragraph + suffix) < 8192) body.Append(paragraph);
        string prompt = body.ToString() + suffix;
        while (Count(prompt) < 8192)
        {
            body.Append(" detail");
            prompt = body.ToString() + suffix;
        }
        int[] ids = template.Encode(tokenizer, [ChatMessage.User(prompt)], true, false);
        if (ids.Length != 8192) throw new InvalidOperationException("Unable to reach the frozen token count exactly.");
        Directory.CreateDirectory(directory);
        string promptPath = Path.Combine(directory, "prompt.txt");
        File.WriteAllText(promptPath, prompt, new UTF8Encoding(false));
        using MemoryStream identity = new();
        using (Utf8JsonWriter json = new(identity))
        {
            json.WriteStartObject();
            foreach (string key in loader.Metadata.Keys.Where(key => key.StartsWith("tokenizer.", StringComparison.Ordinal))
                .Order(StringComparer.Ordinal))
            {
                loader.Metadata.TryGetValue(key, out object? value);
                json.WritePropertyName(key);
                WriteMetadata(json, value);
            }
            json.WriteEndObject();
        }
        string tokenizerPath = Path.Combine(directory, "embedded-tokenizer.json");
        File.WriteAllBytes(tokenizerPath, identity.ToArray());
        using FileStream manifest = new(Path.Combine(directory, "workload.json"), FileMode.CreateNew);
        using Utf8JsonWriter writer = new(manifest, new JsonWriterOptions { Indented = true });
        writer.WriteStartObject();
        writer.WriteNumber("schema", 1);
        writer.WriteString("family", "qwen");
        writer.WriteString("model_id", "Qwen/Qwen3-4B-GGUF");
        writer.WriteString("model_path", model);
        writer.WriteString("model_sha256", Hash(model));
        writer.WriteString("prompt_path", promptPath);
        writer.WriteString("prompt_sha256", Hash(promptPath));
        writer.WriteString("embedded_tokenizer_sha256", Hash(tokenizerPath));
        writer.WriteNumber("rendered_prompt_tokens", ids.Length);
        writer.WriteString("rendered_tokens_sha256", Convert.ToHexString(SHA256.HashData(
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(ids.AsSpan()))).ToLowerInvariant());
        writer.WriteString("status", "candidate; B0 natural 128-token completion and executed-path proof required");
        writer.WriteEndObject();
        return 0;
    }

    private static void WriteMetadata(Utf8JsonWriter writer, object? value)
    {
        switch (value)
        {
            case null: writer.WriteNullValue(); break;
            case string text: writer.WriteStringValue(text); break;
            case bool boolean: writer.WriteBooleanValue(boolean); break;
            case byte or ushort or uint or ulong:
                writer.WriteNumberValue(Convert.ToUInt64(value, System.Globalization.CultureInfo.InvariantCulture)); break;
            case sbyte or short or int or long:
                writer.WriteNumberValue(Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture)); break;
            case float single: writer.WriteNumberValue(single); break;
            case double number: writer.WriteNumberValue(number); break;
            case Array array:
                writer.WriteStartArray();
                foreach (object? item in array) WriteMetadata(writer, item);
                writer.WriteEndArray();
                break;
            default: throw new NotSupportedException("Unknown tokenizer metadata type: " + value.GetType().Name);
        }
    }

    private static string Hash(string path)
    {
        using FileStream file = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(file)).ToLowerInvariant();
    }
}
