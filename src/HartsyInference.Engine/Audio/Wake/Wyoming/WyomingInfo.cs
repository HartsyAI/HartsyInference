using System.Text.Json;

namespace HartsyInference.Engine.Audio.Wake.Wyoming;

/// <summary>Builds the <c>info</c> manifest Home Assistant reads to decide what this endpoint can do.
///
/// <para>Every service and every model/voice inside it must carry <c>name</c>, <c>attribution</c>,
/// <c>installed</c>, <c>description</c> and <c>version</c>, and every model/voice must carry <c>languages</c>:
/// Wyoming's decoder fills only its <c>Optional</c> fields in, so a missing required key makes the whole manifest
/// fail to parse and Home Assistant shows the service as unavailable with nothing useful logged. Unknown keys are
/// skipped, so being generous costs nothing and being terse breaks silently — which is why the keys are written
/// unconditionally here even when the value is null.</para></summary>
public static class WyomingInfo
{
    /// <summary>Serializes the manifest's <c>data</c> object.</summary>
    /// <param name="wakeAvailable">False omits the wake service even when models are configured, because no detector was wired and the words could never fire.</param>
    public static byte[] Build(WyomingOptions options, bool wakeAvailable)
    {
        ArgumentNullException.ThrowIfNull(options);
        return WyomingFrameCodec.BuildData(writer =>
        {
            WriteServices(writer, "asr", options, options.AsrModels, "models", WriteAsrModel);
            WriteServices(writer, "tts", options, options.TtsVoices, "voices", WriteTtsVoice);
            WriteServices(writer, "wake", options, wakeAvailable ? options.WakeModels : [], "models", WriteWakeModel);
            // Present-but-empty rather than absent: Wyoming's reader defaults them anyway, and an explicit empty
            // list is how a service says "I am not one of these" instead of "I am an older peer".
            writer.WriteStartArray("handle");
            writer.WriteEndArray();
            writer.WriteStartArray("intent");
            writer.WriteEndArray();
            writer.WriteStartArray("mic");
            writer.WriteEndArray();
            writer.WriteStartArray("snd");
            writer.WriteEndArray();
        });
    }

    private static void WriteServices(Utf8JsonWriter writer, string domain, WyomingOptions options,
        IReadOnlyList<WyomingArtifact> artifacts, string artifactKey, Action<Utf8JsonWriter, WyomingArtifact> writeArtifact)
    {
        writer.WriteStartArray(domain);
        if (artifacts.Count > 0)
        {
            writer.WriteStartObject();
            WriteArtifactHeader(writer, options.ProgramName, options.Attribution, options.ProgramDescription, options.ProgramVersion);
            writer.WriteStartArray(artifactKey);
            foreach (WyomingArtifact artifact in artifacts)
            {
                writer.WriteStartObject();
                writeArtifact(writer, artifact);
                writer.WriteEndObject();
            }
            writer.WriteEndArray();
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteAsrModel(Utf8JsonWriter writer, WyomingArtifact model)
    {
        WriteArtifactHeader(writer, model.Name, model.Attribution, model.Description, model.Version);
        WriteLanguages(writer, model.Languages);
    }

    private static void WriteWakeModel(Utf8JsonWriter writer, WyomingArtifact model)
    {
        WriteArtifactHeader(writer, model.Name, model.Attribution, model.Description, model.Version);
        WriteLanguages(writer, model.Languages);
        writer.WriteString("phrase", model.Phrase ?? model.Name);
    }

    private static void WriteTtsVoice(Utf8JsonWriter writer, WyomingArtifact voice)
    {
        WriteArtifactHeader(writer, voice.Name, voice.Attribution, voice.Description, voice.Version);
        WriteLanguages(writer, voice.Languages);
        if (voice.Speakers is null) return;
        writer.WriteStartArray("speakers");
        foreach (string speaker in voice.Speakers)
        {
            writer.WriteStartObject();
            writer.WriteString("name", speaker);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteArtifactHeader(Utf8JsonWriter writer, string name, WyomingAttribution attribution,
        string? description, string? version)
    {
        writer.WriteString("name", name);
        writer.WriteStartObject("attribution");
        writer.WriteString("name", attribution.Name);
        writer.WriteString("url", attribution.Url);
        writer.WriteEndObject();
        writer.WriteBoolean("installed", true);
        if (description is null) writer.WriteNull("description"); else writer.WriteString("description", description);
        if (version is null) writer.WriteNull("version"); else writer.WriteString("version", version);
    }

    private static void WriteLanguages(Utf8JsonWriter writer, IReadOnlyList<string> languages)
    {
        writer.WriteStartArray("languages");
        foreach (string language in languages) writer.WriteStringValue(language);
        writer.WriteEndArray();
    }
}
