using System.Globalization;
using HartsyInference.Cli.Infra;
using HartsyInference.Engine;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;

namespace HartsyInference.Cli.Dispatch;

/// <summary><c>hartsy duplex</c>: answers a recording of the user with a full-duplex speech-to-speech model.</summary>
internal static class DuplexDispatch
{
    /// <summary>Plays <c>input</c> to the model and returns the conversation as a WAV: stereo (user left, model right) with
    /// <c>stereo</c>, else the model's side. <c>card</c> is the character card; the sampling knobs fall back to the request's defaults.</summary>
    public static async Task<GeneratedArtifact> RunAsync(IInferenceEngine engine, ModelSpec spec, ParamState parameters,
        CancellationToken cancel)
    {
        string inputPath = parameters.GetStringOrNull("input") ?? throw new ArgumentException("A recording of the user is required.");
        AudioClip user = GenerationDispatch.LoadAudioClip(inputPath) ?? throw new FileNotFoundException($"Audio file not found: {inputPath}");
        DuplexSpeechRequest defaults = new();
        DuplexSpeechRequest request = new()
        {
            Card = parameters.GetStringOrNull("card") ?? "",
            Reference = GenerationDispatch.LoadAudioClip(parameters.GetStringOrNull("reference")),
            Guidance = parameters.GetFloatOrNull("guidance-scale") ?? defaults.Guidance,
            Temperature = parameters.GetFloatOrNull("temperature") ?? defaults.Temperature,
            TopK = parameters.GetIntOrNull("top-k") ?? defaults.TopK,
            RepetitionPenalty = !parameters.GetBool("no-dry", false),
            EopBias = parameters.GetFloatOrNull("eop-bias") ?? 0f,
            SpeakFirst = parameters.GetBool("speak-first", false),
            Seed = parameters.GetInt("seed", defaults.Seed),
        };
        double tailSeconds = parameters.GetDoubleOrNull("tail-seconds") ?? 0d;
        int tailFrames = (int)Math.Ceiling(Math.Max(0d, tailSeconds) * 24000 / 3200);

        DuplexResult result = await engine.Speech.RespondAsync(spec, request, user, tailFrames, cancel).ConfigureAwait(false);
        double seconds = result.ModelAudio.Length / (double)result.SampleRate;
        string summary = string.Create(CultureInfo.InvariantCulture,
            $"{seconds:F1}s, {result.Frames} frames, {result.MedianFrameSeconds * 1000:F0} ms median per frame ({result.MaxFrameSeconds * 1000:F0} ms max)");
        if (parameters.GetStringOrNull("transcript") is { Length: > 0 } transcriptPath)
            File.WriteAllText(transcriptPath, result.Text, new System.Text.UTF8Encoding(false));
        GeneratedArtifact artifact = new()
        {
            Kind = ArtifactKind.Audio,
            FileBytes = parameters.GetBool("stereo", false) ? result.ToStereoWav() : result.ToModelWav(),
            Extension = "wav",
            Text = result.Text.Length > 0 ? $"{summary}\n{result.Text}" : summary,
        };
        artifact.Meta["seconds"] = seconds.ToString("F1", CultureInfo.InvariantCulture);
        artifact.Meta["sample_rate"] = result.SampleRate.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["frames"] = result.Frames.ToString(CultureInfo.InvariantCulture);
        artifact.Meta["median_frame_ms"] = (result.MedianFrameSeconds * 1000).ToString("F1", CultureInfo.InvariantCulture);
        artifact.Meta["transcript"] = result.Text;
        return artifact;
    }
}
