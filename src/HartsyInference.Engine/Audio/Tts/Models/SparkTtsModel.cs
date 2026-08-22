using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>Spark-TTS-0.5B (SparkAudio/Spark-TTS-0.5B) — a Qwen2.5-0.5B LM emits the unified global + semantic BiCodec token stream, decoded to 16 kHz. Runs the controllable mode: text plus a coarse style (gender from the voice field, speed bucketed from the rate multiplier). Zero-shot cloning needs the BiCodec encoder side, which is not built, so a supplied reference clip is ignored.</summary>
internal static class SparkTtsModel
{
    private const string Repo = "SparkAudio/Spark-TTS-0.5B";

    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => Repo,
        LoadAsync = async (_, _, cancel) =>
        {
            SparkTtsPipeline pipeline = await SparkTtsPipeline.LoadAsync(Repo, ct: cancel).ConfigureAwait(false);
            Logs.Info("[Audio][Spark-TTS] Loaded SparkAudio/Spark-TTS-0.5B (Qwen2.5-0.5B LM + BiCodec, 16 kHz, controllable).");
            return new TtsRunner(pipeline.SampleRate, (backend, job) =>
            {
                string gender = string.Equals(job.Voice, "male", StringComparison.OrdinalIgnoreCase) ? "male" : "female";
                return pipeline.SynthesizeControllable(backend, job.Text, gender, pitch: "moderate",
                    speed: SpeedLevel(job.Speed), seed: job.Seed);
            }, pipeline);
        },
    };

    /// <summary>Maps a speaking-rate multiplier to Spark's five coarse speed buckets; null → moderate.</summary>
    private static string SpeedLevel(double? speed)
    {
        if (speed is null)
        {
            return "moderate";
        }
        double value = speed.Value;
        return value < 0.7 ? "very_low" : value < 0.9 ? "low" : value <= 1.15 ? "moderate" : value <= 1.4 ? "high" : "very_high";
    }
}
