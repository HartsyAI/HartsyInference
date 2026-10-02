using System.Text;

namespace HartsyInference.Audio.Models.Auk;

/// <summary>English instruction templates of the AuK cookbook, keyed by <see cref="AukTask"/>; a caller may always bypass them with a verbatim instruction.</summary>
public static class AukTemplates
{
    private static readonly Dictionary<AukTask, string> Templates = new()
    {
        [AukTask.ZeroShotTts] = "Say the following with the same voice: \"{text}\"",
        [AukTask.InstructTts] = "Generate speech based on the following description: \"{description}\". The content to speak is: \"{text}\".",
        [AukTask.InstructTtsCookbookForm] = "Based on the following description: \"{description}\", generate speech content \"{text}\".",
        [AukTask.ReplaceText] = "Replace '{original}' with '{new}'.",
        [AukTask.InsertBefore] = "Add '{text}' before '{anchor}'.",
        [AukTask.InsertAfter] = "Add '{text}' after '{anchor}'.",
        [AukTask.RemoveText] = "Remove '{text}'.",
        [AukTask.RemoveTextBefore] = "Remove '{text}' before '{anchor}'.",
        [AukTask.RemoveTextAfter] = "Remove '{text}' after '{anchor}'.",
        [AukTask.LyricChange] = "Change \"{original}\" to \"{new}\" in the vocal recording.",
        [AukTask.PitchRaise] = "Raise the pitch by {semitones} semitones.",
        [AukTask.PitchLower] = "Lower the pitch by {semitones} semitones.",
        [AukTask.Speed] = "Adjust the speech speed to {factor}x.",
        [AukTask.VolumeIncrease] = "Increase the volume by {db} dB.",
        [AukTask.VolumeDecrease] = "Decrease the volume by {db} dB.",
        [AukTask.Emotion] = "Change the emotion to {emotion}.",
        [AukTask.Timbre] = "Keep the spoken content unchanged and change the timbre to: \"{description}\".",
        [AukTask.DeAccent] = "Remove the regional accent while preserving the speaker's voice and content.",
        [AukTask.NonverbalRemove] = "Remove all {sound} from the audio.",
        [AukTask.NonverbalAdd] = "Add a {sound} at the {position} of the speech.",
        [AukTask.WhisperConvert] = "Convert this speech into a soft whisper while preserving the speaker and content.",
        [AukTask.WhisperToNormal] = "Convert this whispered speech into a normal speaking voice while preserving the speaker and content.",
        [AukTask.Denoise] = "Remove only the background noise, preserve everything else, and output audio of the same length.",
        [AukTask.Dereverb] = "Remove only the room reverberation, preserve everything else, and output audio of the same length.",
        [AukTask.EnhanceSpeech] = "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.",
        [AukTask.QualityRestoration] = "Repair the {defect} and restore natural, clear speech.",
        [AukTask.SpeechSeparation] = "Keep only the {ordinal} speaker to start talking and remove all other speakers.",
        [AukTask.MusicSeparationSinging] = "Keep only the singing voice and remove everything else.",
        [AukTask.MusicSeparationAllVoices] = "Keep all human voices, including speech and singing, and remove everything else.",
        [AukTask.TargetSpeakerExtraction] = "Keep only the speaker who says \"{content}\" and remove all other speakers.",
    };

    /// <summary>The raw template of <paramref name="task"/>, with <c>{name}</c> placeholders.</summary>
    public static string Get(AukTask task) => Templates.TryGetValue(task, out string? template)
        ? template
        : throw new ArgumentOutOfRangeException(nameof(task), task, "No template is registered for this task.");

    /// <summary>Whether the task conditions on a source or reference audio clip; only instruct TTS runs without one.</summary>
    public static bool RequiresAudio(AukTask task) => task is not (AukTask.InstructTts or AukTask.InstructTtsCookbookForm);

    /// <summary>Placeholder names of the template in order of first appearance.</summary>
    public static IReadOnlyList<string> Placeholders(AukTask task)
    {
        List<string> names = [];
        string template = Get(task);
        int i = 0;
        while ((i = template.IndexOf('{', i)) >= 0)
        {
            int end = template.IndexOf('}', i);
            string name = template[(i + 1)..end];
            if (!names.Contains(name)) names.Add(name);
            i = end + 1;
        }
        return names;
    }

    /// <summary>Substitutes <paramref name="values"/> for the placeholders in <see cref="Placeholders"/> order, in one pass so a value is never re-expanded.</summary>
    public static string Format(AukTask task, params string[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        List<string> names = [.. Placeholders(task)];
        if (values.Length != names.Count)
        {
            throw new ArgumentException($"{task} takes {names.Count} value(s) ({string.Join(", ", names)}); got {values.Length}.", nameof(values));
        }
        string template = Get(task);
        StringBuilder sb = new(template.Length + 32);
        int i = 0;
        while (i < template.Length)
        {
            int open = template.IndexOf('{', i);
            if (open < 0)
            {
                sb.Append(template, i, template.Length - i);
                break;
            }
            sb.Append(template, i, open - i);
            int close = template.IndexOf('}', open);
            string name = template[(open + 1)..close];
            sb.Append(values[names.IndexOf(name)]);
            i = close + 1;
        }
        return sb.ToString();
    }
}
