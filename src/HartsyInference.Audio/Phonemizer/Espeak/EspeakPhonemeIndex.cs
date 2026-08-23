using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Logging;

namespace HartsyInference.Audio.Phonemizer.Espeak;

/// <summary>Holds espeak-ng's compiled <c>phonindex</c> file: a flat array of 16-bit program words, indexed by each phoneme's <see cref="EspeakPhoneme.Program"/> and executed by <see cref="EspeakPhonemeInterpreter"/> for context-dependent allophone selection and IPA name output; the big <c>phondata</c> spectrum/wave file is NOT needed for phonemization (audio only).</summary>
internal sealed class EspeakPhonemeIndex
{
    /// <summary>The program words, indexed directly by <see cref="EspeakPhoneme.Program"/>.</summary>
    public ushort[] Program { get; }

    private EspeakPhonemeIndex(ushort[] program) => Program = program;

    /// <summary>Loads <c>phonindex</c> from <paramref name="path"/> (little-endian 16-bit words).</summary>
    public static EspeakPhonemeIndex Load(string path)
    {
        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            ushort[] words = new ushort[bytes.Length / 2];
            for (int i = 0; i < words.Length; i++)
                words[i] = (ushort)(bytes[i * 2] | (bytes[i * 2 + 1] << 8));
            return new EspeakPhonemeIndex(words);
        }
        catch (Exception ex) when (ex is not HartsyInferenceException)
        {
            Logs.Error($"Failed to load espeak phonindex '{path}': {ex.Message}");
            throw new HartsyInferenceException($"Failed to load espeak phonindex '{path}'.", ex);
        }
    }
}
