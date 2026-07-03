using System.Text;
using System.Text.RegularExpressions;
using HartsyInference.Audio.Io;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VibeVoice;

/// <summary>Multi-speaker prompt processor for VibeVoice. Takes a per-speaker script and a
/// list of reference-audio paths, builds the Qwen-tokenized prompt that VibeVoice's LM
/// expects, and returns the token-IDs + the <c>speech_input_mask</c> that tells the
/// pipeline which positions to overwrite with acoustic-VAE-encoded voice latents at
/// prefill time.
///
/// <para>Mirrors <c>vibevoice/processor/vibevoice_processor.py</c>. Prompt template:</para>
/// <code>
///  Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.
///  Voice input:
///  Speaker 0:&lt;|vision_start|&gt;&lt;|vision_pad|&gt;*N_0&lt;|vision_end|&gt;
///  Speaker 1:&lt;|vision_start|&gt;&lt;|vision_pad|&gt;*N_1&lt;|vision_end|&gt;
///  ...
///  Text input:
///  Speaker 0: Hello, welcome to the podcast.
///  Speaker 1: Thanks, glad to be here.
///  ...
///  Speech output:
/// &lt;|vision_start|&gt;
/// </code>
///
/// <para>The trailing <c>&lt;|vision_start|&gt;</c> is the cursor — generation begins from
/// there. <c>N_i = ceil(speaker_i_audio_samples / 3200)</c> is the count of
/// <c>&lt;|vision_pad|&gt;</c> sentinels needed to represent the speaker's voice prompt at
/// the acoustic VAE's 7.5 Hz frame rate.</para></summary>
public sealed class VibeVoiceProcessor
{
    private const string SystemPrompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.";
    private const string VoiceHeader = " Voice input:";
    private const string TextHeader = " Text input:";
    private const string OutputHeader = " Speech output:";
    private const float TargetDbFs = -25.0f;
    private const float TargetRmsEps = 1e-6f;

    private readonly VibeVoiceTokenizer _tokenizer;

    public VibeVoiceProcessor(VibeVoiceTokenizer tokenizer)
    {
        _tokenizer = tokenizer;
    }

    /// <summary>Parses a script and assembles all the inputs the pipeline needs to start
    /// generation. Each entry in <paramref name="lines"/> is one speaker turn — either a
    /// "Speaker N: text" formatted line, or a plain string that's assumed to be the next
    /// speaker in rotation (Speaker 0 on the first line, Speaker 1 on the second, etc.).
    ///
    /// <para><paramref name="voiceAudioPaths"/> is one 24 kHz mono WAV path per speaker.
    /// The processor reads each file, normalizes to -25 dB FS, and counts the resulting
    /// <c>N_i = ceil(samples / 3200)</c> latent slots.</para></summary>
    public PreparedPrompt Prepare(IReadOnlyList<string> lines, IReadOnlyList<string> voiceAudioPaths)
    {
        if (voiceAudioPaths is null || voiceAudioPaths.Count == 0)
            throw new ArgumentException("At least one voice prompt is required.", nameof(voiceAudioPaths));

        // 1. Load + normalize each voice prompt.
        VoicePrompt[] voices = new VoicePrompt[voiceAudioPaths.Count];
        for (int i = 0; i < voiceAudioPaths.Count; i++)
        {
            WavFile.DecodedAudio wav = WavFile.Read(voiceAudioPaths[i]);
            float[] mono = wav.Channels[0];
            if (wav.SampleRate != 24_000)
            {
                Resampler rs = Resampler.Create(wav.SampleRate, 24_000);
                mono = rs.Resample(mono);
            }
            NormalizeDbFsInPlace(mono);
            int latentCount = (int)Math.Ceiling(mono.Length / 3_200.0);
            if (latentCount <= 0) latentCount = 1;     // empty/very-short prompts still need a slot.
            voices[i] = new VoicePrompt(mono, latentCount);
        }

        // 2. Parse the script into (speakerIdx, text) turns.
        Turn[] turns = ParseScript(lines, voiceAudioPaths.Count);

        // 3. Assemble the token-ID stream, populating speechInputMask as we go.
        List<int> tokens = new(2048);
        List<bool> mask = new(2048);

        // System prompt.
        AppendText(tokens, mask, SystemPrompt + "\n");

        // Voice input section.
        AppendText(tokens, mask, VoiceHeader + "\n");
        for (int i = 0; i < voices.Length; i++)
        {
            AppendText(tokens, mask, $" Speaker {i}:");
            tokens.Add(VibeVoiceTokenizer.SpeechStartTokenId); mask.Add(false);
            for (int j = 0; j < voices[i].LatentCount; j++)
            {
                tokens.Add(VibeVoiceTokenizer.SpeechDiffusionTokenId);
                mask.Add(true);     // pipeline replaces this position's embedding with acoustic VAE output.
            }
            tokens.Add(VibeVoiceTokenizer.SpeechEndTokenId); mask.Add(false);
            AppendText(tokens, mask, "\n");
        }

        // Text input section. Upstream formats this as f" Speaker {id}:{text}\n" with NO
        // space after the colon (the parsed text already has its leading space stripped),
        // so " Speaker 0:Hello" — matching that exactly keeps the BPE tokens identical.
        AppendText(tokens, mask, TextHeader + "\n");
        foreach (Turn turn in turns)
            AppendText(tokens, mask, $" Speaker {turn.SpeakerIndex}:{turn.Text}\n");

        // Speech output cursor.
        AppendText(tokens, mask, OutputHeader + "\n");
        tokens.Add(VibeVoiceTokenizer.SpeechStartTokenId); mask.Add(false);

        return new PreparedPrompt
        {
            TokenIds = tokens.ToArray(),
            SpeechInputMask = mask.ToArray(),
            Voices = voices,
        };
    }

    /// <summary>Tokenizes <paramref name="text"/> and appends each resulting id to
    /// <paramref name="tokens"/> with a matching <c>false</c> in
    /// <paramref name="mask"/> — text positions are never voice-prompt positions.</summary>
    private void AppendText(List<int> tokens, List<bool> mask, string text)
    {
        int[] ids = _tokenizer.Encode(text);
        foreach (int id in ids)
        {
            tokens.Add(id);
            mask.Add(false);
        }
    }

    /// <summary>Parses <paramref name="lines"/> into <see cref="Turn"/> records. Recognizes
    /// "Speaker N: text" pattern; falls back to round-robin speaker assignment for plain
    /// lines that don't match.</summary>
    private static Turn[] ParseScript(IReadOnlyList<string> lines, int numSpeakers)
    {
        Regex pat = new(@"^\s*Speaker\s+(\d+)\s*:\s*(.*)$", RegexOptions.CultureInvariant);
        Turn[] result = new Turn[lines.Count];
        int rr = 0;
        for (int i = 0; i < lines.Count; i++)
        {
            Match m = pat.Match(lines[i]);
            if (m.Success)
            {
                int spk = int.Parse(m.Groups[1].Value);
                if (spk >= numSpeakers) spk %= numSpeakers;
                result[i] = new Turn(spk, m.Groups[2].Value);
            }
            else
            {
                result[i] = new Turn(rr % numSpeakers, lines[i]);
                rr++;
            }
        }
        return result;
    }

    /// <summary>In-place normalize a PCM buffer to <see cref="TargetDbFs"/> dBFS.</summary>
    private static void NormalizeDbFsInPlace(float[] pcm)
    {
        if (pcm.Length == 0) return;
        double sumSq = 0d;
        for (int i = 0; i < pcm.Length; i++) sumSq += (double)pcm[i] * pcm[i];
        float rms = (float)Math.Sqrt(sumSq / pcm.Length);
        float targetRms = MathF.Pow(10f, TargetDbFs / 20f);
        float gain = targetRms / (rms + TargetRmsEps);
        for (int i = 0; i < pcm.Length; i++) pcm[i] *= gain;
    }

    /// <summary>One speaker turn in the script.</summary>
    private readonly record struct Turn(int SpeakerIndex, string Text);

    /// <summary>One speaker's voice prompt — the PCM bytes plus the latent count that the
    /// prompt template needs to know up front.</summary>
    public sealed record VoicePrompt(float[] Pcm, int LatentCount);

    /// <summary>Output of <see cref="Prepare"/>: everything the pipeline needs to start the
    /// AR loop.</summary>
    public sealed class PreparedPrompt
    {
        /// <summary>Full token-ID stream for the LM.</summary>
        public required int[] TokenIds { get; init; }

        /// <summary>Parallel array — <c>true</c> at positions where the pipeline should
        /// replace the embedded token with the acoustic-VAE latent for the matching voice
        /// prompt. The order is: voice 0's <c>N_0</c> slots, then voice 1's <c>N_1</c>
        /// slots, then voice 2's, and so on.</summary>
        public required bool[] SpeechInputMask { get; init; }

        /// <summary>Per-speaker reference audio (normalized to -25 dB FS) plus the latent
        /// count that should be encoded for the prefill.</summary>
        public required VoicePrompt[] Voices { get; init; }
    }
}
