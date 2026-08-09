using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;

namespace HartsyInference.Engine.Audio;

/// <summary>MeloTTS English-v3 (VITS + multi-stream text encoder) at 44.1 kHz. Self-contained: the engine facade
/// bundles the CMUdict g2p (phoneme + tone + language streams) and the bert-base-uncased prosody front-end, so no
/// external phonemizer wiring is needed. Not zero-shot — no voice reference required.</summary>
internal static class MeloTtsModel
{
    /// <summary>The MeloTTS descriptor.</summary>
    internal static TtsModelDescriptor Descriptor { get; } = new TtsModelDescriptor
    {
        ResolveRepo = _ => "myshell-ai/MeloTTS-English-v3",
        LoadAsync = async (_, _, cancel) =>
        {
            MeloTts melo = await MeloTts.LoadAsync(ct: cancel).ConfigureAwait(false);
            Logs.Info("[Audio][MeloTTS] Loaded myshell-ai/MeloTTS-English-v3 (VITS + CMUdict g2p + prosody BERT, 44.1 kHz).");
            return new TtsRunner(melo.SampleRate,
                // MeloTTS speaker slots come from the checkpoint's spk2id (EN-US 0, EN-BR 1, EN_INDIA 2,
                // EN-AU 3, EN-Default 4). VITS length_scale is the inverse of speed.
                (backend, job) => melo.SynthesizeText(backend, job.Text,
                    speakerId: job.SpeakerId ?? 0,
                    lengthScale: job.Speed is > 0 ? (float)(1.0 / job.Speed.Value) : null,
                    seed: job.Seed),
                melo);
        },
    };
}
