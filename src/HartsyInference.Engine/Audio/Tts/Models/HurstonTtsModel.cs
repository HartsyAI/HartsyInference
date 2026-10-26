using HartsyInference.Audio.Frontends;
using HartsyInference.Audio.Models.Orpheus;

namespace HartsyInference.Engine.Audio;

/// <summary>Hurston-1 (destined/hurston-1, Apache-2.0): accented English TTS steered by a voice description, a Maya1
/// LoRA fine-tune on the Orpheus stack (Llama-3.2-3B + SNAC 24 kHz). Shares <see cref="OrpheusTtsModel.Build"/>; only
/// the config (<see cref="OrpheusConfig.Hurston1"/>), the backbone repo and the <c>&lt;description=…&gt;</c> prompt differ.
/// The backbone is the single-file repack made by <c>tools/repack/repack.py repack hurston-1</c>.</summary>
internal static class HurstonTtsModel
{
    private const string BackboneRepo = "Hartsy/hurston-1-safetensors";
    private const string DefaultDescription = "Female, adult, American accent, clear conversational voice, natural pacing";

    internal static TtsModelDescriptor Descriptor { get; } = OrpheusTtsModel.Build(BackboneRepo, OrpheusConfig.Hurston1,
        "Hurston-1", job => AudioTextFrontend.HurstonText(
            string.IsNullOrWhiteSpace(job.Instruction) ? DefaultDescription : job.Instruction, job.Text));
}
