using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.OpenVoice;
using HartsyInference.Audio.Models.Vits;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;

namespace HartsyInference.Engine.Audio;

/// <summary>OpenVoice V2 tone-color converter (myshell-ai/OpenVoiceV2) — a VITS posterior + flow + HiFi-GAN run as a
/// voice converter: it re-voices a source clip into a target speaker's tone color. The converter checkpoint
/// auto-downloads.
///
/// <para>Config is the Piper-high VITS shape (upsample product 256 hop, 22.05 kHz) with 256 gin channels for speaker
/// conditioning and 513 linear-spec bins. The pipeline carries its own reference encoder, so the source/target
/// speaker vectors are extracted internally from the two spectrograms.</para></summary>
internal static class OpenVoiceModel
{
    private const string Repo = "myshell-ai/OpenVoiceV2";
    private const string CheckpointFile = "converter/checkpoint.pth";
    private const int NFft = 1024;
    private const int Hop = 256;
    private const int SpecChannels = 513;

    internal static VcModelDescriptor Descriptor { get; } = new VcModelDescriptor
    {
        ManagesOwnWeights = true,
        InputSampleRate = 22_050,
        CacheKey = _ => "openvoice-v2",
        LoadAsync = async (_, cancel) =>
        {
            string checkpoint = await AudioModelCache.GetAsync(Repo, CheckpointFile, category: "clone", ct: cancel).ConfigureAwait(false);
            PytorchPickleLoader loader = new PytorchPickleLoader();
            loader.Load(checkpoint);

            VitsConfig config = VitsConfig.PiperHigh with { GinChannels = 256 };
            OpenVoicePipeline pipeline = new OpenVoicePipeline(config, SpecChannels, posteriorLayers: 16);
            pipeline.LoadWeights(loader.GetAllTensors());
            Logs.Info("[Audio][OpenVoice] Loaded myshell-ai/OpenVoiceV2 tone-color converter (22.05 kHz).");

            return new VcRunner(pipeline.SampleRate, (backend, source, target, _) =>
            {
                if (target is null || target.Length == 0)
                {
                    throw new InvalidOperationException(
                        "OpenVoice needs a target voice — supply a reference clip as the request's target.");
                }
                Tensor sourceSpec = LinearSpectrogram.Extract(source, NFft, Hop);
                Tensor targetSpec = LinearSpectrogram.Extract(target, NFft, Hop);
                try
                {
                    return pipeline.ConvertWithReferences(backend, sourceSpec, (int)sourceSpec.Shape[2], targetSpec, (int)targetSpec.Shape[2]);
                }
                finally
                {
                    sourceSpec.Dispose();
                    targetSpec.Dispose();
                }
            }, pipeline, loader);
        },
    };
}
