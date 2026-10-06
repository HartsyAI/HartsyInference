using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Engine.Audio;

/// <summary>ControlFoley (Xiaomi, CC-BY-NC-4.0) text-to-audio: DFN5B CLIP text features, a flow-matching DiT, then the VAE and
/// BigVGAN v2 decoder at 44.1 kHz. The released <c>controlfoley.pth</c> is 11 GB of float32; a sibling
/// <c>controlfoley_bf16.safetensors</c> (see <c>tools/controlfoley/convert_network_bf16.py</c>) is preferred when present
/// and needs half the memory. Video and reference-audio conditioning are not wired yet.</summary>
internal static class ControlFoleyMusicModel
{
    private const string Repo = "YJX-Xiaomi/ControlFoley";
    private const string VocoderRepo = "nvidia/bigvgan_v2_44khz_128band_512x";
    private const string ClipRepo = "apple/DFN5B-CLIP-ViT-H-14-384";
    private const string Bf16Name = "controlfoley_bf16.safetensors";

    internal static MusicModelDescriptor Descriptor { get; } = new MusicModelDescriptor
    {
        ManagesOwnWeights = true,
        CacheKey = _ => Repo,
        ResolveFiles = (_, _) => Task.FromResult<IReadOnlyList<AudioModelFile>>(
        [
            new AudioModelFile("ext_weights/v1-44.pth"),
            new AudioModelFile("bigvgan_generator.pt", Repo: VocoderRepo),
            new AudioModelFile("open_clip_pytorch_model.bin", Repo: ClipRepo),
            new AudioModelFile("weights/controlfoley.pth"),
        ]),
        LoadAsync = LoadAsync,
    };

    private static async Task<IMusicRunner> LoadAsync(MusicLoadContext context, AudioModelSelector selector, CancellationToken cancel)
    {
        string networkPath = await AudioModelCache.GetAsync(Repo, "weights/controlfoley.pth", "music", ct: cancel).ConfigureAwait(false);
        string vaePath = await AudioModelCache.GetAsync(Repo, "ext_weights/v1-44.pth", "music", ct: cancel).ConfigureAwait(false);
        string vocoderPath = await AudioModelCache.GetAsync(VocoderRepo, "bigvgan_generator.pt", "music", ct: cancel).ConfigureAwait(false);
        string clipPath = await AudioModelCache.GetAsync(ClipRepo, "open_clip_pytorch_model.bin", "music", ct: cancel).ConfigureAwait(false);
        string bf16 = Path.Combine(Path.GetDirectoryName(networkPath)!, Bf16Name);
        if (File.Exists(bf16))
        {
            networkPath = bf16;
        }

        List<IDisposable> owned = [];
        try
        {
            (IReadOnlyDictionary<string, Tensor> networkWeights, IDisposable networkLoader) = AudioCheckpoints.LoadFile(networkPath);
            owned.Add(networkLoader);
            ControlFoleyNetwork network = new(ControlFoleyNetworkConfig.Large44k);
            owned.Add(network);
            network.LoadWeights(networkWeights);

            (IReadOnlyDictionary<string, Tensor> clipWeights, IDisposable clipLoader) = AudioCheckpoints.LoadFile(clipPath);
            owned.Add(clipLoader);
            ControlFoleyClip clip = new(ControlFoleyClipConfig.Dfn5bViTH14);
            clip.LoadWeights(clipWeights);

            ControlFoleyAudioDecoder decoder = new();
            owned.Add(decoder);
            decoder.LoadFromFiles(vaePath, vocoderPath);

            ControlFoleyPipeline pipeline = new(network, clip, decoder);
            Logs.Info($"[Audio][ControlFoley] Loaded (network {Path.GetFileName(networkPath)}, 44.1 kHz mono, text-to-audio).");

            MusicAudio Synth(IBackend device, MusicRequest request, CancellationToken ct)
            {
                ControlFoleyPipeline.Request run = new()
                {
                    Prompt = request.Prompt,
                    DurationSeconds = request.Duration,
                    Seed = request.Seed,
                    Steps = request.InferSteps ?? 25,
                    CfgStrength = (float)(request.CfgScale ?? 4.5),
                };
                return MusicAudio.Mono(pipeline.Generate(device, run, ct));
            }

            return new MusicRunner(pipeline.SampleRate, Synth, [.. owned]);
        }
        catch
        {
            foreach (IDisposable d in owned)
            {
                d.Dispose();
            }

            throw;
        }
    }
}
