using HartsyInference.Audio.Cache;
using HartsyInference.Audio.Models.ControlFoley;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.Engine.Requests;
using HartsyInference.Video.Encoding;

namespace HartsyInference.Engine.Audio;

/// <summary>ControlFoley (Xiaomi, CC-BY-NC-4.0) sound effects at 44.1 kHz: text, video-scored (<see cref="MusicRequest.Video"/>) or
/// following a reference clip. The video and reference encoders download on first use. A sibling <c>controlfoley_bf16.safetensors</c>
/// (<c>tools/controlfoley/convert_network_bf16.py</c>) is preferred over the 11 GB float32 <c>controlfoley.pth</c>.</summary>
internal static class ControlFoleyMusicModel
{
    private const string Repo = "YJX-Xiaomi/ControlFoley";
    private const string VocoderRepo = "nvidia/bigvgan_v2_44khz_128band_512x";
    private const string ClipRepo = "apple/DFN5B-CLIP-ViT-H-14-384";
    private const string Bf16Name = "controlfoley_bf16.safetensors";
    private const string SynchformerFile = "ext_weights/synchformer_state_dict.pth";
    private const string CavMaeFile = "ext_weights/cav_mae_st.pth";
    private const string ClapFile = "ext_weights/music_speech_audioset_epoch_15_esc_89.98.pt";
    private const string StyleRepo = "facebook/musicgen-style";
    private const string StyleFile = "state_dict.bin";
    private const string StyleHead = "condition_provider.conditioners.self_wav.";
    private const string MertRepo = "m-a-p/MERT-v1-95M";
    private const string MertFile = "pytorch_model.bin";

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

            ControlFoleyPipeline textPipeline = new(network, clip, decoder);
            object encoderLock = new();
            (ControlFoleySynchformer Synchformer, ControlFoleyCavMae CavMae)? video = null;
            (ControlFoleyClap Clap, ControlFoleyStyleEncoder Style)? reference = null;
            Logs.Info($"[Audio][ControlFoley] Loaded (network {Path.GetFileName(networkPath)}, 44.1 kHz mono).");

            // Each encoder set is fetched and loaded the first time a request needs it, then kept for the runner's life.
            ControlFoleyPipeline PipelineFor(bool needVideo, bool needReference, CancellationToken ct)
            {
                if (!needVideo && !needReference)
                {
                    return textPipeline;
                }

                lock (encoderLock)
                {
                    if (needVideo && video is null)
                    {
                        video = LoadVideoEncoders(owned, ct);
                    }

                    if (needReference && reference is null)
                    {
                        reference = LoadReferenceEncoders(owned, ct);
                    }

                    return new ControlFoleyPipeline(network, clip, decoder,
                        needVideo ? video!.Value.Synchformer : null, needVideo ? video!.Value.CavMae : null,
                        needReference ? reference!.Value.Clap : null, needReference ? reference!.Value.Style : null);
                }
            }

            MusicAudio Synth(IBackend device, MusicRequest request, CancellationToken ct)
            {
                ControlFoleyRawVideo? raw = request.Video is null ? null : DecodeVideo(request.Video, request.Duration, ct);
                ControlFoleyPipeline.ReferenceAudio? clipReference = ReadReference(request);
                if (raw is null && clipReference is null && string.IsNullOrWhiteSpace(request.Prompt))
                {
                    throw new ArgumentException("ControlFoley needs a prompt, a video or a usable reference clip.", nameof(request));
                }

                ControlFoleyPipeline.Request run = new()
                {
                    Prompt = request.Prompt,
                    NegativePrompt = request.NegativePrompt,
                    DurationSeconds = request.Duration,
                    Seed = request.Seed,
                    Steps = request.InferSteps ?? 25,
                    CfgStrength = (float)(request.CfgScale ?? 4.5),
                    Video = raw,
                    MaskAwayClip = request.MaskAwayClip,
                    Reference = clipReference,
                };
                return MusicAudio.Mono(PipelineFor(raw is not null, clipReference is not null, ct).Generate(device, run, ct));
            }

            return new MusicRunner(textPipeline.SampleRate, Synth, new OwnedDisposables(owned));
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

    private static (ControlFoleySynchformer, ControlFoleyCavMae) LoadVideoEncoders(List<IDisposable> owned, CancellationToken ct)
    {
        List<IDisposable> fresh = [];
        try
        {
            (ControlFoleySynchformer, ControlFoleyCavMae) encoders = LoadVideoEncodersInto(fresh, ct);
            lock (owned)
            {
                owned.AddRange(fresh);
            }

            return encoders;
        }
        catch
        {
            DisposeAll(fresh);
            throw;
        }
    }

    private static (ControlFoleySynchformer, ControlFoleyCavMae) LoadVideoEncodersInto(List<IDisposable> owned, CancellationToken ct)
    {
        string syncPath = AudioModelCache.GetAsync(Repo, SynchformerFile, "music", ct: ct).GetAwaiter().GetResult();
        string cavPath = AudioModelCache.GetAsync(Repo, CavMaeFile, "music", ct: ct).GetAwaiter().GetResult();
        (IReadOnlyDictionary<string, Tensor> syncWeights, IDisposable syncLoader) = AudioCheckpoints.LoadFile(syncPath);
        owned.Add(syncLoader);
        ControlFoleySynchformer synchformer = new(ControlFoleySynchformerConfig.Released);
        owned.Add(synchformer);
        synchformer.LoadWeights(syncWeights);
        (IReadOnlyDictionary<string, Tensor> cavWeights, IDisposable cavLoader) = AudioCheckpoints.LoadFile(cavPath);
        owned.Add(cavLoader);
        ControlFoleyCavMae cavMae = new(ControlFoleyCavMaeConfig.Released);
        owned.Add(cavMae);
        cavMae.LoadWeights(cavWeights);
        return (synchformer, cavMae);
    }

    /// <summary>CLAP audio tower plus the MusicGen-Style and MERT timbre encoder, with the names <c>tools/controlfoley</c> verifies.</summary>
    private static (ControlFoleyClap, ControlFoleyStyleEncoder) LoadReferenceEncoders(List<IDisposable> owned, CancellationToken ct)
    {
        List<IDisposable> fresh = [];
        try
        {
            (ControlFoleyClap, ControlFoleyStyleEncoder) encoders = LoadReferenceEncodersInto(fresh, ct);
            lock (owned)
            {
                owned.AddRange(fresh);
            }

            return encoders;
        }
        catch
        {
            DisposeAll(fresh);
            throw;
        }
    }

    private static void DisposeAll(List<IDisposable> items)
    {
        foreach (IDisposable item in items)
        {
            item.Dispose();
        }
    }

    private static (ControlFoleyClap, ControlFoleyStyleEncoder) LoadReferenceEncodersInto(List<IDisposable> owned, CancellationToken ct)
    {
        string clapPath = AudioModelCache.GetAsync(Repo, ClapFile, "music", ct: ct).GetAwaiter().GetResult();
        string stylePath = AudioModelCache.GetAsync(StyleRepo, StyleFile, "music", ct: ct).GetAwaiter().GetResult();
        string mertPath = AudioModelCache.GetAsync(MertRepo, MertFile, "music", ct: ct).GetAwaiter().GetResult();
        (IReadOnlyDictionary<string, Tensor> clapAll, IDisposable clapLoader) = AudioCheckpoints.LoadFile(clapPath);
        owned.Add(clapLoader);
        Dictionary<string, Tensor> clapWeights = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in clapAll)
        {
            string name = entry.Key.StartsWith("module.", StringComparison.Ordinal) ? entry.Key["module.".Length..] : entry.Key;
            if (name.StartsWith("audio_branch.", StringComparison.Ordinal) || name.StartsWith("audio_projection.", StringComparison.Ordinal))
            {
                clapWeights[name] = entry.Value;
            }
        }

        ControlFoleyClap clap = new(ControlFoleyClapConfig.HtsatBase);
        clap.LoadWeights(clapWeights);

        (IReadOnlyDictionary<string, Tensor> styleAll, IDisposable styleLoader) = AudioCheckpoints.LoadFile(stylePath);
        owned.Add(styleLoader);
        (IReadOnlyDictionary<string, Tensor> mertAll, IDisposable mertLoader) = AudioCheckpoints.LoadFile(mertPath);
        owned.Add(mertLoader);
        Dictionary<string, Tensor> styleWeights = new(StringComparer.Ordinal);
        foreach (KeyValuePair<string, Tensor> entry in styleAll)
        {
            if (!entry.Key.StartsWith(StyleHead, StringComparison.Ordinal))
            {
                continue;
            }

            string name = entry.Key[StyleHead.Length..];
            if (name.Contains("num_batches_tracked", StringComparison.Ordinal) || name.EndsWith("inited", StringComparison.Ordinal)
                || name.Contains("cluster_size", StringComparison.Ordinal) || name.Contains("embed_avg", StringComparison.Ordinal))
            {
                continue;
            }

            styleWeights["style." + name] = entry.Value;
        }

        foreach (KeyValuePair<string, Tensor> entry in mertAll)
        {
            if (entry.Key != "masked_spec_embed")
            {
                styleWeights["mert." + entry.Key] = entry.Value;
            }
        }

        ControlFoleyStyleEncoder style = new(ControlFoleyStyleConfig.MusicGenStyle);
        owned.Add(style);
        style.LoadWeights(styleWeights);
        return (clap, style);
}

    /// <summary>The reference clip as mono samples at its own rate, cut to the requested span; null for none.</summary>
    private static ControlFoleyPipeline.ReferenceAudio? ReadReference(MusicRequest request)
    {
        if (request.ReferenceAudio is null)
        {
            return null;
        }

        AudioBuffer buffer = AudioClipCodec.DecodeNative(request.ReferenceAudio);
        if (buffer.IsEmpty)
        {
            return null;
        }

        float[] mono = buffer.ToMono();
        int start = Math.Clamp((int)(request.ReferenceStartSeconds * buffer.SampleRate), 0, mono.Length);
        int end = Math.Clamp((int)(request.ReferenceEndSeconds * buffer.SampleRate), start, mono.Length);
        return end > start ? new ControlFoleyPipeline.ReferenceAudio(mono[start..end], buffer.SampleRate) : null;
    }

    /// <summary>Decodes the clip at its native size and rate, up to the requested duration, as the official PyAV loader does.</summary>
    private static ControlFoleyRawVideo DecodeVideo(VideoClip clip, double duration, CancellationToken cancel)
    {
        // One extra second keeps the frame at the duration boundary; the preprocessor drops frames past the duration.
        FfmpegProcessDecoder.Result decoded = new FfmpegProcessDecoder()
            .DecodeAsync(clip.Data, clip.Format, maxFrames: null, scaleWidth: null, scaleHeight: null, cancel, maxSeconds: duration + 1.0)
            .GetAwaiter().GetResult();
        return ControlFoleyRawVideo.FromConstantRate(decoded.Frames, decoded.Width, decoded.Height, decoded.Fps);
    }

    /// <summary>Disposes a list that grows after the runner is built (the video encoders join it on first use).</summary>
    private sealed class OwnedDisposables(List<IDisposable> items) : IDisposable
    {
        public void Dispose()
        {
            lock (items)
            {
                foreach (IDisposable d in items)
                {
                    d.Dispose();
                }

                items.Clear();
            }
        }
    }
}
