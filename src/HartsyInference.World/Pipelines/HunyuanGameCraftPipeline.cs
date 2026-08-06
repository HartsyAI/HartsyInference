using HartsyInference.Conditioning;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.World.ActionEncoders;
using HartsyInference.World.Camera;
using HartsyInference.World.Models;
using HartsyInference.Video;

namespace HartsyInference.World.Pipelines;

/// <summary>Hunyuan-GameCraft world-model pipeline: per-chunk action-conditioned video generation. Each chunk
/// denoises a noisy latent with the <see cref="HunyuanVideoDit"/> conditioned on Llava text tokens, a pooled
/// CLIP vector, and WASD/camera action via <see cref="GameCraftCameraNet"/> camera tokens, against a 33-channel
/// composite that carries the history/reference latent + mask (<see cref="GameCraftLatentBuilder"/>). The
/// denoised latent is decoded by the reused HunyuanVideo 3D VAE. Reuses the flow-match scheduler helpers
/// (<see cref="LancePipelineCommon"/>), <see cref="SeedGenerator"/>, and <see cref="VideoRgbFrames"/>.
/// <para><b>Numerics validation-pending</b> — timestep scaling, the txt token-refiner, and CameraNet temporal
/// schedule are validation-gated. Structurally green on synthetic weights via <see cref="DenoiseChunk"/>.</para></summary>
public sealed unsafe class HunyuanGameCraftPipeline : DiffusionPipelineBase
{
    private readonly HunyuanVideoDit _dit;
    private readonly HunyuanVideoVaeDecoder _vaeDec;
    private readonly HunyuanVideoVaeEncoder? _vaeEnc;
    private readonly GameCraftCameraNet _cameraNet;
    private readonly GameCraftActionEncoder _actionEncoder;
    private readonly HunyuanVideoConfig _cfg;
    private readonly int _spatialComp, _temporalComp;
    private readonly float _flowShift;
    private readonly List<IDisposable> _ownedLoaders = [];

    public HunyuanGameCraftPipeline(IBackend backend, HunyuanVideoDit dit, HunyuanVideoVaeDecoder vaeDec,
        GameCraftCameraNet cameraNet, GameCraftActionEncoder actionEncoder, HunyuanVideoConfig cfg,
        HunyuanVideoVaeEncoder? vaeEnc = null, int spatialCompression = 8, int temporalCompression = 4, float flowShift = 5f)
        : base(backend)
    {
        _dit = dit; _vaeDec = vaeDec; _vaeEnc = vaeEnc; _cameraNet = cameraNet; _actionEncoder = actionEncoder;
        _cfg = cfg; _spatialComp = spatialCompression; _temporalComp = temporalCompression; _flowShift = flowShift;
    }

    /// <summary>Whether <see cref="EncodeReferenceFrame"/> can run — false when the pipeline was assembled without a
    /// <see cref="HunyuanVideoVaeEncoder"/> (always false on a pipeline <see cref="LoadFromPath"/> returns — see its
    /// remarks: the structural key remap exists (<c>CheckpointConvertUtils.ConvertVaeEncoderKey</c> covers
    /// <c>encoder.*</c>/<c>quant_conv.*</c>), but the mid-block attention Conv3d→Linear reshape
    /// <c>HunyuanVideoCheckpointConverter.ConvertVaeDecoder</c>'s <c>AttnProj</c> local applies for the decoder has
    /// no encoder-side counterpart wired up yet).</summary>
    public bool CanEncodeReferenceFrame => _vaeEnc is not null;

    /// <summary>True once <see cref="LoadFromPath"/> builds a VAE encoder (currently always false — see
    /// <see cref="CanEncodeReferenceFrame"/>'s remarks). A dedicated static flag, not an instance check, so a caller
    /// that only opens sessions on top of <see cref="LoadFromPath"/> (<c>WorldService.LoadHunyuanGameCraft</c>) can
    /// fail fast — before loading the multi-GB checkpoint set — for a session that could never succeed regardless
    /// of which files are supplied.</summary>
    public static bool LoadFromPathBuildsVaeEncoder => false;

    /// <summary>Assembles a Hunyuan-GameCraft pipeline from its on-disk checkpoint set: the merged DiT+CameraNet
    /// dump (<paramref name="ditPath"/> — Tencent's <c>mp_rank_00_model_states.pt</c>, or an equivalent
    /// <c>.safetensors</c> repack) and the standalone HunyuanVideo 3D-VAE checkpoint (<paramref name="vaePath"/>,
    /// the same file family as the base HunyuanVideo T2V recipe's side-model VAE). Routes the merged dump through
    /// <see cref="HunyuanGameCraftCheckpointConverter.Convert"/> (coarse prefix split: <c>camera_in.*</c> →
    /// CameraNet, everything else → DiT under its native Tencent key names), then the DiT partition through
    /// <see cref="HunyuanVideoCheckpointConverter.Convert"/> (native Tencent/Comfy or diffusers naming → the hybrid
    /// keys <see cref="HunyuanVideoDit.LoadWeights"/> expects — the same remap the base HunyuanVideo T2V recipe
    /// uses). The VAE checkpoint is converted with <c>HunyuanVideoCheckpointConverter.ConvertVaeDecoder</c> only:
    /// <b>no VAE encoder is constructed</b> (<see cref="CanEncodeReferenceFrame"/> is false on the returned
    /// pipeline) — see that property's remarks for exactly what's missing (a reshape, not a whole converter). Text
    /// encoders (Llava-Llama-3-8B, CLIP-L) are NOT loaded here — they are model-generic components a caller
    /// assembles separately (see <c>WorldService.LoadHunyuanGameCraft</c>).
    /// <para><b>Numerics validation-pending</b> — this assembler has not been run against the real ~51GB checkpoint
    /// set (see <c>docs/Checklists/MODEL_STATUS_WORLD.md</c>); it is verified only by chaining the two converters
    /// on synthetic weight dicts and confirming the resulting key sets match what <see cref="HunyuanVideoDit.LoadWeights"/>
    /// and <see cref="GameCraftCameraNet.LoadWeights"/> index.</para></summary>
    /// <param name="height">Fixed generation height (px) the returned pipeline's <see cref="GameCraftActionEncoder"/>
    /// is bound to — Plücker-map sizing is a construction-time choice of the encoder, not a per-call argument (see
    /// <see cref="BuildCameraTokens"/>); must match the resolution every <see cref="DenoiseChunk"/> call on this
    /// pipeline actually runs at.</param>
    /// <param name="width">Fixed generation width (px); see <paramref name="height"/>.</param>
    public static HunyuanGameCraftPipeline LoadFromPath(IBackend backend, string ditPath, string vaePath,
        HunyuanVideoConfig? cfg = null, int height = 320, int width = 512)
    {
        ArgumentNullException.ThrowIfNull(backend);
        HunyuanVideoConfig config = cfg ?? HunyuanVideoConfig.GameCraftBase;
        List<IDisposable> owned = [];
        try
        {
            (Dictionary<string, Tensor> ditAll, IDisposable ditLoader) = HunyuanGameCraftCheckpointConverter.LoadRaw(ditPath);
            owned.Add(ditLoader);
            HunyuanGameCraftCheckpointConverter.ConvertedWeights routed = HunyuanGameCraftCheckpointConverter.Convert(ditAll);
            if (routed.Dit.Count == 0)
                throw new InvalidOperationException($"'{ditPath}' has no recognized DiT weights after routing.");
            if (routed.CameraNet.Count == 0)
                throw new InvalidOperationException($"'{ditPath}' has no 'camera_in.*' weights — CameraNet cannot load from it.");

            Dictionary<string, Tensor> ditHybrid = HunyuanVideoCheckpointConverter.Convert(routed.Dit);
            HunyuanVideoDit dit = new HunyuanVideoDit(config);
            dit.LoadWeights(ditHybrid);

            GameCraftCameraNet cameraNet = new GameCraftCameraNet(config.HiddenSize);
            cameraNet.LoadWeights(routed.CameraNet);

            (Dictionary<string, Tensor> vaeAll, IDisposable vaeLoader) = HunyuanGameCraftCheckpointConverter.LoadRaw(vaePath);
            owned.Add(vaeLoader);
            Dictionary<string, Tensor> vaeDecoderWeights = HunyuanVideoCheckpointConverter.ConvertVaeDecoder(vaeAll);
            if (!vaeDecoderWeights.ContainsKey("decoder.conv_in.weight"))
                throw new InvalidOperationException(
                    $"'{vaePath}' did not convert to a recognized HunyuanVideo VAE decoder (missing 'decoder.conv_in.weight' " +
                    "after ldm/Tencent→diffusers remap) — expected the same checkpoint layout as the base HunyuanVideo VAE.");
            HunyuanVideoVaeDecoder vaeDec = new HunyuanVideoVaeDecoder();
            vaeDec.LoadWeights(vaeDecoderWeights);

            GameCraftActionEncoder actionEncoder = GameCraftActionEncoder.CreateDefault(height, width);

            HunyuanGameCraftPipeline pipeline = new HunyuanGameCraftPipeline(backend, dit, vaeDec, cameraNet,
                actionEncoder, config, vaeEnc: null, flowShift: config.FlowShift);
            pipeline._ownedLoaders.AddRange(owned);
            return pipeline;
        }
        catch
        {
            foreach (IDisposable loader in owned) loader.Dispose();
            throw;
        }
    }

    /// <inheritdoc/>
    protected override void DisposeCore()
    {
        foreach (IDisposable loader in _ownedLoaders) loader.Dispose();
        _ownedLoaders.Clear();
    }

    /// <summary>Denoises one chunk and returns the latent <c>[1, 16, tLat, Hlat, Wlat]</c> (no VAE decode) — the
    /// testable integration core. <paramref name="historyLatent"/> + <paramref name="mask"/> form the composite;
    /// <paramref name="actionPayload"/> drives the per-frame Plücker → CameraNet camera tokens.</summary>
    public Tensor DenoiseChunk(Tensor promptEmbeds, Tensor pooled, Tensor negEmbeds, Tensor negPooled,
        Tensor historyLatent, ReadOnlySpan<float> mask, ReadOnlySpan<byte> actionPayload,
        int steps, float guidance, int seed)
    {
        ThrowIfDisposed();
        int tLat = (int)historyLatent.Shape[2], hLat = (int)historyLatent.Shape[3], wLat = (int)historyLatent.Shape[4];
        int fullH = hLat * _spatialComp, fullW = wLat * _spatialComp;

        // Camera tokens from the action (built once; conditioning is constant across denoise steps).
        Backend.PreloadWeights(_cameraNet.EnumerateWeights());
        Tensor cameraTokens = BuildCameraTokens(actionPayload, tLat, fullH, fullW);
        Backend.FreeWeights(_cameraNet.EnumerateWeights());

        Tensor noisy = SeedGenerator.CreateNoise(new TensorShape([1L, _cfg.OutChannels, tLat, hLat, wLat]), seed);
        float[] tsteps = LancePipelineCommon.BuildShiftedTimesteps(steps, _flowShift);
        float[] maskArr = mask.ToArray();

        Backend.PreloadWeights(_dit.EnumerateWeights());
        for (int k = 0; k < steps; k++)
        {
            float t = tsteps[k], dt = t - tsteps[k + 1];
            Tensor composite = GameCraftLatentBuilder.Build(noisy, historyLatent, maskArr);
            Tensor vCond = _dit.Forward(Backend, composite, promptEmbeds, pooled, t, guidance: 0f, cameraTokens: cameraTokens);
            Tensor vUncond = _dit.Forward(Backend, composite, negEmbeds, negPooled, t, guidance: 0f, cameraTokens: cameraTokens);
            composite.Dispose();
            LancePipelineCommon.EulerCfgStep(noisy, vCond, vUncond, guidance, dt);
            vCond.Dispose();
            vUncond.Dispose();
        }
        Backend.Sync();
        Backend.FreeWeights(_dit.EnumerateWeights());
        cameraTokens.Dispose();
        return noisy;
    }

    /// <summary>Denoises one chunk and decodes it to interleaved-RGB frames via the HunyuanVideo VAE.</summary>
    public byte[][] GenerateChunk(Tensor promptEmbeds, Tensor pooled, Tensor negEmbeds, Tensor negPooled,
        Tensor historyLatent, ReadOnlySpan<float> mask, ReadOnlySpan<byte> actionPayload, int steps, float guidance, int seed)
    {
        Tensor latent = DenoiseChunk(promptEmbeds, pooled, negEmbeds, negPooled, historyLatent, mask, actionPayload, steps, guidance, seed);
        try { return DecodeLatentToFrames(latent); }
        finally { latent.Dispose(); }
    }

    /// <summary>Decodes a latent <c>[1,16,tLat,Hlat,Wlat]</c> to interleaved-RGB frames via the HunyuanVideo VAE.</summary>
    public byte[][] DecodeLatentToFrames(Tensor latent)
    {
        Backend.PreloadWeights(_vaeDec.EnumerateWeights());
        Tensor rgb;
        try { rgb = _vaeDec.Decode(Backend, latent); }
        finally { Backend.FreeWeights(_vaeDec.EnumerateWeights()); }

        int f = (int)rgb.Shape[2];
        byte[][] frames = new byte[f][];
        for (int i = 0; i < f; i++) frames[i] = VideoRgbFrames.ExtractFrame(rgb, i);
        rgb.Dispose();
        return frames;
    }

    /// <summary>Encodes a reference RGB frame to a single-frame latent for chunk-0 history (requires a VAE encoder).</summary>
    public Tensor EncodeReferenceFrame(ReadOnlySpan<byte> rgb24, int width, int height)
    {
        if (_vaeEnc is null) throw new InvalidOperationException("Reference-frame encoding requires a HunyuanVideoVaeEncoder.");
        return _vaeEnc.EncodeRgbFrame(Backend, rgb24, width, height);
    }

    /// <summary>Builds camera tokens: per-frame Plücker maps (HWC from the action encoder, transposed to CHW)
    /// assembled into <c>[1, T_pix, 6, H, W]</c>, then run through the CameraNet.</summary>
    private Tensor BuildCameraTokens(ReadOnlySpan<byte> actionPayload, int tLat, int H, int W)
    {
        int tPix = (tLat - 1) * _temporalComp + 1;
        Tensor plucker = new(new TensorShape([1L, tPix, PluckerEmbedding.Channels, H, W]), DType.F32);
        float* pp = (float*)plucker.DataPointer;
        long frame = (long)H * W;
        byte[] payload = actionPayload.ToArray();
        float[] hwc = new float[H * W * PluckerEmbedding.Channels];
        for (int f = 0; f < tPix; f++)
        {
            ActionInput action = new(payload, f, 0);
            _actionEncoder.Encode(action, "plucker", hwc);
            // HWC (y,x,c) → CHW (c,y,x) into the plucker tensor.
            long frameBase = (long)f * PluckerEmbedding.Channels * frame;
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int c = 0; c < PluckerEmbedding.Channels; c++)
                        pp[frameBase + (long)c * frame + (long)y * W + x] = hwc[((long)y * W + x) * PluckerEmbedding.Channels + c];
        }
        return _cameraNet.Forward(Backend, plucker);
    }
}
