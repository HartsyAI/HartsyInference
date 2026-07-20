using HartsyInference.Conditioning;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Utilities;
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

    public HunyuanGameCraftPipeline(IBackend backend, HunyuanVideoDit dit, HunyuanVideoVaeDecoder vaeDec,
        GameCraftCameraNet cameraNet, GameCraftActionEncoder actionEncoder, HunyuanVideoConfig cfg,
        HunyuanVideoVaeEncoder? vaeEnc = null, int spatialCompression = 8, int temporalCompression = 4, float flowShift = 5f)
        : base(backend)
    {
        _dit = dit; _vaeDec = vaeDec; _vaeEnc = vaeEnc; _cameraNet = cameraNet; _actionEncoder = actionEncoder;
        _cfg = cfg; _spatialComp = spatialCompression; _temporalComp = temporalCompression; _flowShift = flowShift;
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
