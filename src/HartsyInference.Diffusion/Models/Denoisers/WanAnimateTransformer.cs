using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Wan2.2-Animate DiT (<c>AnimateWanModel</c>, ported from ComfyUI <c>comfy/ldm/wan/model_animate.py</c>):
/// the Wan i2v backbone (concat conditioning + optional CLIP <c>img_emb</c> with per-block image KV) plus character
/// animation conditioning. <c>pose_patch_embedding</c> (Conv3d, patch-size kernel) tokens are ADDED to the latent
/// tokens of temporal frames <c>1..pose_T</c> (frame 0 is the reference latent). Face pathway:
/// <see cref="WanAnimateMotionEncoder"/> (512×512 face pixels in [-1,1] → per-frame motion vectors via QR Linear
/// Motion Decomposition) → <see cref="WanAnimateFaceEncoder"/> (4× temporal downsample, per-head streams) → a zero
/// frame prepended and the sequence zero-padded/truncated to the latent frame count → fused residually as
/// <c>x = x + face_adapter.fuser_blocks[i/5](x, motion)</c> after every block where <c>i % 5 == 0</c>. Optional
/// <c>ref_conv</c> (Conv2d) reference-latent tokens are PREPENDED to the sequence (RoPE spans gt+1 frames; the rows
/// are trimmed after the head). Reuses <see cref="WanDitOps"/> + <see cref="WanVideoBlock"/>; B=1.</summary>
public sealed unsafe class WanAnimateTransformer : IDisposable
{
    /// <summary>Face-adapter fusion interval: a fuser block runs after every DiT block where <c>i % 5 == 0</c>.</summary>
    public const int FuseEvery = 5;

    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanAnimateMotionEncoder _motionEncoder;
    private readonly WanAnimateFaceEncoder _faceEncoder;
    private readonly WanImageEmbedder _imgEmbedder;
    private readonly WanRope _rope;
    private readonly int _patchVec;
    private WanAnimateFaceBlock[] _faceAdapter = [];
    private int _poseChannels;
    private int _disposed;

    private Tensor? _patchW2d, _patchB, _posePatchW2d, _posePatchB, _refConvW2d, _refConvB;
    private Tensor? _projOutW, _projOutB, _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;

    public WanAnimateTransformer(WanVideoConfig config)
    {
        _config = config;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        _motionEncoder = new WanAnimateMotionEncoder();
        _faceEncoder = new WanAnimateFaceEncoder(config.Eps);
        _imgEmbedder = new WanImageEmbedder(config.Eps);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public WanVideoConfig Config => _config;

    /// <summary>True when the checkpoint ships <c>ref_conv</c> (reference-latent token prepend).</summary>
    public bool HasRefConv => _refConvW2d is not null;

    /// <summary>Loads the converted (post-<c>WanVideoCheckpointConverter</c>) weights. The base i2v keys are in
    /// diffusers naming; the animate-specific keys (<c>pose_patch_embedding.*</c>, <c>motion_encoder.*</c>,
    /// <c>face_encoder.*</c>, <c>face_adapter.fuser_blocks.*</c>, <c>ref_conv.*</c>) pass the converter unchanged.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW2d = WanDitOps.Reshape2d(w["patch_embedding.weight"], _config.InnerDim, _patchVec);
        w.TryGetValue("patch_embedding.bias", out _patchB);

        Tensor poseW = w["pose_patch_embedding.weight"];   // [inner, poseC, pt, ph, pw]
        _poseChannels = (int)poseW.Shape[1];
        _posePatchW2d = WanDitOps.Reshape2d(poseW, _config.InnerDim,
            _poseChannels * _config.PatchSize.T * _config.PatchSize.H * _config.PatchSize.W);
        w.TryGetValue("pose_patch_embedding.bias", out _posePatchB);

        if (w.TryGetValue("ref_conv.weight", out Tensor? refW))   // Conv2d [inner, refC, ph, pw]
        {
            _refConvW2d = WanDitOps.Reshape2d(refW, _config.InnerDim,
                (int)refW.Shape[1] * _config.PatchSize.H * _config.PatchSize.W);
            w.TryGetValue("ref_conv.bias", out _refConvB);
        }

        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["condition_embedder.time_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["condition_embedder.time_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_2.bias", out _timeEmb2B);
        _timeProjW = w["condition_embedder.time_proj.weight"]; w.TryGetValue("condition_embedder.time_proj.bias", out _timeProjB);
        _textW1 = w["condition_embedder.text_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_1.bias", out _textB1);
        _textW2 = w["condition_embedder.text_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_2.bias", out _textB2);
        _imgEmbedder.TryLoadWeights(w);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");

        int fusers = 0;
        while (w.ContainsKey($"face_adapter.fuser_blocks.{fusers}.linear1_kv.weight")) fusers++;
        if (fusers != (_config.NumLayers + FuseEvery - 1) / FuseEvery)
            throw new ArgumentException(
                $"face_adapter has {fusers} fuser blocks; expected ceil({_config.NumLayers}/{FuseEvery}) " +
                $"(reference: num_layers // 5 with fusion at i % 5 == 0).", nameof(w));
        _faceAdapter = new WanAnimateFaceBlock[fusers];
        for (int i = 0; i < fusers; i++)
        {
            _faceAdapter[i] = new WanAnimateFaceBlock(_config.InnerDim, _config.NumHeads, _config.HeadDim, _config.Eps);
            _faceAdapter[i].LoadWeights(w, $"face_adapter.fuser_blocks.{i}");
        }
        _motionEncoder.LoadWeights(w, "motion_encoder");
        _faceEncoder.LoadWeights(w, "face_encoder");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _posePatchW2d, _posePatchB, _refConvW2d, _refConvB,
            _projOutW, _projOutB, _finalScaleShift, _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB,
            _textW1, _textB1, _textW2, _textB2 })
            if (t is not null) yield return t;
        foreach (Tensor t in _imgEmbedder.EnumerateWeights()) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
        for (int i = 0; i < _faceAdapter.Length; i++) foreach (Tensor t in _faceAdapter[i].EnumerateWeights()) yield return t;
        foreach (Tensor t in _motionEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _faceEncoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction. <paramref name="latent"/> is <c>[1, inChannels, T, H, W]</c> (noise + concat
    /// conditioning); <paramref name="pose"/> is the pose-video latent <c>[1, poseC, Tpose, H, W]</c> (added to
    /// frames 1..Tpose, clamped to T−1); <paramref name="facePixels"/> is <c>[1, 3, Tface, 512, 512]</c> in [-1, 1];
    /// <paramref name="encoder"/> is umT5 features <c>[L, textDim]</c>; <paramref name="clipImageEmbeds"/> is the
    /// optional CLIP-ViT-H penultimate hidden state for the per-block image cross-attention;
    /// <paramref name="referenceLatent"/> is the optional <c>ref_conv</c> reference latent <c>[1, refC, H, W]</c>
    /// (token-prepend variant — mutually exclusive with the face pathway, as in the reference).</summary>
    /// <summary>Precomputed face-motion features for <see cref="Forward"/>'s <c>motion</c> parameter. The face clip is
    /// constant across the denoise, so encode it ONCE per CFG branch — the StyleGAN motion encoder is host-side and
    /// running it inside every forward (2×/step) makes a 14B run impractically slow.</summary>
    public Tensor EncodeMotion(IBackend backend, Tensor facePixels, int latentFrames) =>
        BuildMotion(backend, facePixels, latentFrames / _config.PatchSize.T);

    public Tensor Forward(IBackend backend, Tensor latent, Tensor? pose, Tensor? facePixels, Tensor encoder,
        float timestep, Tensor? clipImageEmbeds = null, Tensor? referenceLatent = null, Tensor? motion = null)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int frame = gh * gw, dim = _config.InnerDim;

        bool useRef = referenceLatent is not null && _refConvW2d is not null;
        if (useRef && (facePixels is not null || motion is not null))
            throw new ArgumentException("ref_conv token-prepend and the face pathway cannot combine: the fuser's " +
                "temporal grouping (T | S) breaks once ref tokens are prepended (same constraint as the reference).");

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);

        // after_patch_embedding: pose tokens added to frames 1..pose_T (x[:, :, 1:pose_T+1] += pose[:, :, :x_T−1]).
        if (pose is not null)
        {
            Tensor poseTokens = WanDitOps.Patchify(backend, pose, _poseChannels, dim, _config.PatchSize, _posePatchW2d!, _posePatchB);
            int gtPose = (int)pose.Shape[2] / pt;
            AddPose(hidden, poseTokens, frame, dim, Math.Min(gtPose, gt - 1));
            poseTokens.Dispose();
        }

        // Face pathway: pixels → motion vectors → face features; zero frame prepended, then padded/truncated to gt.
        // Prefer the caller's precomputed features (EncodeMotion) — the per-forward encode is a debug/compat path.
        bool ownsMotion = motion is null && facePixels is not null;
        motion ??= facePixels is not null ? BuildMotion(backend, facePixels, gt) : null;

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, [timestep], _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor textProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        int imageContextLen = 0;
        Tensor encoderProj = textProj;
        if (clipImageEmbeds is not null && _imgEmbedder.IsLoaded)
        {
            Tensor imgProj = _imgEmbedder.Forward(backend, clipImageEmbeds, dim);
            imageContextLen = (int)imgProj.Shape[0];
            encoderProj = WanDitOps.ConcatRows(imgProj, textProj, dim);
            imgProj.Dispose();
            textProj.Dispose();
        }

        // ref_conv: reference tokens prepended; RoPE spans gt+1 temporal frames (ref = frame 0, video = 1..gt).
        int refRows = 0;
        if (useRef)
        {
            Tensor refTokens = PatchifyRef(backend, referenceLatent!, dim);
            refRows = (int)refTokens.Shape[0];
            Tensor combined = WanDitOps.ConcatRows(refTokens, hidden, dim);
            refTokens.Dispose();
            hidden.Dispose();
            hidden = combined;
        }
        int s = gt * frame + refRows;
        (Tensor cos, Tensor sin) = _rope.BuildCosSin(useRef ? gt + 1 : gt, gh, gw);

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, s,
                imageContextLen: imageContextLen);
            cur.Dispose();
            cur = next;
            if (motion is not null && i % FuseEvery == 0)
            {
                Tensor adapted = _faceAdapter[i / FuseEvery].Forward(backend, cur, motion);
                AddInPlace(cur, adapted);
                adapted.Dispose();
            }
        }
        if (ownsMotion) motion?.Dispose();
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, s);
        cur.Dispose();
        temb.Dispose();

        // Trim the ref rows after the head (x = x[:, full_ref.shape[1]:]), then unpatchify the video tokens.
        if (refRows > 0)
        {
            Tensor trimmed = SliceRows(projected, refRows, gt * frame);
            projected.Dispose();
            projected = trimmed;
        }
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    /// <summary>Runs the motion encoder over each face frame and the face encoder over the sequence, prepends a zero
    /// frame (temporal slot 0 = reference latent frame) and zero-pads/truncates to <paramref name="gt"/> frames.
    /// <paramref name="facePixels"/> <c>[1, 3, Tface, size, size]</c> → motion features <c>[gt, N+1, dim]</c>.</summary>
    private Tensor BuildMotion(IBackend backend, Tensor facePixels, int gt)
    {
        int tFace = (int)facePixels.Shape[2], h = (int)facePixels.Shape[3], wdt = (int)facePixels.Shape[4];
        // [1,3,Tface,H,W] → [Tface,3,H,W] (the reference's "b c t h w -> (b t) c h w").
        Tensor frames = new Tensor(new TensorShape(tFace, 3, h, wdt), DType.F32);
        float* fp = (float*)facePixels.DataPointer, frp = (float*)frames.DataPointer;
        long frameSize = (long)h * wdt;
        for (int ti = 0; ti < tFace; ti++)
            for (int c = 0; c < 3; c++)
                Buffer.MemoryCopy(fp + ((long)c * tFace + ti) * frameSize, frp + ((long)ti * 3 + c) * frameSize, frameSize * 4, frameSize * 4);
        Tensor motionVec = _motionEncoder.Forward(backend, frames);   // [Tface, motionDim]
        frames.Dispose();

        Tensor faceFeat = _faceEncoder.Forward(backend, motionVec);   // [T', N+1, dim]
        motionVec.Dispose();

        // Prepend a zero frame, then zero-pad or truncate along the temporal axis to gt.
        int tp = (int)faceFeat.Shape[0], n = (int)faceFeat.Shape[1], d = (int)faceFeat.Shape[2];
        Tensor padded = new Tensor(new TensorShape(gt, n, d), DType.F32);
        float* pp = (float*)padded.DataPointer, ffp = (float*)faceFeat.DataPointer;
        long perFrame = (long)n * d;
        new Span<float>(pp, (int)(gt * perFrame)).Clear();
        int copyFrames = Math.Min(tp, gt - 1);
        if (copyFrames > 0)
            Buffer.MemoryCopy(ffp, pp + perFrame, copyFrames * perFrame * 4, copyFrames * perFrame * 4);
        faceFeat.Dispose();
        return padded;
    }

    /// <summary>ref_conv as a linear over 2D patches: reference latent <c>[1, refC, H, W]</c> → tokens <c>[gh·gw, dim]</c>.</summary>
    private Tensor PatchifyRef(IBackend backend, Tensor referenceLatent, int dim)
    {
        int refC = (int)referenceLatent.Shape[1];
        int h = (int)referenceLatent.Shape[referenceLatent.Shape.Rank - 2];
        int w = (int)referenceLatent.Shape[referenceLatent.Shape.Rank - 1];
        // View as [1, refC, 1, H, W] and reuse the Conv3d-as-linear patchify with patch (1, ph, pw); the Conv2d
        // weight [dim, refC, ph, pw] has the same contiguous layout as a Conv3d weight [dim, refC, 1, ph, pw].
        Tensor view = new Tensor(new TensorShape([1L, refC, 1, h, w]), DType.F32);
        Buffer.MemoryCopy((float*)referenceLatent.DataPointer, (float*)view.DataPointer,
            (long)refC * h * w * 4, (long)refC * h * w * 4);
        Tensor tokens = WanDitOps.Patchify(backend, view, refC, dim,
            (1, _config.PatchSize.H, _config.PatchSize.W), _refConvW2d!, _refConvB);
        view.Dispose();
        return tokens;
    }

    /// <summary>Adds pose tokens to the latent tokens of frames 1..<paramref name="framesAdded"/> (token offset <c>frame</c>).</summary>
    private static void AddPose(Tensor hidden, Tensor pose, int frame, int dim, int framesAdded)
    {
        if (framesAdded <= 0) return;
        float* hp = (float*)hidden.DataPointer, pp = (float*)pose.DataPointer;
        long offset = (long)frame * dim;
        long count = (long)framesAdded * frame * dim;
        for (long i = 0; i < count; i++) hp[offset + i] += pp[i];
    }

    private static void AddInPlace(Tensor acc, Tensor add)
    {
        long n = acc.Shape.ElementCount;
        float* ap = (float*)acc.DataPointer, dp = (float*)add.DataPointer;
        for (long i = 0; i < n; i++) ap[i] += dp[i];
    }

    private static Tensor SliceRows(Tensor x, int start, int len)
    {
        int cols = (int)x.Shape[x.Shape.Rank - 1];
        Tensor o = new Tensor(new TensorShape(len, cols), DType.F32);
        Buffer.MemoryCopy((float*)x.DataPointer + (long)start * cols, (float*)o.DataPointer,
            (long)len * cols * 4, (long)len * cols * 4);
        return o;
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _posePatchW2d = _posePatchB = _refConvW2d = _refConvB = null;
            _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
            _imgEmbedder.Clear();
        }
        GC.SuppressFinalize(this);
    }
}
