using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Wan-Animate DiT (<c>WanAnimateTransformer3DModel</c>), ported from diffusers
/// <c>transformer_wan_animate.py</c>. The base Wan2.1 video transformer plus character-animation conditioning: a
/// <c>pose_patch_embedding</c> whose tokens are added to the non-first latent frames, and a face pathway — the
/// <see cref="WanAnimateMotionEncoder"/> (pixel face frames → motion vectors via QR Linear Motion Decomposition) →
/// <see cref="WanAnimateFaceEncoder"/> (temporal features) → a <see cref="WanAnimateFaceBlock"/> face adapter
/// cross-attention injected after every <c>InjectFaceLatentsBlocks</c>-th main block. Reuses <see cref="WanDitOps"/>
/// + <see cref="WanVideoBlock"/> for the base path; B=1; numerics validation-pending.</summary>
public sealed unsafe class WanAnimateTransformer : IDisposable
{
    private readonly WanVideoConfig _config;
    private readonly WanVideoBlock[] _blocks;
    private readonly WanAnimateFaceBlock[] _faceAdapter;
    private readonly WanAnimateMotionEncoder _motionEncoder;
    private readonly WanAnimateFaceEncoder _faceEncoder;
    private readonly WanRope _rope;
    private readonly int _patchVec, _posePatchVec, _injectBlocks, _poseLatentChannels, _motionSize;
    private int _disposed;

    private Tensor? _patchW2d, _patchB, _posePatchW2d, _posePatchB;
    private Tensor? _projOutW, _projOutB, _finalScaleShift;
    private Tensor? _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB;
    private Tensor? _textW1, _textB1, _textW2, _textB2;

    public WanAnimateTransformer(WanVideoConfig config, int poseLatentChannels = 16, int motionEncoderSize = 512,
        int motionDim = 512, int faceHiddenDim = 1024, int faceNumHeads = 4, int injectFaceLatentsBlocks = 5,
        Dictionary<int, int>? motionChannelSizes = null, int motionVecDim = 20, int motionBlocks = 5)
    {
        _config = config;
        _injectBlocks = injectFaceLatentsBlocks;
        _poseLatentChannels = poseLatentChannels;
        _motionSize = motionEncoderSize;
        _blocks = new WanVideoBlock[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++) _blocks[i] = new WanVideoBlock(config, crossAttnNorm: true);
        int numAdapters = config.NumLayers / injectFaceLatentsBlocks;
        _faceAdapter = new WanAnimateFaceBlock[numAdapters];
        for (int i = 0; i < numAdapters; i++) _faceAdapter[i] = new WanAnimateFaceBlock(config.InnerDim, config.NumHeads, config.HeadDim, config.Eps);
        _motionEncoder = new WanAnimateMotionEncoder(motionEncoderSize, styleDim: motionDim, motionDim: motionVecDim,
            outDim: motionDim, motionBlocks: motionBlocks, channels: motionChannelSizes);
        _faceEncoder = new WanAnimateFaceEncoder(motionDim, config.InnerDim, faceHiddenDim, faceNumHeads);
        _rope = new WanRope(config.HeadDim, config.RopeTheta, config.RopeMaxSeqLen);
        _patchVec = config.InChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
        _posePatchVec = poseLatentChannels * config.PatchSize.T * config.PatchSize.H * config.PatchSize.W;
    }

    public WanVideoConfig Config => _config;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _patchW2d = WanDitOps.Reshape2d(w["patch_embedding.weight"], _config.InnerDim, _patchVec);
        w.TryGetValue("patch_embedding.bias", out _patchB);
        _posePatchW2d = WanDitOps.Reshape2d(w["pose_patch_embedding.weight"], _config.InnerDim, _posePatchVec);
        w.TryGetValue("pose_patch_embedding.bias", out _posePatchB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _finalScaleShift = LoadF32(w, "scale_shift_table");
        _timeEmb1W = w["condition_embedder.time_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_1.bias", out _timeEmb1B);
        _timeEmb2W = w["condition_embedder.time_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.time_embedder.linear_2.bias", out _timeEmb2B);
        _timeProjW = w["condition_embedder.time_proj.weight"]; w.TryGetValue("condition_embedder.time_proj.bias", out _timeProjB);
        _textW1 = w["condition_embedder.text_embedder.linear_1.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_1.bias", out _textB1);
        _textW2 = w["condition_embedder.text_embedder.linear_2.weight"]; w.TryGetValue("condition_embedder.text_embedder.linear_2.bias", out _textB2);
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"blocks.{i}");
        for (int i = 0; i < _faceAdapter.Length; i++) _faceAdapter[i].LoadWeights(w, $"face_adapter.{i}");
        _motionEncoder.LoadWeights(w, "motion_encoder");
        _faceEncoder.LoadWeights(w, "face_encoder");
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _patchW2d, _patchB, _posePatchW2d, _posePatchB, _projOutW, _projOutB, _finalScaleShift,
            _timeEmb1W, _timeEmb1B, _timeEmb2W, _timeEmb2B, _timeProjW, _timeProjB, _textW1, _textB1, _textW2, _textB2 })
            if (t is not null) yield return t;
        for (int i = 0; i < _blocks.Length; i++) foreach (Tensor t in _blocks[i].EnumerateWeights()) yield return t;
        for (int i = 0; i < _faceAdapter.Length; i++) foreach (Tensor t in _faceAdapter[i].EnumerateWeights()) yield return t;
        foreach (Tensor t in _motionEncoder.EnumerateWeights()) yield return t;
        foreach (Tensor t in _faceEncoder.EnumerateWeights()) yield return t;
    }

    /// <summary>Velocity prediction. <paramref name="latent"/> is <c>[1, inChannels, T, H, W]</c>;
    /// <paramref name="pose"/> is <c>[1, poseLatentChannels, T−1, H, W]</c>; <paramref name="facePixels"/> is
    /// <c>[1, 3, Tface, size, size]</c> in [-1, 1]; <paramref name="encoder"/> is umT5 features <c>[L, textDim]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor pose, Tensor facePixels, Tensor encoder, float timestep)
    {
        int t = (int)latent.Shape[2], hh = (int)latent.Shape[3], ww = (int)latent.Shape[4];
        (int pt, int ph, int pw) = _config.PatchSize;
        int gt = t / pt, gh = hh / ph, gw = ww / pw;
        int s = gt * gh * gw, frame = gh * gw, dim = _config.InnerDim;

        (Tensor cos, Tensor sin) = _rope.BuildCosSin(gt, gh, gw);

        Tensor hidden = WanDitOps.Patchify(backend, latent, _config.InChannels, dim, _config.PatchSize, _patchW2d!, _patchB);
        // Pose tokens (gt−1 frames) added to the non-first latent frames.
        Tensor poseTokens = WanDitOps.Patchify(backend, pose, _poseLatentChannels, dim, _config.PatchSize, _posePatchW2d!, _posePatchB);
        AddPose(hidden, poseTokens, frame, dim);
        poseTokens.Dispose();

        (Tensor temb, Tensor timestepProj) = WanDitOps.ConditionTimeGroups(backend, [timestep], _config.FreqDim, dim,
            _timeEmb1W!, _timeEmb1B, _timeEmb2W!, _timeEmb2B, _timeProjW!, _timeProjB);
        Tensor encoderProj = WanDitOps.TextEmbed(backend, encoder, dim, _textW1!, _textB1, _textW2!, _textB2);

        // Face pathway: pixel frames → motion vectors → temporal face features (+ a prepended zero frame).
        Tensor motion = BuildMotion(backend, facePixels);

        Tensor cur = hidden;
        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, cur, encoderProj, timestepProj, _rope, cos, sin, s);
            cur.Dispose();
            cur = next;
            if (i % _injectBlocks == 0)
            {
                Tensor adapted = _faceAdapter[i / _injectBlocks].Forward(backend, cur, motion);
                AddInPlace(cur, adapted);
                adapted.Dispose();
            }
        }
        motion.Dispose();
        cos.Dispose(); sin.Dispose(); timestepProj.Dispose(); encoderProj.Dispose();

        Tensor projected = WanDitOps.FinalLayer(backend, cur, temb, _finalScaleShift!, _projOutW!, _projOutB, s, dim, _config.Eps, s);
        cur.Dispose();
        temb.Dispose();
        Tensor outVel = WanDitOps.Unpatchify(projected, _config.OutChannels, gt, gh, gw, _config.PatchSize);
        projected.Dispose();
        return outVel;
    }

    /// <summary>Runs the motion encoder over each face frame, the face encoder over the sequence, and prepends a zero
    /// frame. <paramref name="facePixels"/> <c>[1, 3, Tface, size, size]</c> → motion features <c>[Tface'+1, N+1, dim]</c>.</summary>
    private Tensor BuildMotion(IBackend backend, Tensor facePixels)
    {
        int tFace = (int)facePixels.Shape[2], h = (int)facePixels.Shape[3], wdt = (int)facePixels.Shape[4];
        // [1,3,Tface,H,W] → [Tface,3,H,W]
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

        // Prepend a zero frame along the temporal axis → [T'+1, N+1, dim].
        int tp = (int)faceFeat.Shape[0], n = (int)faceFeat.Shape[1], d = (int)faceFeat.Shape[2];
        Tensor padded = new Tensor(new TensorShape(tp + 1, n, d), DType.F32);
        float* pp = (float*)padded.DataPointer, ffp = (float*)faceFeat.DataPointer;
        long perFrame = (long)n * d;
        new Span<float>(pp, (int)perFrame).Clear();   // zero first frame
        Buffer.MemoryCopy(ffp, pp + perFrame, (long)tp * perFrame * 4, (long)tp * perFrame * 4);
        faceFeat.Dispose();
        return padded;
    }

    /// <summary>Adds pose tokens (<c>(gt−1)·frame</c> rows) to the latent tokens of frames 1..gt−1 (offset <c>frame</c>).</summary>
    private static void AddPose(Tensor hidden, Tensor pose, int frame, int dim)
    {
        long poseRows = pose.Shape[0];
        float* hp = (float*)hidden.DataPointer, pp = (float*)pose.DataPointer;
        long offset = (long)frame * dim;
        long n = poseRows * dim;
        for (long i = 0; i < n; i++) hp[offset + i] += pp[i];
    }

    private static void AddInPlace(Tensor acc, Tensor add)
    {
        long n = acc.Shape.ElementCount;
        float* ap = (float*)acc.DataPointer, dp = (float*)add.DataPointer;
        for (long i = 0; i < n; i++) ap[i] += dp[i];
    }

    private static Tensor LoadF32(IReadOnlyDictionary<string, Tensor> w, string key) { Tensor t = w[key]; return t.DType == DType.F32 ? t : t.CastTo(DType.F32); }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _patchW2d = _patchB = _posePatchW2d = _posePatchB = _projOutW = _projOutB = _finalScaleShift = null;
            _timeEmb1W = _timeEmb1B = _timeEmb2W = _timeEmb2B = _timeProjW = _timeProjB = null;
            _textW1 = _textB1 = _textW2 = _textB2 = null;
        }
        GC.SuppressFinalize(this);
    }
}
