using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.World.Camera;

namespace HartsyInference.World.Models;

/// <summary>Hunyuan-GameCraft CameraNet: turns per-frame 6-channel Plücker ray maps into camera tokens that are
/// token-added to the DiT image stream. Pipeline: <c>PixelUnshuffle(8)</c> → <c>Conv1×1 384→192</c> + GN + ReLU →
/// temporal compression → <c>Conv1×1 192→96</c> + GN + ReLU → <c>Conv1×1 96→16</c> (zero-init) →
/// <c>PatchEmbed[1,2,2]→hidden</c> → learnable per-channel scale. Output tokens align with the DiT image grid
/// <c>(tLat, H/16, W/16)</c>. Reuses <see cref="PluckerEmbedding"/>'s 6-channel input convention and the backend's
/// Conv2D/GroupNorm.
/// <para><b>Validation-gated:</b> the temporal-compression schedule and zero-init are reconciled against the
/// reference; structurally green on synthetic weights.</para></summary>
public sealed unsafe class GameCraftCameraNet
{
    private const int PluckerCh = PluckerEmbedding.Channels; // 6
    private readonly int _hidden, _downscale, _patchH, _patchW, _outCh, _gnGroups;

    private Tensor? _enc1W, _enc1B, _gn1W, _gn1B;   // conv 384→192
    private Tensor? _enc2W, _enc2B, _gn2W, _gn2B;   // conv 192→96
    private Tensor? _finalW, _finalB;               // conv 96→16
    private Tensor? _patchW_, _patchB;              // [hidden, 16*pH*pW]
    private Tensor? _scale;                         // [hidden]

    /// <param name="temporalCompression">Obsolete/ignored: temporal compression now follows upstream's fixed schedule
    /// (two factor-2 <c>compress_time</c> passes, ~4× overall). Retained for call-site compatibility.</param>
    public GameCraftCameraNet(int hiddenSize, int downscale = 8, int outChannels = 16, int patchH = 2, int patchW = 2,
        int temporalCompression = 4, int gnGroups = 2)
    {
        _ = temporalCompression;
        _hidden = hiddenSize; _downscale = downscale; _outCh = outChannels;
        _patchH = patchH; _patchW = patchW; _gnGroups = gnGroups;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "camera_in")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _enc1W = F32(w[$"{p}encode_first.0.weight"]); _enc1B = F32(w[$"{p}encode_first.0.bias"]);
        _gn1W = F32(w[$"{p}encode_first.1.weight"]); _gn1B = F32(w[$"{p}encode_first.1.bias"]);
        _enc2W = F32(w[$"{p}encode_second.0.weight"]); _enc2B = F32(w[$"{p}encode_second.0.bias"]);
        _gn2W = F32(w[$"{p}encode_second.1.weight"]); _gn2B = F32(w[$"{p}encode_second.1.bias"]);
        _finalW = F32(w[$"{p}final_proj.weight"]); _finalB = F32(w[$"{p}final_proj.bias"]);
        // Upstream names the PatchEmbed module `camera_in` (so within the camera_in.* prefix it is camera_in.camera_in.proj.*).
        _patchW_ = F32(w[$"{p}camera_in.proj.weight"]); _patchB = F32(w[$"{p}camera_in.proj.bias"]);
        _scale = F32(w[$"{p}scale"]); // nn.Parameter(torch.ones(1)) — a single scalar, not per-channel
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] all = [_enc1W, _enc1B, _gn1W, _gn1B, _enc2W, _enc2B, _gn2W, _gn2B, _finalW, _finalB, _patchW_, _patchB, _scale];
        foreach (Tensor? t in all) if (t is not null) yield return t;
    }

    /// <summary>Encodes Plücker maps <c>[B, T, 6, H, W]</c> into camera tokens <c>[B, tLat·(H/(8·pH))·(W/(8·pW)), hidden]</c>.</summary>
    public Tensor Forward(IBackend backend, Tensor plucker)
    {
        if (plucker.Shape.Rank != 5 || plucker.Shape[2] != PluckerCh)
            throw new ArgumentException($"plucker must be [B,T,6,H,W]; got {plucker.Shape}.", nameof(plucker));
        int b = (int)plucker.Shape[0], t = (int)plucker.Shape[1], h = (int)plucker.Shape[3], w = (int)plucker.Shape[4];
        int r = _downscale, h8 = h / r, w8 = w / r, n = b * t;
        int unCh = PluckerCh * r * r;

        // Upstream order (cameranet.py forward): unshuffle → encode_first → compress_time → encode_second →
        // compress_time → final_proj → camera_in (PatchEmbed) → *scale. The two factor-2 compress_time calls sit
        // *between* the nonlinear (GroupNorm+ReLU) encode blocks, so temporal pooling cannot be deferred to the end.
        // 1. PixelUnshuffle → [N, unCh, h8, w8].
        Tensor un = new(new TensorShape([(long)n, unCh, h8, w8]), DType.F32);
        PixelUnshuffle(plucker, un, b, t, h, w, r);

        // 2. conv 384→192 + GN + ReLU.
        Tensor x = Conv1x1(backend, un, _enc1W!, _enc1B!, n, h8, w8); un.Dispose();
        GroupNormRelu(backend, ref x, _gn1W!, _gn1B!);
        // 3. compress_time #1 (factor 2, first frame preserved).
        (Tensor xc, int t1) = CompressTimeFactor2(x, b, t, 192, h8, w8); x.Dispose();
        // 4. conv 192→96 + GN + ReLU.
        Tensor x2 = Conv1x1(backend, xc, _enc2W!, _enc2B!, b * t1, h8, w8); xc.Dispose();
        GroupNormRelu(backend, ref x2, _gn2W!, _gn2B!);
        // 5. compress_time #2 (factor 2).
        (Tensor x2c, int tLat) = CompressTimeFactor2(x2, b, t1, 96, h8, w8); x2.Dispose();
        // 6. conv 96→16 → [B·tLat, 16, h8, w8] == [B, tLat, 16, h8, w8] contiguous.
        Tensor feat = Conv1x1(backend, x2c, _finalW!, _finalB!, b * tLat, h8, w8); x2c.Dispose();

        // 7. patch embed [1,pH,pW] → tokens [B, tLat*hOut*wOut, hidden].
        int hOut = h8 / _patchH, wOut = w8 / _patchW;
        int sImg = tLat * hOut * wOut, patchVec = _outCh * _patchH * _patchW;
        Tensor patches = new(new TensorShape(b, sImg, patchVec), DType.F32);
        PatchEmbedGather(feat, patches, b, tLat, _outCh, h8, w8, _patchH, _patchW, hOut, wOut); feat.Dispose();
        Tensor tokens = new(new TensorShape(b, sImg, _hidden), DType.F32);
        backend.Linear(tokens, patches, _patchW_!, _patchB!); patches.Dispose();

        // 8. learnable scalar scale (torch.ones(1)).
        float scale = ((float*)_scale!.DataPointer)[0];
        float* tp = (float*)tokens.DataPointer;
        for (long i = 0; i < (long)b * sImg * _hidden; i++) tp[i] *= scale;
        return tokens;
    }

    private static Tensor Conv1x1(IBackend backend, Tensor input, Tensor weight, Tensor bias, int n, int h, int w)
    {
        int cout = (int)weight.Shape[0];
        Tensor outp = new(new TensorShape([(long)n, cout, h, w]), DType.F32);
        backend.Conv2D(outp, input, weight, bias, 1, 1, 0, 0);
        return outp;
    }

    private void GroupNormRelu(IBackend backend, ref Tensor x, Tensor gw, Tensor gb)
    {
        Tensor gn = new(x.Shape, DType.F32);
        backend.GroupNorm(gn, x, gw, gb, _gnGroups, 1e-6f);
        x.Dispose();
        Tensor relu = new(gn.Shape, DType.F32);
        backend.Clamp(relu, gn, 0f, float.MaxValue);
        gn.Dispose();
        x = relu;
    }

    private static void PixelUnshuffle(Tensor src, Tensor dst, int b, int t, int h, int w, int r)
    {
        float* sp = (float*)src.DataPointer; float* dp = (float*)dst.DataPointer;
        int h8 = h / r, w8 = w / r, unCh = PluckerCh * r * r;
        for (int n = 0; n < b * t; n++)
            for (int c = 0; c < PluckerCh; c++)
                for (int ph = 0; ph < r; ph++) for (int pw = 0; pw < r; pw++)
                {
                    int oc = c * r * r + ph * r + pw;
                    for (int y = 0; y < h8; y++) for (int x = 0; x < w8; x++)
                    {
                        long s = (((long)n * PluckerCh + c) * h + (y * r + ph)) * w + (x * r + pw);
                        long d = (((long)n * unCh + oc) * h8 + y) * w8 + x;
                        dp[d] = sp[s];
                    }
                }
    }

    /// <summary>One upstream <c>compress_time</c> pass (factor-2 temporal pooling). Input/output are flattened
    /// <c>[B·F, C, H, W]</c>; pooling is along the frame dim per (B,C,H,W). Mirrors cameranet.py: 66/34-frame inputs
    /// split into two clips (each keeps its first frame and avg-pools the rest by 2); odd inputs keep frame 0 and
    /// avg-pool the rest by 2; even inputs avg-pool by 2 straight. Returns the pooled tensor and the new frame count.</summary>
    private static (Tensor Pooled, int NewFrames) CompressTimeFactor2(Tensor feat, int b, int t, int c, int h, int w)
    {
        List<int[]> groups = BuildPoolGroups(t);
        int newF = groups.Count;
        Tensor outT = new(new TensorShape([(long)b * newF, c, h, w]), DType.F32);
        float* fp = (float*)feat.DataPointer; float* op = (float*)outT.DataPointer;
        long frame = (long)c * h * w;
        for (int bb = 0; bb < b; bb++)
            for (int of = 0; of < newF; of++)
            {
                int[] g = groups[of];
                long dst = ((long)bb * newF + of) * frame;
                float inv = 1f / g.Length;
                for (long e = 0; e < frame; e++)
                {
                    float acc = 0;
                    foreach (int tf in g) acc += fp[((long)bb * t + tf) * frame + e];
                    op[dst + e] = acc * inv;
                }
            }
        return (outT, newF);
    }

    /// <summary>Builds the input-frame groups averaged into each output frame for a single <c>compress_time</c> pass.</summary>
    private static List<int[]> BuildPoolGroups(int t)
    {
        List<int[]> groups = [];
        if (t == 66 || t == 34)
        {
            int half = t / 2;
            AppendClip(groups, 0, half);      // clip 1: [0, half)
            AppendClip(groups, half, half);   // clip 2: [half, t)
        }
        else if ((t & 1) == 1)
        {
            AppendClip(groups, 0, t);         // odd: keep frame 0, avg-pool the rest by 2
        }
        else
        {
            for (int j = 0; j < t / 2; j++) groups.Add([2 * j, 2 * j + 1]); // even: straight avg-pool by 2
        }
        return groups;
    }

    /// <summary>Appends one clip's groups starting at <paramref name="offset"/> spanning <paramref name="len"/> frames:
    /// the first frame is preserved, the remaining (len-1) frames are average-pooled by 2 (kernel 2, stride 2; a
    /// trailing odd frame is dropped, matching <c>F.avg_pool1d</c>).</summary>
    private static void AppendClip(List<int[]> groups, int offset, int len)
    {
        groups.Add([offset]);
        int pooled = (len - 1) / 2; // avg_pool1d(kernel=2, stride=2) over the (len-1) trailing frames; drops a trailing odd frame
        for (int j = 0; j < pooled; j++) groups.Add([offset + 1 + 2 * j, offset + 2 + 2 * j]);
    }

    private static void PatchEmbedGather(Tensor comp, Tensor patches, int b, int tLat, int c, int h8, int w8,
        int pH, int pW, int hOut, int wOut)
    {
        float* cp = (float*)comp.DataPointer; float* pp = (float*)patches.DataPointer;
        int patchVec = c * pH * pW;
        long frame = (long)c * h8 * w8;
        for (int bb = 0; bb < b; bb++)
            for (int lf = 0; lf < tLat; lf++) for (int hi = 0; hi < hOut; hi++) for (int wi = 0; wi < wOut; wi++)
            {
                long token = ((long)lf * hOut + hi) * wOut + wi;
                long dstBase = ((long)bb * (tLat * hOut * wOut) + token) * patchVec;
                int idx = 0;
                for (int cc = 0; cc < c; cc++) for (int kh = 0; kh < pH; kh++) for (int kw = 0; kw < pW; kw++)
                {
                    long src = ((long)bb * tLat + lf) * frame + ((long)cc * h8 + (hi * pH + kh)) * w8 + (wi * pW + kw);
                    pp[dstBase + idx++] = cp[src];
                }
            }
    }

    private static Tensor F32(Tensor t) => t.DType != DType.F32 ? t.CastTo(DType.F32) : t;
}
