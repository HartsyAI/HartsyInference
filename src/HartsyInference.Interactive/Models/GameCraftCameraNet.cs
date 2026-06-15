using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Interactive.Camera;

namespace HartsyInference.Interactive.Models;

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
    private readonly int _hidden, _downscale, _patchH, _patchW, _outCh, _temporalComp, _gnGroups;

    private Tensor? _enc1W, _enc1B, _gn1W, _gn1B;   // conv 384→192
    private Tensor? _enc2W, _enc2B, _gn2W, _gn2B;   // conv 192→96
    private Tensor? _finalW, _finalB;               // conv 96→16
    private Tensor? _patchW_, _patchB;              // [hidden, 16*pH*pW]
    private Tensor? _scale;                         // [hidden]

    public GameCraftCameraNet(int hiddenSize, int downscale = 8, int outChannels = 16, int patchH = 2, int patchW = 2,
        int temporalCompression = 4, int gnGroups = 2)
    {
        _hidden = hiddenSize; _downscale = downscale; _outCh = outChannels;
        _patchH = patchH; _patchW = patchW; _temporalComp = temporalCompression; _gnGroups = gnGroups;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w, string prefix = "camera_in")
    {
        string p = prefix.Length == 0 ? "" : prefix + ".";
        _enc1W = F32(w[$"{p}encode_first.0.weight"]); _enc1B = F32(w[$"{p}encode_first.0.bias"]);
        _gn1W = F32(w[$"{p}encode_first.1.weight"]); _gn1B = F32(w[$"{p}encode_first.1.bias"]);
        _enc2W = F32(w[$"{p}encode_second.0.weight"]); _enc2B = F32(w[$"{p}encode_second.0.bias"]);
        _gn2W = F32(w[$"{p}encode_second.1.weight"]); _gn2B = F32(w[$"{p}encode_second.1.bias"]);
        _finalW = F32(w[$"{p}final_proj.weight"]); _finalB = F32(w[$"{p}final_proj.bias"]);
        _patchW_ = F32(w[$"{p}patch_embed.proj.weight"]); _patchB = F32(w[$"{p}patch_embed.proj.bias"]);
        _scale = F32(w[$"{p}scale"]);
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

        // 1. PixelUnshuffle → [N, unCh, h8, w8].
        Tensor un = new(new TensorShape([(long)n, unCh, h8, w8]), DType.F32);
        PixelUnshuffle(plucker, un, b, t, h, w, r);

        // 2. conv 384→192 + GN + ReLU.
        Tensor x = Conv1x1(backend, un, _enc1W!, _enc1B!, n, h8, w8); un.Dispose();
        GroupNormRelu(backend, ref x, _gn1W!, _gn1B!);
        // 3. conv 192→96 + GN + ReLU.
        Tensor x2 = Conv1x1(backend, x, _enc2W!, _enc2B!, n, h8, w8); x.Dispose();
        GroupNormRelu(backend, ref x2, _gn2W!, _gn2B!);
        // 4. conv 96→16.
        Tensor feat = Conv1x1(backend, x2, _finalW!, _finalB!, n, h8, w8); x2.Dispose(); // [N,16,h8,w8]

        // 5. temporal compression T → tLat.
        int tLat = (t - 1) / _temporalComp + 1;
        Tensor comp = TemporalCompress(feat, b, t, _outCh, h8, w8, tLat); feat.Dispose(); // [B,tLat,16,h8,w8]

        // 6. patch embed [1,pH,pW] → tokens [B, tLat*hOut*wOut, hidden].
        int hOut = h8 / _patchH, wOut = w8 / _patchW;
        int sImg = tLat * hOut * wOut, patchVec = _outCh * _patchH * _patchW;
        Tensor patches = new(new TensorShape(b, sImg, patchVec), DType.F32);
        PatchEmbedGather(comp, patches, b, tLat, _outCh, h8, w8, _patchH, _patchW, hOut, wOut); comp.Dispose();
        Tensor tokens = new(new TensorShape(b, sImg, _hidden), DType.F32);
        backend.Linear(tokens, patches, _patchW_!, _patchB!); patches.Dispose();

        // 7. learnable per-channel scale.
        float* tp = (float*)tokens.DataPointer; float* sc = (float*)_scale!.DataPointer;
        for (long i = 0; i < (long)b * sImg; i++) for (int c = 0; c < _hidden; c++) tp[i * _hidden + c] *= sc[c];
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

    private static Tensor TemporalCompress(Tensor feat, int b, int t, int c, int h, int w, int tLat)
    {
        // Causal grouping: latent 0 = input frame 0; latent lf>=1 = mean of input frames [1+(lf-1)*4 .. 1+lf*4-1].
        Tensor outT = new(new TensorShape([(long)b, tLat, c, h, w]), DType.F32);
        float* fp = (float*)feat.DataPointer; float* op = (float*)outT.DataPointer;
        long frame = (long)c * h * w;
        int comp = (t - 1) <= 0 ? 1 : (t - 1) / Math.Max(1, tLat - 1 == 0 ? 1 : tLat - 1);
        comp = Math.Max(1, comp);
        for (int bb = 0; bb < b; bb++)
            for (int lf = 0; lf < tLat; lf++)
            {
                int start = lf == 0 ? 0 : 1 + (lf - 1) * comp;
                int end = lf == 0 ? 1 : Math.Min(t, 1 + lf * comp);
                int count = Math.Max(1, end - start);
                long dst = ((long)bb * tLat + lf) * frame;
                for (long e = 0; e < frame; e++)
                {
                    float acc = 0;
                    for (int tf = start; tf < end; tf++) acc += fp[((long)bb * t + tf) * frame + e];
                    op[dst + e] = acc / count;
                }
            }
        return outT;
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
