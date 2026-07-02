using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>HunyuanVideo 3D causal VAE decoder (<c>AutoencoderKLHunyuanVideo</c> decode path), used by
/// Kandinsky 5.0 T2V/I2V. Assembled from the shared primitives: <see cref="CausalConv3d"/> (replicate
/// padding mode), <see cref="HunyuanVideoResnetBlock3d"/>, <see cref="HunyuanVideoMidBlock3d"/>
/// (frame-causal attention via <see cref="VaeAttention.Forward3D"/>), and <see cref="Vae3dLayout"/>.
///
/// <para>Decode: <c>post_quant_conv(16→16) → conv_in(16→512) → mid → 4 up stages (3 resnets each;
/// upsamplers: stage 0 spatial-only, stages 1–2 spatial+temporal, stage 3 none) → GroupNorm+SiLU →
/// conv_out(128→3)</c>. The upsampler treats frame 0 specially (spatial-only nearest), so
/// <c>T → 1 + 2·(T−1)</c> per temporal stage and <c>F = (T_lat−1)·4 + 1</c> overall. Latent scaling
/// (÷0.476986) is the caller's responsibility. <b>Untiled</b> — large clips need the diffusers-style
/// tiling follow-up before GPU-scale decodes; numerics are validation-pending vs the real checkpoint.</para></summary>
public sealed class HunyuanVideoVaeDecoder
{
    private readonly HunyuanVideoVaeConfig _config;

    private CausalConv3d? _postQuantConv;
    private CausalConv3d? _convIn;
    private HunyuanVideoMidBlock3d? _mid;
    private UpStage[] _stages = [];
    private Tensor? _normOutWeight, _normOutBias;
    private CausalConv3d? _convOut;

    private sealed class UpStage
    {
        public required HunyuanVideoResnetBlock3d[] Resnets;
        public required bool Spatial;
        public required bool Temporal;
        public CausalConv3d? UpsampleConv;
    }

    public HunyuanVideoVaeDecoder(HunyuanVideoVaeConfig? config = null)
    {
        _config = config ?? HunyuanVideoVaeConfig.Default;
    }

    /// <summary>The configuration this decoder was built with.</summary>
    public HunyuanVideoVaeConfig Config => _config;

    /// <summary>Loads weights keyed <c>post_quant_conv.*</c> + <c>decoder.*</c> (diffusers state-dict names).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int[] channels = _config.BlockOutChannels;
        int last = channels[^1];

        _postQuantConv = new CausalConv3d(w["post_quant_conv.weight"], Bias(w, "post_quant_conv.bias"));
        _convIn = HunyuanVideoVaeKeys.Conv(w, "decoder.conv_in", padT: 1, padH: 1, padW: 1);

        _mid = new HunyuanVideoMidBlock3d(last, _config.MidBlockAttention, _config.NormGroups, _config.NormEps);
        _mid.LoadWeights(w, "decoder.mid_block");

        int numStages = channels.Length;
        int resnetsPerStage = _config.LayersPerBlock + 1;
        _stages = new UpStage[numStages];
        int prevOut = last;
        for (int i = 0; i < numStages; i++)
        {
            int outCh = channels[numStages - 1 - i];
            HunyuanVideoResnetBlock3d[] resnets = new HunyuanVideoResnetBlock3d[resnetsPerStage];
            int cur = prevOut;
            for (int j = 0; j < resnetsPerStage; j++)
            {
                resnets[j] = new HunyuanVideoResnetBlock3d(cur, outCh, _config.NormGroups, _config.NormEps);
                resnets[j].LoadWeights(w, $"decoder.up_blocks.{i}.resnets.{j}");
                cur = outCh;
            }

            bool spatial = _config.StageHasSpatialResample(i);
            bool temporal = _config.StageHasTemporalResample(i);
            UpStage stage = new UpStage { Resnets = resnets, Spatial = spatial, Temporal = temporal };
            if (spatial || temporal)
                stage.UpsampleConv = HunyuanVideoVaeKeys.Conv(w, $"decoder.up_blocks.{i}.upsamplers.0.conv",
                    padT: 1, padH: 1, padW: 1);
            _stages[i] = stage;
            prevOut = outCh;
        }

        _normOutWeight = w["decoder.conv_norm_out.weight"];
        _normOutBias   = w["decoder.conv_norm_out.bias"];
        _convOut = HunyuanVideoVaeKeys.Conv(w, "decoder.conv_out", padT: 1, padH: 1, padW: 1);
    }

    /// <summary>Enumerates all weights for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_postQuantConv is not null) foreach (Tensor t in _postQuantConv.EnumerateWeights()) yield return t;
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        if (_mid is not null) foreach (Tensor t in _mid.EnumerateWeights()) yield return t;
        foreach (UpStage s in _stages)
        {
            foreach (HunyuanVideoResnetBlock3d r in s.Resnets)
                foreach (Tensor t in r.EnumerateWeights()) yield return t;
            if (s.UpsampleConv is not null) foreach (Tensor t in s.UpsampleConv.EnumerateWeights()) yield return t;
        }
        if (_normOutWeight is not null) yield return _normOutWeight;
        if (_normOutBias is not null) yield return _normOutBias;
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes a raw VAE-space latent <c>[B, 16, T_lat, H_lat, W_lat]</c> (already divided by the scaling factor) → RGB <c>[B, 3, (T_lat−1)·4+1, H_lat·8, W_lat·8]</c> in [−1, 1]. <paramref name="trimStages"/> keeps the per-stage memory-pool trim that bounds an UNTILED full-res decode's peak; <see cref="DecodeTiled"/> passes false — a tile's working set is small, and each trim costs a sync + multi-GB driver release/re-map.</summary>
    public Tensor Decode(IBackend backend, Tensor latent, bool trimStages = true)
    {
        if (latent.Shape.Rank != 5 || (int)latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException(
                $"Expected latent [B, {_config.LatentChannels}, T, H, W]; got {latent.Shape}.", nameof(latent));

        Stat(backend, "latent_in", latent);
        Tensor x = _postQuantConv!.Forward(backend, latent);
        Stat(backend, "post_quant", x);
        Tensor h = _convIn!.Forward(backend, x);
        Stat(backend, "conv_in", h);
        x.Dispose();

        Tensor cur = _mid!.Forward(backend, h);
        Stat(backend, "mid", cur);
        h.Dispose();
        // The stream-ordered mempool (cuMemAllocAsync) only recycles a freed block once its cuMemFreeAsync is
        // processed — which needs a stream sync. Without one, a full-res decode's pool grows to the SUM of every
        // intermediate (>24 GB at 512×320×25f) instead of the per-stage working set. Trim after each stage (sync +
        // cuMemPoolTrimTo, correctness-neutral, no-op off-CUDA) so peak stays bounded to one stage. See DecodeTiled.
        if (trimStages) backend.TrimMemoryPool();

        int si = 0;
        foreach (UpStage stage in _stages)
        {
            int ri = 0;
            foreach (HunyuanVideoResnetBlock3d resnet in stage.Resnets)
            {
                Tensor next = resnet.Forward(backend, cur);
                cur.Dispose();
                cur = next;
                Stat(backend, $"up{si}.resnet{ri++}", cur);
            }
            if (stage.UpsampleConv is not null)
            {
                Tensor up = Upsample(backend, cur, stage.Temporal, stage.Spatial, stage.UpsampleConv);
                cur.Dispose();
                cur = up;
                Stat(backend, $"up{si}.upsample", cur);
            }
            if (trimStages) backend.TrimMemoryPool();
            si++;
        }

        Tensor normed = HunyuanVideoVaeKeys.GroupNormSilu3d(backend, cur, _normOutWeight!, _normOutBias!,
            _config.NormGroups, _config.NormEps);
        Stat(backend, "norm_out", normed);
        cur.Dispose();
        Tensor rgb = _convOut!.Forward(backend, normed);
        Stat(backend, "conv_out(rgb)", rgb);
        normed.Dispose();
        return rgb;
    }

    /// <summary>Memory-bounded spatial-tiled decode (diffusers <c>tiled_decode</c> analogue). Splits the latent
    /// into overlapping <paramref name="tileLatent"/>×<paramref name="tileLatent"/> spatial tiles (stride =
    /// <c>tileLatent·(1−overlapFactor)</c>), decodes each independently, and feather-blends the pixel tiles into a
    /// single canvas so the full-res peak (>24 GB at 512×320×25f) drops to a per-tile working set. Because GroupNorm
    /// statistics are computed per tile this is a close approximation of <see cref="Decode"/> (not bit-identical),
    /// exactly as in diffusers; the feathered overlap hides tile seams. Falls back to <see cref="Decode"/> when the
    /// latent already fits one tile.</summary>
    public unsafe Tensor DecodeTiled(IBackend backend, Tensor latent, int tileLatent = 24, float overlapFactor = 0.25f)
    {
        if (latent.Shape.Rank != 5 || (int)latent.Shape[1] != _config.LatentChannels)
            throw new ArgumentException($"Expected latent [B, {_config.LatentChannels}, T, H, W]; got {latent.Shape}.", nameof(latent));
        int C = (int)latent.Shape[1], Tl = (int)latent.Shape[2], Hl = (int)latent.Shape[3], Wl = (int)latent.Shape[4];
        if (Hl <= tileLatent && Wl <= tileLatent) return Decode(backend, latent);

        int sf = _config.SpatialCompression;
        int stride = Math.Max(1, (int)(tileLatent * (1f - overlapFactor)));
        int blendPx = (tileLatent - stride) * sf;
        List<int> iS = TileStarts(Hl, tileLatent, stride), jS = TileStarts(Wl, tileLatent, stride);
        int Hpx = Hl * sf, Wpx = Wl * sf;

        float* lp = (float*)latent.DataPointer;
        Tensor? outT = null;
        float[]? weight = null;
        int tout = 0, cout = 3;

        for (int ii = 0; ii < iS.Count; ii++)
        {
            int i0 = iS[ii], th = Math.Min(tileLatent, Hl - i0);
            bool topEdge = ii == 0, botEdge = ii == iS.Count - 1;
            for (int jj = 0; jj < jS.Count; jj++)
            {
                int j0 = jS[jj], tw = Math.Min(tileLatent, Wl - j0);
                bool leftEdge = jj == 0, rightEdge = jj == jS.Count - 1;

                Tensor rgb;
                using (Tensor tileL = new(new TensorShape([1L, C, Tl, th, tw]), DType.F32))
                {
                    float* tp = (float*)tileL.DataPointer;
                    for (int c = 0; c < C; c++)
                        for (int t = 0; t < Tl; t++)
                            for (int y = 0; y < th; y++)
                            {
                                float* src = lp + ((((long)c * Tl + t) * Hl + (i0 + y)) * Wl + j0);
                                float* dst = tp + (((long)c * Tl + t) * th + y) * tw;
                                for (int x = 0; x < tw; x++) dst[x] = src[x];
                            }
                    rgb = Decode(backend, tileL, trimStages: false);
                    backend.Sync();
                }

                if (outT is null)
                {
                    tout = (int)rgb.Shape[2];
                    outT = new Tensor(new TensorShape([1L, cout, tout, Hpx, Wpx]), DType.F32);
                    new Span<float>((float*)outT.DataPointer, checked((int)outT.Shape.ElementCount)).Clear();
                    weight = new float[(long)Hpx * Wpx];
                }

                int tph = th * sf, tpw = tw * sf;
                float* rp = (float*)rgb.DataPointer, op = (float*)outT.DataPointer;
                // Separable feather weights precomputed per axis; blend iterates (c, t) outermost so both the tile
                // read and the canvas write are row-sequential (the previous (y, x, c, t) order strided the whole
                // canvas per pixel — cache-hostile on a 25-frame full-res canvas).
                float[] wyA = new float[tph], wxA = new float[tpw];
                for (int y = 0; y < tph; y++) wyA[y] = Feather(y, tph, blendPx, topEdge, botEdge);
                for (int x = 0; x < tpw; x++) wxA[x] = Feather(x, tpw, blendPx, leftEdge, rightEdge);
                for (int y = 0; y < tph; y++)
                {
                    long wRow = (long)(i0 * sf + y) * Wpx + j0 * sf;
                    for (int x = 0; x < tpw; x++) weight![wRow + x] += wyA[y] * wxA[x];
                }
                for (int c = 0; c < cout; c++)
                    for (int t = 0; t < tout; t++)
                        for (int y = 0; y < tph; y++)
                        {
                            float wy = wyA[y];
                            float* srcRow = rp + (((long)c * tout + t) * tph + y) * tpw;
                            float* dstRow = op + (((long)c * tout + t) * Hpx + (i0 * sf + y)) * Wpx + j0 * sf;
                            for (int x = 0; x < tpw; x++) dstRow[x] += srcRow[x] * (wy * wxA[x]);
                        }
                rgb.Dispose();
                // The tile's rgb is now copied into the host canvas, so every activation from this tile's decode is
                // dead. The decoder's ops (mid/resnet/upsample) leave some intermediates cached-but-undisposed (freed
                // only at GC), and TrimMemoryPool can't reclaim those (the pool sees them as in-use) — so across many
                // tiles they accumulate to OOM. FreeActivations clears the whole activation cache, keeping peak flat
                // at one tile's working set. Safe here because DecodeTiled is a terminal decode stage: the host
                // latent/canvas aren't cached, and no caller-live activation crosses a tile boundary. trimPool:false —
                // tiles are identical, so the pool reservation is re-used; the single trim below returns it once.
                backend.FreeActivations(trimPool: false);
            }
        }
        backend.TrimMemoryPool();

        float* fop = (float*)outT!.DataPointer;
        for (int c = 0; c < cout; c++)
            for (int t = 0; t < tout; t++)
                for (int gy = 0; gy < Hpx; gy++)
                {
                    float* row = fop + (((long)c * tout + t) * Hpx + gy) * Wpx;
                    long wRow = (long)gy * Wpx;
                    for (int gx = 0; gx < Wpx; gx++)
                    {
                        float wgt = weight![wRow + gx];
                        if (wgt > 0f) row[gx] *= 1f / wgt;
                    }
                }
        return outT;
    }

    /// <summary>Diagnostic: logs mean/std/range of a decode intermediate (forces a D2H sync — debug only).</summary>
    private static unsafe void Stat(IBackend backend, string tag, Tensor t)
    {
        if (Environment.GetEnvironmentVariable("HYV_VAE_STAGES") != "1") return;
        backend.Sync();
        float* p = (float*)t.DataPointer; long n = t.Shape.ElementCount;
        double s = 0, s2 = 0; float mn = float.MaxValue, mx = float.MinValue; long nan = 0;
        for (long i = 0; i < n; i++) { float x = p[i]; if (float.IsNaN(x) || float.IsInfinity(x)) { nan++; continue; } s += x; s2 += (double)x * x; if (x < mn) mn = x; if (x > mx) mx = x; }
        double m = s / n, sd = Math.Sqrt(Math.Max(0, s2 / n - m * m));
        Logs.Info($"[VAE {tag,-16}] {t.Shape} mean={m:F4} std={sd:F4} range=[{mn:F3},{mx:F3}] NaN={nan}");
    }

    /// <summary>Tile start offsets along one axis: stride steps, with the final tile snapped to cover the edge.</summary>
    private static List<int> TileStarts(int dim, int tile, int stride)
    {
        List<int> l = [];
        if (dim <= tile) { l.Add(0); return l; }
        for (int s = 0; s < dim; s += stride) { l.Add(s); if (s + tile >= dim) break; }
        return l;
    }

    /// <summary>Separable feather weight: ramps 0→1 over the first <paramref name="blend"/> px and 1→0 over the last
    /// (skipping the ramp on a global-image edge so border pixels keep full weight). Overlapping ramps from adjacent
    /// tiles sum to a partition of unity after the weight-normalization pass.</summary>
    private static float Feather(int pos, int len, int blend, bool startEdge, bool endEdge)
    {
        float w = 1f;
        if (!startEdge && blend > 0 && pos < blend) w *= (pos + 0.5f) / blend;
        if (!endEdge && blend > 0 && pos >= len - blend) w *= (len - pos - 0.5f) / blend;
        return w < 1e-6f ? 1e-6f : w;
    }

    /// <summary>HunyuanVideo causal upsampler: frame 0 is spatially nearest-×2 only; frames 1..T−1 get the full nearest interpolation (incl. temporal ×2 when <paramref name="temporal"/>), then a k3 causal conv.</summary>
    private static Tensor Upsample(IBackend backend, Tensor x, bool temporal, bool spatial, CausalConv3d conv)
    {
        int b = (int)x.Shape[0], c = (int)x.Shape[1], t = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];

        Tensor spatialUp;
        if (spatial)
        {
            // GPU-resident layout ops (the host Vae3dLayout overloads read DataPointer → D2H sync per call).
            Tensor frames = Vae3dLayout.ToFrames(backend, x);
            Tensor upFrames = new Tensor(new TensorShape(b * t, c, h * 2, w * 2), DType.F32);
            backend.UpsampleNearest2D(upFrames, frames, 2, 2);
            frames.Dispose();
            spatialUp = Vae3dLayout.FromFrames(backend, upFrames, b, c, t, h * 2, w * 2);
            upFrames.Dispose();
        }
        else
        {
            spatialUp = Vae3dLayout.SliceFrames(backend, x, 0, t);   // contiguous copy so we can dispose uniformly
        }

        Tensor interp = spatialUp;
        if (temporal && t > 1)
        {
            List<Tensor> parts = new(1 + 2 * (t - 1));
            Tensor first = Vae3dLayout.SliceFrames(backend, spatialUp, 0, 1);
            parts.Add(first);
            for (int ti = 1; ti < t; ti++)
            {
                Tensor frame = Vae3dLayout.SliceFrames(backend, spatialUp, ti, 1);
                parts.Add(frame);
                parts.Add(frame);   // nearest temporal ×2 = each later frame repeated twice
            }
            interp = Vae3dLayout.ConcatFrames(backend, parts);
            // Repeated frames appear twice in the list; dispose each distinct tensor once.
            first.Dispose();
            for (int pi = 1; pi < parts.Count; pi += 2) parts[pi].Dispose();
            spatialUp.Dispose();
        }

        Tensor output = conv.Forward(backend, interp);
        interp.Dispose();
        return output;
    }

    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string key) =>
        w.TryGetValue(key, out Tensor? b) ? b : null;
}
