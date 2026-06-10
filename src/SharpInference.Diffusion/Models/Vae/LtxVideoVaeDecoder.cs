using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;

namespace SharpInference.Diffusion.Models.Vae;

/// <summary>LTX-Video VAE decoder (<c>LTXVideoDecoder3d</c> in <c>AutoencoderKLLTXVideo</c>), ported from diffusers for the base LTX-Video 0.9 config: <c>conv_in</c> → mid resnets → up blocks (optional channel-change <c>conv_in</c> + pixel-shuffle upsampler + resnets) → channel-RMS <c>norm_out</c> → SiLU → <c>conv_out</c> → spatial pixel-unshuffle (patch 4). Reuses <see cref="LtxVaeResnetBlock3d"/>/<see cref="LtxVaeUpsampler3d"/>/<see cref="CausalConv3d"/>.
///
/// <para>The released 0.9 VAE config has <c>timestep_conditioning=False</c> and <c>decoder_causal=False</c> (non-causal, symmetric edge-replicate temporal padding) — so this is a plain decoder (no decode timestep, no <c>feat_cache</c>). Output frames = <c>(T_lat − 1)·8 + 1</c>; spatial = <c>latent · 32</c>. The timestep-conditioned 0.9.5 variant is a follow-up (the block <see cref="LtxVaeResnetBlock3d"/> already supports it). Numerics vs the real checkpoint are validation-pending.</para></summary>
public sealed unsafe class LtxVideoVaeDecoder
{
    private sealed class UpStage
    {
        public LtxVaeResnetBlock3d? ConvIn;     // channel change (in != out)
        public LtxVaeUpsampler3d? Upsampler;    // spatio-temporal ×2
        public LtxVaeResnetBlock3d[] Resnets = [];
    }

    private readonly int _latentChannels;
    private readonly int _outChannels;
    private readonly int _patch;
    private readonly bool _isCausal;
    private readonly int[] _blockOutRev;        // reversed block_out_channels
    private readonly bool[] _scalingRev;        // reversed spatio_temporal_scaling
    private readonly int[] _layersRev;          // reversed layers_per_block

    private CausalConv3d? _convIn, _convOut;
    private LtxVaeResnetBlock3d[] _midResnets = [];
    private UpStage[] _upStages = [];
    private int _lastChannel;                   // channels into conv_out

    public LtxVideoVaeDecoder(
        int latentChannels = 128, int outChannels = 3,
        int[]? blockOutChannels = null, bool[]? spatioTemporalScaling = null, int[]? layersPerBlock = null,
        int patchSize = 4, bool isCausal = false)
    {
        _latentChannels = latentChannels;
        _outChannels = outChannels;
        _patch = patchSize;
        _isCausal = isCausal;
        blockOutChannels ??= [128, 256, 512, 512];
        spatioTemporalScaling ??= [true, true, true, false];
        layersPerBlock ??= [4, 3, 3, 3, 4];
        _blockOutRev = Reverse(blockOutChannels);
        _scalingRev = Reverse(spatioTemporalScaling);
        _layersRev = Reverse(layersPerBlock);
    }

    /// <summary>Output frames for a given latent frame count: <c>(T_lat − 1)·8 + 1</c> (3 temporal upsamplers ×2 each).</summary>
    public int OutputFrames(int latentFrames)
    {
        int t = latentFrames;
        foreach (bool s in _scalingRev) if (s) t = t * 2 - 1;
        return t;
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        int output = _blockOutRev[0];
        _convIn = new CausalConv3d(w["decoder.conv_in.conv.weight"], Bias(w, "decoder.conv_in.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);

        _midResnets = new LtxVaeResnetBlock3d[_layersRev[0]];
        for (int j = 0; j < _midResnets.Length; j++)
        {
            _midResnets[j] = new LtxVaeResnetBlock3d(output, output, timestepCond: false, isCausal: _isCausal);
            _midResnets[j].LoadWeights(w, $"decoder.mid_block.resnets.{j}");
        }

        _upStages = new UpStage[_blockOutRev.Length];
        for (int i = 0; i < _blockOutRev.Length; i++)
        {
            int inC = output;
            int outC = _blockOutRev[i];
            UpStage stage = new();
            string p = $"decoder.up_blocks.{i}";
            if (inC != outC)
            {
                stage.ConvIn = new LtxVaeResnetBlock3d(inC, outC, timestepCond: false, isCausal: _isCausal);
                stage.ConvIn.LoadWeights(w, $"{p}.conv_in");
            }
            if (_scalingRev[i])
            {
                stage.Upsampler = new LtxVaeUpsampler3d(outC, (2, 2, 2), upscaleFactor: 1, residual: false, isCausal: _isCausal);
                stage.Upsampler.LoadWeights(w, $"{p}.upsamplers.0");
            }
            stage.Resnets = new LtxVaeResnetBlock3d[_layersRev[i + 1]];
            for (int j = 0; j < stage.Resnets.Length; j++)
            {
                stage.Resnets[j] = new LtxVaeResnetBlock3d(outC, outC, timestepCond: false, isCausal: _isCausal);
                stage.Resnets[j].LoadWeights(w, $"{p}.resnets.{j}");
            }
            _upStages[i] = stage;
            output = outC;
        }

        _lastChannel = output;
        _convOut = new CausalConv3d(w["decoder.conv_out.conv.weight"], Bias(w, "decoder.conv_out.conv.bias"),
            padT: 1, padH: 1, padW: 1, replicateFirstPad: true, causal: _isCausal);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_convIn is not null) foreach (Tensor t in _convIn.EnumerateWeights()) yield return t;
        foreach (LtxVaeResnetBlock3d r in _midResnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        foreach (UpStage s in _upStages)
        {
            if (s.ConvIn is not null) foreach (Tensor t in s.ConvIn.EnumerateWeights()) yield return t;
            if (s.Upsampler is not null) foreach (Tensor t in s.Upsampler.EnumerateWeights()) yield return t;
            foreach (LtxVaeResnetBlock3d r in s.Resnets) foreach (Tensor t in r.EnumerateWeights()) yield return t;
        }
        if (_convOut is not null) foreach (Tensor t in _convOut.EnumerateWeights()) yield return t;
    }

    /// <summary>Decodes a latent <c>[1, latentChannels, T_lat, H, W]</c> to RGB <c>[1, 3, (T_lat−1)·8+1, H·32, W·32]</c> in [-1,1]. (The base config has identity latent normalization; scaling_factor 1.0.)</summary>
    public Tensor Decode(IBackend backend, Tensor latent)
    {
        if ((int)latent.Shape[1] != _latentChannels)
            throw new ArgumentException($"latent channels {latent.Shape[1]} != {_latentChannels}.", nameof(latent));

        Tensor h = _convIn!.Forward(backend, latent);
        foreach (LtxVaeResnetBlock3d r in _midResnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }

        foreach (UpStage s in _upStages)
        {
            if (s.ConvIn is not null) { Tensor n = s.ConvIn.Forward(backend, h, null); h.Dispose(); h = n; }
            if (s.Upsampler is not null) { Tensor n = s.Upsampler.Forward(backend, h); h.Dispose(); h = n; }
            foreach (LtxVaeResnetBlock3d r in s.Resnets) { Tensor n = r.Forward(backend, h, null); h.Dispose(); h = n; }
        }

        Tensor normed = ChannelRms(h, _lastChannel);
        h.Dispose();
        backend.Silu(normed, normed);
        Tensor patched = _convOut!.Forward(backend, normed);   // [1, outChannels·p², F, 8H, 8W]
        normed.Dispose();

        Tensor rgb = PixelUnshuffle(patched);
        patched.Dispose();
        return rgb;
    }

    /// <summary>Spatial pixel-unshuffle (patch p): <c>[1, oc·p², F, H, W] → [1, oc, F, H·p, W·p]</c>, matching the upstream reshape/permute (channel = <c>oc·p² + p_a·p + p_b</c>, with p_b→H, p_a→W).</summary>
    private Tensor PixelUnshuffle(Tensor x)
    {
        int b = (int)x.Shape[0], srcC = (int)x.Shape[1], f = (int)x.Shape[2], h = (int)x.Shape[3], w = (int)x.Shape[4];
        int p = _patch;
        int oc = srcC / (p * p);
        int outH = h * p, outW = w * p;
        Tensor outT = new Tensor(new TensorShape([(long)b, oc, f, outH, outW]), DType.F32);
        float* sp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        long srcFrame = (long)h * w, dstFrame = (long)outH * outW;
        for (int bi = 0; bi < b; bi++)
            for (int c = 0; c < oc; c++)
                for (int fi = 0; fi < f; fi++)
                    for (int ho = 0; ho < outH; ho++)
                    {
                        int hi = ho / p, pb = ho % p;
                        for (int wo = 0; wo < outW; wo++)
                        {
                            int wi = wo / p, pa = wo % p;
                            int ch = oc > 0 ? (c * p + pa) * p + pb : 0;
                            long srcOff = (((long)bi * srcC + ch) * f + fi) * srcFrame + (long)hi * w + wi;
                            long dstOff = (((long)bi * oc + c) * f + fi) * dstFrame + (long)ho * outW + wo;
                            op[dstOff] = sp[srcOff];
                        }
                    }
        return outT;
    }

    private static Tensor ChannelRms(Tensor x, int c)
    {
        int b = (int)x.Shape[0];
        long spatial = x.Shape.ElementCount / ((long)b * c);
        Tensor outT = new Tensor(x.Shape, DType.F32);
        float* xp = (float*)x.DataPointer;
        float* op = (float*)outT.DataPointer;
        for (int bi = 0; bi < b; bi++)
            for (long s = 0; s < spatial; s++)
            {
                long basePos = (long)bi * c * spatial + s;
                double sum = 0;
                for (int ci = 0; ci < c; ci++) { float v = xp[basePos + (long)ci * spatial]; sum += (double)v * v; }
                float inv = 1f / MathF.Sqrt((float)(sum / c) + 1e-8f);
                for (int ci = 0; ci < c; ci++) { long off = basePos + (long)ci * spatial; op[off] = xp[off] * inv; }
            }
        return outT;
    }

    private static int[] Reverse(int[] a) { int[] r = (int[])a.Clone(); Array.Reverse(r); return r; }
    private static bool[] Reverse(bool[] a) { bool[] r = (bool[])a.Clone(); Array.Reverse(r); return r; }
    private static Tensor? Bias(IReadOnlyDictionary<string, Tensor> w, string k) => w.TryGetValue(k, out Tensor? b) ? b : null;
}
