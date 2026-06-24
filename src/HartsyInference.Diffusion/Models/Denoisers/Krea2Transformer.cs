using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Krea 2 diffusion transformer (<c>Krea2Transformer2DModel</c>). Single-stream MMDiT flow-matching backbone:
/// a text-fusion stage collapses the tapped Qwen3-VL-4B hidden states, the result is concatenated with patchified
/// image latents into one <c>[text, image]</c> sequence, 28 sigmoid-gate GQA blocks (modulated by a shared timestep
/// vector + per-block tables) run over it, and the image tail is projected back to a velocity. See
/// <c>docs/Research/KREA2.md</c>.
///
/// <para>Reuses <see cref="FluxRope"/> (3-axis [32,48,48] θ=1000, text rows at position 0, image rows on the latent
/// grid — byte-compatible with Krea 2's FluxPosEmbed convention) and <see cref="DiTUtils"/>. The only Krea 2-specific
/// pieces are the sigmoid output-gate attention, the 6-way <c>scale_shift_table</c> modulation, and the text-fusion
/// stage (see <see cref="Krea2Block"/> / <see cref="Krea2TextFusion"/>).</para></summary>
public sealed unsafe class Krea2Transformer : IDisposable
{
    private readonly Krea2Config _config;
    private readonly FluxRope _rope;
    private readonly Krea2TextFusion _textFusion;
    private readonly Krea2Block[] _blocks;

    private Tensor? _imgInW, _imgInB;
    private Tensor? _time1W, _time1B, _time2W, _time2B;     // time_embed MLP
    private Tensor? _timeModW, _timeModB;                   // time_mod_proj
    private Tensor? _txtNormW, _txt1W, _txt1B, _txt2W, _txt2B; // txt_in (Krea2TextProjection)
    private Tensor? _finalTable, _finalNormW, _finalLinW, _finalLinB;

    private int _disposed;

    public Krea2Transformer(Krea2Config config)
    {
        _config = config;
        if (config.AxesDimRope[0] + config.AxesDimRope[1] + config.AxesDimRope[2] != config.AttentionHeadDim)
            throw new ArgumentException("sum(axes_dim_rope) must equal attention_head_dim.");
        _rope = new FluxRope(config.AxesDimRope, config.RopeTheta);
        _textFusion = new Krea2TextFusion(config.NumTextLayers, config.TextHiddenDim, config.TextNumHeads,
            config.TextNumKvHeads, config.TextIntermediateSize, config.NumLayerwiseTextBlocks,
            config.NumRefinerTextBlocks, config.NormEps);
        _blocks = new Krea2Block[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
            _blocks[i] = new Krea2Block(config.HiddenSize, config.IntermediateSize, config.NumAttentionHeads,
                config.NumKvHeads, config.NormEps);
    }

    public Krea2Config Config => _config;

    /// <summary>Loads weights from a diffusers-style key dict (bare keys).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        _imgInW = w["img_in.weight"]; _imgInB = w["img_in.bias"];
        _time1W = w["time_embed.linear_1.weight"]; _time1B = w["time_embed.linear_1.bias"];
        _time2W = w["time_embed.linear_2.weight"]; _time2B = w["time_embed.linear_2.bias"];
        _timeModW = w["time_mod_proj.weight"]; _timeModB = w["time_mod_proj.bias"];

        _txtNormW = Krea2Norm.LoadZeroCentered(w["txt_in.norm.weight"]);
        _txt1W = w["txt_in.linear_1.weight"]; _txt1B = w["txt_in.linear_1.bias"];
        _txt2W = w["txt_in.linear_2.weight"]; _txt2B = w["txt_in.linear_2.bias"];

        _finalTable = F32(w["final_layer.scale_shift_table"]);
        _finalNormW = Krea2Norm.LoadZeroCentered(w["final_layer.norm.weight"]);
        _finalLinW = w["final_layer.linear.weight"]; _finalLinB = w["final_layer.linear.bias"];

        _textFusion.LoadWeights(w, "text_fusion");
        for (int i = 0; i < _blocks.Length; i++) _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
    }

    /// <summary>Enumerates every weight tensor for GPU preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top =
        [
            _imgInW, _imgInB, _time1W, _time1B, _time2W, _time2B, _timeModW, _timeModB,
            _txtNormW, _txt1W, _txt1B, _txt2W, _txt2B, _finalTable, _finalNormW, _finalLinW, _finalLinB,
        ];
        foreach (Tensor? t in top) if (t is not null) yield return t;
        foreach (Tensor t in _textFusion.EnumerateWeights()) yield return t;
        foreach (Krea2Block b in _blocks) foreach (Tensor t in b.EnumerateWeights()) yield return t;
    }

    /// <summary>Predicts the flow-match velocity for one step.</summary>
    /// <param name="latent">Noisy latent <c>[1, 16, H, W]</c>.</param>
    /// <param name="timestep">Flow-match time <c>t</c> in <c>[0, 1]</c> (1 = noise, 0 = data).</param>
    /// <param name="encoderHidden">Tapped text hidden states <c>[1, txt_seq, numTextLayers · textHiddenDim]</c> (layer-major).</param>
    /// <returns>Velocity <c>[1, 16, H, W]</c>.</returns>
    public Tensor Forward(IBackend backend, Tensor latent, float timestep, Tensor encoderHidden)
    {
        ThrowIfDisposed();
        if (latent.Shape.Rank != 4 || latent.Shape[0] != 1)
            throw new ArgumentException($"latent must be [1, 16, H, W], got {latent.Shape}.", nameof(latent));

        int batch = 1;
        int channels = (int)latent.Shape[1];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int patch = _config.PatchSize;
        int hidden = _config.HiddenSize;
        int hPacked = latentH / patch;
        int wPacked = latentW / patch;
        int imgSeq = hPacked * wPacked;
        int txtSeq = (int)encoderHidden.Shape[1];

        // ── timestep embedding + shared modulation ──
        Tensor temb = ComputeTimeEmbedding(backend, timestep, batch, hidden);          // [B, hidden]
        Tensor tembGelu = new Tensor(temb.Shape, DType.F32);
        backend.Gelu(tembGelu, temb);
        Tensor tembMod = new Tensor(new TensorShape(batch, 6 * hidden), DType.F32);
        backend.Linear(tembMod, tembGelu, _timeModW!, _timeModB);                       // [B, 6·hidden]
        tembGelu.Dispose();

        // ── text: fusion → projection ──
        Tensor fused = _textFusion.Forward(backend, encoderHidden, batch, txtSeq);      // [B, S, textDim]
        Tensor txt = ApplyTxtIn(backend, fused, batch, txtSeq, hidden);                 // [B, S, hidden]
        fused.Dispose();

        // ── image: patchify + img_in ──
        Tensor imgFlat = PatchifyChannelOuter(latent, batch, channels, latentH, latentW, patch);
        Tensor img = new Tensor(new TensorShape(batch, imgSeq, hidden), DType.F32);
        backend.Linear(img, imgFlat, _imgInW!, _imgInB);
        imgFlat.Dispose();

        // ── concat [text, image]; RoPE for the joint sequence ──
        Tensor joint = DiTUtils.ConcatAlongSeqDim(txt, img);
        txt.Dispose(); img.Dispose();
        int jointSeq = txtSeq + imgSeq;
        Tensor posIds = FluxRope.BuildPositionIds(txtSeq, hPacked, wPacked);
        _rope.Precompute(posIds);
        posIds.Dispose();

        for (int i = 0; i < _blocks.Length; i++)
        {
            Tensor next = _blocks[i].Forward(backend, joint, tembMod, _rope, batch, jointSeq);
            joint.Dispose();
            joint = next;
        }
        tembMod.Dispose();

        // ── strip text prefix, final layer, unpatchify ──
        Tensor imgTail = SliceTail(joint, txtSeq, imgSeq, hidden);
        joint.Dispose();
        Tensor projected = ApplyFinalLayer(backend, imgTail, temb, batch, imgSeq, hidden);
        imgTail.Dispose();
        temb.Dispose();

        Tensor velocity = UnpatchifyChannelOuter(projected, batch, channels, hPacked, wPacked, patch);
        projected.Dispose();
        return velocity;
    }

    /// <summary>Sinusoidal(t·1000, cos-first, dim 256) → Linear → gelu_tanh → Linear. Returns <c>[B, hidden]</c>.</summary>
    private Tensor ComputeTimeEmbedding(IBackend backend, float timestep, int batch, int hidden)
    {
        int freqDim = _config.TimestepEmbedDim;
        Tensor sin = new Tensor(new TensorShape(batch, freqDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sin, timestep * 1000.0f, batch, freqDim);
        Tensor h1 = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(h1, sin, _time1W!, _time1B);
        sin.Dispose();
        Tensor act = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(outp, act, _time2W!, _time2B);
        act.Dispose();
        return outp;
    }

    /// <summary>txt_in: zero-center RMSNorm(textDim) → Linear(textDim→hidden) → gelu_tanh → Linear(hidden→hidden).</summary>
    private Tensor ApplyTxtIn(IBackend backend, Tensor fused, int batch, int seqLen, int hidden)
    {
        int textDim = _config.TextHiddenDim;
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, textDim), DType.F32);
        backend.RmsNorm(normed, fused, _txtNormW!, _config.NormEps);
        Tensor h1 = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Linear(h1, normed, _txt1W!, _txt1B);
        normed.Dispose();
        Tensor act = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Gelu(act, h1);
        h1.Dispose();
        Tensor outp = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.Linear(outp, act, _txt2W!, _txt2B);
        act.Dispose();
        return outp;
    }

    /// <summary>final_layer: <c>scale = temb + table[0]</c>, <c>shift = temb + table[1]</c>;
    /// <c>(1+scale)·RMSNorm(h) + shift</c> → <c>Linear(hidden → p²·channels)</c>.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor h, Tensor temb, int batch, int seqLen, int hidden)
    {
        Tensor normed = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        backend.RmsNorm(normed, h, _finalNormW!, _config.NormEps);

        Tensor modulated = new Tensor(new TensorShape(batch, seqLen, hidden), DType.F32);
        float* np = (float*)normed.DataPointer, mp = (float*)modulated.DataPointer;
        float* tp = (float*)temb.DataPointer, tab = (float*)_finalTable!.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long tb = (long)b * hidden;
            for (int s = 0; s < seqLen; s++)
            {
                long vb = ((long)b * seqLen + s) * hidden;
                for (int d = 0; d < hidden; d++)
                {
                    float scale = tp[tb + d] + tab[d];
                    float shift = tp[tb + d] + tab[hidden + d];
                    mp[vb + d] = (1.0f + scale) * np[vb + d] + shift;
                }
            }
        }
        normed.Dispose();

        int outDim = _config.InChannels; // p²·channels = 64
        Tensor projected = new Tensor(new TensorShape(batch, seqLen, outDim), DType.F32);
        backend.Linear(projected, modulated, _finalLinW!, _finalLinB);
        modulated.Dispose();
        return projected;
    }

    /// <summary>Patchifies <c>[B, C, H, W]</c> → <c>[B, (H/p)(W/p), C·p²]</c> with <b>channel-outer</b> feature order
    /// <c>(c, ph, pw)</c> (diffusers <c>view → permute(0,2,4,1,3,5) → reshape</c>).</summary>
    private static Tensor PatchifyChannelOuter(Tensor latent, int batch, int channels, int height, int width, int patch)
    {
        int hPacked = height / patch, wPacked = width / patch;
        int imgSeq = hPacked * wPacked, patchVol = channels * patch * patch;
        Tensor result = new Tensor(new TensorShape(batch, imgSeq, patchVol), DType.F32);
        float* src = (float*)latent.DataPointer, dst = (float*)result.DataPointer;
        long chw = (long)channels * height * width, hw = (long)height * width;
        for (int b = 0; b < batch; b++)
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    float* tok = dst + (((long)b * imgSeq) + ((long)hp * wPacked + wp)) * patchVol;
                    int idx = 0;
                    for (int c = 0; c < channels; c++)
                        for (int ph = 0; ph < patch; ph++)
                            for (int pw = 0; pw < patch; pw++)
                                tok[idx++] = src[b * chw + c * hw + (long)(hp * patch + ph) * width + (wp * patch + pw)];
                }
        return result;
    }

    /// <summary>Inverse of <see cref="PatchifyChannelOuter"/>: <c>[B, (H/p)(W/p), C·p²]</c> → <c>[B, C, H, W]</c>.</summary>
    private static Tensor UnpatchifyChannelOuter(Tensor tokens, int batch, int channels, int hPacked, int wPacked, int patch)
    {
        int height = hPacked * patch, width = wPacked * patch;
        int imgSeq = hPacked * wPacked, patchVol = channels * patch * patch;
        Tensor result = new Tensor(new TensorShape(batch, channels, height, width), DType.F32);
        float* src = (float*)tokens.DataPointer, dst = (float*)result.DataPointer;
        long chw = (long)channels * height * width, hw = (long)height * width;
        for (int b = 0; b < batch; b++)
            for (int hp = 0; hp < hPacked; hp++)
                for (int wp = 0; wp < wPacked; wp++)
                {
                    float* tok = src + (((long)b * imgSeq) + ((long)hp * wPacked + wp)) * patchVol;
                    int idx = 0;
                    for (int c = 0; c < channels; c++)
                        for (int ph = 0; ph < patch; ph++)
                            for (int pw = 0; pw < patch; pw++)
                                dst[b * chw + c * hw + (long)(hp * patch + ph) * width + (wp * patch + pw)] = tok[idx++];
                }
        return result;
    }

    private static Tensor SliceTail(Tensor joint, int start, int tailLen, int hidden)
    {
        Tensor output = new Tensor(new TensorShape(1, tailLen, hidden), DType.F32);
        long bytes = (long)tailLen * hidden * sizeof(float);
        Buffer.MemoryCopy((float*)joint.DataPointer + (long)start * hidden, (void*)output.DataPointer, bytes, bytes);
        return output;
    }

    private static Tensor F32(Tensor t) => t.DType == DType.F32 ? t : t.CastTo(DType.F32);

    private void ThrowIfDisposed() =>
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) == 0)
        {
            _imgInW = _imgInB = _time1W = _time1B = _time2W = _time2B = _timeModW = _timeModB = null;
            _txtNormW = _txt1W = _txt1B = _txt2W = _txt2B = null;
            _finalTable = _finalNormW = _finalLinW = _finalLinB = null;
        }
        GC.SuppressFinalize(this);
    }
}
