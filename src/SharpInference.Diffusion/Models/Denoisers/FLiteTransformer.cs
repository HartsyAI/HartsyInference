using SharpInference.Core.Backends;
using SharpInference.Core.Tensors;
using SharpInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace SharpInference.Diffusion.Models.Denoisers;

/// <summary>F-Lite top-level transformer. Orchestrates: patch_embed → register-token concat → 2D RoPE → time embed → 40 blocks (with V-residual carry-forward) → strip register tokens → final AdaLN modulation → final proj → unpatchify.
///
/// <para><b>V-residual</b>: F-Lite's distinguishing feature. Block 0 produces a V from its self-attention. Subsequent blocks mix that V into theirs via <c>v = lambda * v_current + (1 - lambda) * v_0</c> where lambda is per-block learnable. This carries via the loop variable — caller need not handle it explicitly.</para></summary>
public sealed unsafe class FLiteTransformer : IDisposable
{
    private readonly FLiteConfig _config;
    private readonly FLiteRope _rope;
    private readonly FLiteBlock[] _blocks;
    private int _disposed;

    private Tensor? _patchEmbedWeight, _patchEmbedBias;
    private Tensor? _registerTokens;
    private Tensor? _timeEmbedUpWeight, _timeEmbedUpBias;
    private Tensor? _timeEmbedDownWeight, _timeEmbedDownBias;
    private Tensor? _finalModulationWeight, _finalModulationBias;
    private Tensor? _finalNormWeight;
    private Tensor? _finalProjWeight, _finalProjBias;

    public FLiteConfig Config => _config;

    public FLiteTransformer(FLiteConfig config)
    {
        _config = config;
        _rope = new FLiteRope(config.HeadDim, config.RopeBase, config.RopeMaxGrid);
        _blocks = new FLiteBlock[config.Depth];
        for (int i = 0; i < config.Depth; i++) _blocks[i] = new FLiteBlock(config);
    }

    /// <summary>Loads the full transformer weight set from a diffusers-format dict (keys rooted at the dit_model component).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _patchEmbedWeight = weights["patch_embed.patch_proj.weight"];
        weights.TryGetValue("patch_embed.patch_proj.bias", out _patchEmbedBias);
        _registerTokens = weights["register_tokens"];

        _timeEmbedUpWeight = weights["time_embed.0.weight"];
        weights.TryGetValue("time_embed.0.bias", out _timeEmbedUpBias);
        _timeEmbedDownWeight = weights["time_embed.2.weight"];
        weights.TryGetValue("time_embed.2.bias", out _timeEmbedDownBias);

        for (int i = 0; i < _config.Depth; i++)
        {
            _blocks[i].LoadWeights(weights, $"blocks.{i}");
        }

        _finalModulationWeight = weights["final_modulation.1.weight"];
        weights.TryGetValue("final_modulation.1.bias", out _finalModulationBias);
        if (_config.TrainBiasAndRms)
        {
            weights.TryGetValue("final_norm.weight", out _finalNormWeight);
        }
        _finalProjWeight = weights["final_proj.weight"];
        weights.TryGetValue("final_proj.bias", out _finalProjBias);
    }

    /// <summary>Yields all weight tensors for GPU preloading.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_patchEmbedWeight is not null) yield return _patchEmbedWeight;
        if (_patchEmbedBias is not null) yield return _patchEmbedBias;
        if (_registerTokens is not null) yield return _registerTokens;
        if (_timeEmbedUpWeight is not null) yield return _timeEmbedUpWeight;
        if (_timeEmbedUpBias is not null) yield return _timeEmbedUpBias;
        if (_timeEmbedDownWeight is not null) yield return _timeEmbedDownWeight;
        if (_timeEmbedDownBias is not null) yield return _timeEmbedDownBias;
        for (int i = 0; i < _blocks.Length; i++)
            foreach (Tensor w in _blocks[i].EnumerateWeights()) yield return w;
        if (_finalModulationWeight is not null) yield return _finalModulationWeight;
        if (_finalModulationBias is not null) yield return _finalModulationBias;
        if (_finalNormWeight is not null) yield return _finalNormWeight;
        if (_finalProjWeight is not null) yield return _finalProjWeight;
        if (_finalProjBias is not null) yield return _finalProjBias;
    }

    /// <summary>Forward pass. <paramref name="latent"/> shape <c>[B, in_channels, H, W]</c> (typically [B, 16, h/8, w/8]); <paramref name="context"/> shape <c>[B, S_ctx, cross_attn_input_size]</c> (T5 layer-17 hidden states); <paramref name="timestepNormalized"/> in [0, 1] (caller scales by 1000 internally to match the reference). Returns predicted velocity of the same shape as <paramref name="latent"/>.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, Tensor context, float timestepNormalized)
    {
        if (_patchEmbedWeight is null) throw new InvalidOperationException("LoadWeights has not been called.");

        int batch = (int)latent.Shape[0];
        int latentH = (int)latent.Shape[2];
        int latentW = (int)latent.Shape[3];
        int patchSize = _config.PatchSize;
        int hPacked = latentH / patchSize;
        int wPacked = latentW / patchSize;
        int hidden = _config.HiddenSize;
        int registerCount = _config.NumRegisterTokens;

        Tensor patches = PatchEmbed(backend, latent, hPacked, wPacked, hidden);
        Tensor xWithRegister = PrependRegisterTokens(patches, batch, hPacked * wPacked, registerCount, hidden);
        patches.Dispose();

        (Tensor cosRope, Tensor sinRope) = _config.UseRope
            ? _rope.Build(hPacked, wPacked, registerCount)
            : (null!, null!);

        Tensor temb = ComputeTimeEmbedding(backend, batch, timestepNormalized);

        int totalSeqLen = registerCount + hPacked * wPacked;
        Tensor x = xWithRegister;
        Tensor? v0 = null;
        for (int i = 0; i < _config.Depth; i++)
        {
            (Tensor xNext, Tensor v) = _blocks[i].Forward(backend, x, context, temb,
                v0, _config.UseRope ? cosRope : null, _config.UseRope ? sinRope : null);
            x.Dispose();
            x = xNext;
            if (v0 is null) v0 = v;
            else v.Dispose();
        }
        v0?.Dispose();
        cosRope?.Dispose();
        sinRope?.Dispose();

        Tensor stripped = StripRegisterTokens(x, batch, totalSeqLen, registerCount, hidden);
        x.Dispose();

        Tensor velocity = ApplyFinalLayer(backend, stripped, temb, batch, hPacked, wPacked, hidden, latentH, latentW);
        stripped.Dispose();
        temb.Dispose();
        return velocity;
    }

    private Tensor PatchEmbed(IBackend backend, Tensor latent, int hPacked, int wPacked, int hidden)
    {
        int batch = (int)latent.Shape[0];
        int inCh = _config.InChannels;
        int patchSize = _config.PatchSize;
        Tensor patched = new Tensor(new TensorShape(batch, hidden, hPacked, wPacked), DType.F32);
        backend.Conv2D(patched, latent, _patchEmbedWeight!, _patchEmbedBias,
            strideH: patchSize, strideW: patchSize, padH: 0, padW: 0);

        Tensor reshaped = new Tensor(new TensorShape(batch, hPacked * wPacked, hidden), DType.F32);
        float* srcPtr = (float*)patched.DataPointer;
        float* dstPtr = (float*)reshaped.DataPointer;
        long spatial = (long)hPacked * wPacked;
        for (int b = 0; b < batch; b++)
        {
            for (int s = 0; s < spatial; s++)
            {
                int hy = (int)(s / wPacked);
                int wx = (int)(s % wPacked);
                for (int c = 0; c < hidden; c++)
                {
                    long srcIdx = (((long)b * hidden + c) * hPacked + hy) * wPacked + wx;
                    long dstIdx = (((long)b * spatial) + s) * hidden + c;
                    dstPtr[dstIdx] = srcPtr[srcIdx];
                }
            }
        }
        patched.Dispose();
        return reshaped;
    }

    private Tensor PrependRegisterTokens(Tensor patches, int batch, int patchSeqLen, int registerCount, int hidden)
    {
        int totalSeqLen = registerCount + patchSeqLen;
        Tensor result = new Tensor(new TensorShape(batch, totalSeqLen, hidden), DType.F32);
        float* dstPtr = (float*)result.DataPointer;
        float* regPtr = (float*)_registerTokens!.DataPointer;
        float* patchPtr = (float*)patches.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long batchOffset = (long)b * totalSeqLen * hidden;
            for (int r = 0; r < registerCount; r++)
            {
                Buffer.MemoryCopy(regPtr + (long)r * hidden, dstPtr + batchOffset + (long)r * hidden,
                    hidden * sizeof(float), hidden * sizeof(float));
            }
            long patchSrcOffset = (long)b * patchSeqLen * hidden;
            long patchDstOffset = batchOffset + (long)registerCount * hidden;
            Buffer.MemoryCopy(patchPtr + patchSrcOffset, dstPtr + patchDstOffset,
                (long)patchSeqLen * hidden * sizeof(float),
                (long)patchSeqLen * hidden * sizeof(float));
        }
        return result;
    }

    private Tensor ComputeTimeEmbedding(IBackend backend, int batch, float timestepNormalized)
    {
        int hidden = _config.HiddenSize;
        Tensor sinusoidal = new Tensor(new TensorShape(batch, hidden), DType.F32);
        float* sinPtr = (float*)sinusoidal.DataPointer;
        int half = hidden / 2;
        float scaledTime = timestepNormalized * 1000.0f;
        for (int b = 0; b < batch; b++)
        {
            float* row = sinPtr + (long)b * hidden;
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Exp(-MathF.Log(10000.0f) * i / half);
                float arg = scaledTime * freq;
                row[i] = MathF.Cos(arg);
                row[half + i] = MathF.Sin(arg);
            }
        }

        int mlpInner = 4 * hidden;
        Tensor up = new Tensor(new TensorShape(batch, mlpInner), DType.F32);
        backend.Linear(up, sinusoidal, _timeEmbedUpWeight!, _timeEmbedUpBias);
        sinusoidal.Dispose();
        Tensor activated = new Tensor(up.Shape, DType.F32);
        backend.Silu(activated, up);
        up.Dispose();
        Tensor result = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Linear(result, activated, _timeEmbedDownWeight!, _timeEmbedDownBias);
        activated.Dispose();
        return result;
    }

    private static Tensor StripRegisterTokens(Tensor x, int batch, int totalSeqLen, int registerCount, int hidden)
    {
        int patchSeqLen = totalSeqLen - registerCount;
        Tensor result = new Tensor(new TensorShape(batch, patchSeqLen, hidden), DType.F32);
        float* srcPtr = (float*)x.DataPointer;
        float* dstPtr = (float*)result.DataPointer;
        for (int b = 0; b < batch; b++)
        {
            long srcOffset = ((long)b * totalSeqLen + registerCount) * hidden;
            long dstOffset = (long)b * patchSeqLen * hidden;
            Buffer.MemoryCopy(srcPtr + srcOffset, dstPtr + dstOffset,
                (long)patchSeqLen * hidden * sizeof(float),
                (long)patchSeqLen * hidden * sizeof(float));
        }
        return result;
    }

    private Tensor ApplyFinalLayer(IBackend backend, Tensor x, Tensor temb, int batch, int hPacked, int wPacked,
        int hidden, int latentH, int latentW)
    {
        int patchSize = _config.PatchSize;
        int outCh = _config.InChannels;
        int spatial = hPacked * wPacked;

        Tensor activated = new Tensor(new TensorShape(batch, hidden), DType.F32);
        backend.Silu(activated, temb);
        Tensor proj2 = new Tensor(new TensorShape(batch, 2 * hidden), DType.F32);
        backend.Linear(proj2, activated, _finalModulationWeight!, _finalModulationBias);
        activated.Dispose();

        Tensor normed = ApplyFinalNormAndModulate(x, _finalNormWeight, proj2, batch, spatial, hidden);
        proj2.Dispose();

        Tensor projOut = new Tensor(new TensorShape(batch, spatial, patchSize * patchSize * outCh), DType.F32);
        backend.Linear(projOut, normed, _finalProjWeight!, _finalProjBias);
        normed.Dispose();

        Tensor unpatched = Unpatchify(projOut, batch, hPacked, wPacked, patchSize, outCh, latentH, latentW);
        projOut.Dispose();
        return unpatched;
    }

    private static Tensor ApplyFinalNormAndModulate(Tensor x, Tensor? normWeight, Tensor scaleShift,
        int batch, int spatial, int hidden)
    {
        Tensor result = new Tensor(new TensorShape(batch, spatial, hidden), DType.F32);
        float* xPtr = (float*)x.DataPointer;
        float* outPtr = (float*)result.DataPointer;
        float* ssPtr = (float*)scaleShift.DataPointer;
        float* normWPtr = normWeight is not null ? (float*)normWeight.DataPointer : null;
        const float Eps = 1e-6f;
        for (int b = 0; b < batch; b++)
        {
            float* shift = ssPtr + (long)b * 2 * hidden;
            float* scale = shift + hidden;
            for (int s = 0; s < spatial; s++)
            {
                long baseIdx = ((long)b * spatial + s) * hidden;
                float sumSq = 0f;
                for (int d = 0; d < hidden; d++)
                {
                    float v = xPtr[baseIdx + d];
                    sumSq += v * v;
                }
                float invRms = 1.0f / MathF.Sqrt(sumSq / hidden + Eps);
                for (int d = 0; d < hidden; d++)
                {
                    float normed = xPtr[baseIdx + d] * invRms;
                    if (normWPtr is not null) normed *= normWPtr[d];
                    outPtr[baseIdx + d] = normed * (1.0f + scale[d]) + shift[d];
                }
            }
        }
        return result;
    }

    private static Tensor Unpatchify(Tensor patched, int batch, int hPacked, int wPacked,
        int patchSize, int outCh, int latentH, int latentW)
    {
        Tensor result = new Tensor(new TensorShape(batch, outCh, latentH, latentW), DType.F32);
        float* srcPtr = (float*)patched.DataPointer;
        float* dstPtr = (float*)result.DataPointer;
        long perToken = (long)patchSize * patchSize * outCh;
        for (int b = 0; b < batch; b++)
        {
            for (int hy = 0; hy < hPacked; hy++)
            {
                for (int wx = 0; wx < wPacked; wx++)
                {
                    long tokenIdx = ((long)b * hPacked + hy) * wPacked + wx;
                    float* tokenPtr = srcPtr + tokenIdx * perToken;
                    for (int p1 = 0; p1 < patchSize; p1++)
                    {
                        for (int p2 = 0; p2 < patchSize; p2++)
                        {
                            for (int c = 0; c < outCh; c++)
                            {
                                long inIdx = ((long)p1 * patchSize + p2) * outCh + c;
                                int row = hy * patchSize + p1;
                                int col = wx * patchSize + p2;
                                long outIdx = (((long)b * outCh + c) * latentH + row) * latentW + col;
                                dstPtr[outIdx] = tokenPtr[inIdx];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        GC.SuppressFinalize(this);
    }
}
