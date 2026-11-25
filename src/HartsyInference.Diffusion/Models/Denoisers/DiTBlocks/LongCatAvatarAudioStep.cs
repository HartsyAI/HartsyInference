using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>The avatar's audio step inside a LongCat block (<c>LongCatAvatarSingleStreamBlock</c>, <c>modules/avatar/longcat_video_dit_avatar.py</c>
/// @ 58eb336), applied after the text cross-attention and before the FFN. The step reads the per-frame timestep of the noisy frames through
/// <c>audio_adaLN_modulation</c> (SiLU, then a Linear to shift, scale and gate), runs the audio cross-attention over the full stream after
/// <c>pre_video_crs_attn_norm</c>, modulates the noisy frames' audio output with the output-norm (no affine, eps 1e-6), and adds it gated to the
/// noisy rows only. The conditioning rows are left unchanged.</summary>
/// <remarks>
/// Weights under the block's prefix: <c>audio_adaLN_modulation.1</c> (weight, bias), <c>pre_video_crs_attn_norm</c> (weight, bias, eps 1e-6)
/// and <c>audio_cross_attn.*</c> (<see cref="LongCatAudioCrossAttention"/>). The reference's <c>pre_audio_crs_attn_norm</c> is absent here: the
/// avatar configs ship <c>audio_prenorm: false</c>, which makes it the identity. A checkpoint that sets it is refused at load.
/// </remarks>
internal sealed class LongCatAvatarAudioStep : IDisposable
{
    /// <summary>Epsilon of the video pre-norm and the output norm (<c>LayerNorm_FP32</c> in the reference, eps 1e-6).</summary>
    public const float NormEps = 1e-6f;

    private readonly int _dim;
    private readonly LongCatAudioCrossAttention _attention;
    private readonly List<Tensor> _owned = [];
    private Tensor? _adaW, _adaB, _preW, _preB;
    private int _disposed;

    private LongCatAvatarAudioStep(int dim, int encoderDim, int heads)
    {
        _dim = dim;
        _attention = new LongCatAudioCrossAttention(dim, encoderDim, heads);
    }

    /// <summary>Reads the step's tensors under <paramref name="prefix"/> (the block's <c>blocks.N</c>). Matrices are used as the checkpoint stores
    /// them (quantized or not, a LoRA adjunct applies to them); only vectors are widened to F32, and those widened copies belong to this step.
    /// All-or-nothing: a missing or wrong-shaped tensor throws, and nothing is kept.</summary>
    public static LongCatAvatarAudioStep Load(IReadOnlyDictionary<string, Tensor> w, string prefix, int dim, int adaLnWidth, int encoderDim,
        int heads)
    {
        if (w.ContainsKey(prefix + ".pre_audio_crs_attn_norm.weight"))
            throw new InvalidDataException($"{prefix}.pre_audio_crs_attn_norm is present; the avatar configs ship audio_prenorm false, and that norm is not implemented.");
        LongCatAvatarAudioStep step = new(dim, encoderDim, heads);
        List<Tensor> local = [];
        try
        {
            step._adaW = Expect(w[prefix + ".audio_adaLN_modulation.1.weight"], [3 * dim, adaLnWidth], prefix + ".audio_adaLN_modulation.1.weight");
            step._adaB = Expect(LongCatOps.WidenF32(w, prefix + ".audio_adaLN_modulation.1.bias", local), [3 * dim],
                prefix + ".audio_adaLN_modulation.1.bias");
            step._preW = Expect(LongCatOps.WidenF32(w, prefix + ".pre_video_crs_attn_norm.weight", local), [dim], prefix + ".pre_video_crs_attn_norm.weight");
            step._preB = Expect(LongCatOps.WidenF32(w, prefix + ".pre_video_crs_attn_norm.bias", local), [dim], prefix + ".pre_video_crs_attn_norm.bias");
            step._attention.LoadWeights(w, prefix + ".audio_cross_attn.");
            step._owned.AddRange(local);
            local.Clear();
            return step;
        }
        catch
        {
            foreach (Tensor t in local)
                t.Dispose();
            step.Dispose();
            throw;
        }
    }

    /// <summary>The step's weights: the audio modulation and pre-norm, then the audio cross-attention's.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? t in new[] { _adaW, _adaB, _preW, _preB })
            if (t is not null)
                yield return t;
        foreach (Tensor t in _attention.EnumerateWeights())
            yield return t;
    }

    /// <summary>The step's contribution to a block's stream.</summary>
    /// <param name="x">The stream after the text cross-attention, <c>[frames·S, dim]</c>.</param>
    /// <param name="audioTokens">The per-frame audio tokens, <c>[frames·ctx, encoderDim]</c>.</param>
    /// <param name="timeEmbed">The per-frame timestep embedding, <c>[frames, adaLnWidth]</c>.</param>
    /// <returns>A new <c>[frames·S, dim]</c> stream; the conditioning rows are the input rows.</returns>
    public Tensor Apply(IBackend backend, Tensor x, Tensor audioTokens, Tensor timeEmbed, int numCond, int tokensPerFrame)
    {
        ObjectDisposedException.ThrowIf(_disposed != 0, this);
        if (_adaW is null)
            throw new InvalidOperationException("The audio step has no weights: Load was never called, or it failed.");
        int frames = (int)timeEmbed.Shape[0];
        int noisy = frames - numCond, condRows = numCond * tokensPerFrame, noisyRows = noisy * tokensPerFrame;
        if (noisy < 1)
            throw new ArgumentOutOfRangeException(nameof(numCond), numCond, "The audio step needs at least one noisy frame.");

        // Per-frame audio modulation for the noisy frames: Linear(SiLU(t)), packed as [shift | scale | gate].
        using Tensor noisyTime = new(new TensorShape(noisy, timeEmbed.Shape[1]), DType.F32);
        backend.SliceRows(noisyTime, timeEmbed, numCond);
        using Tensor activated = new(noisyTime.Shape, DType.F32);
        backend.Silu(activated, noisyTime);
        using Tensor modulation = new(new TensorShape(noisy, 3 * _dim), DType.F32);
        backend.Linear(modulation, activated, _adaW!, _adaB);

        // Audio cross-attention over the full stream after the affine video pre-norm; the conditioning rows come back as zeros.
        using Tensor preNormed = new(x.Shape, DType.F32);
        backend.LayerNorm(preNormed, x, _preW!, _preB!, NormEps);
        using Tensor audioOut = _attention.Forward(backend, preNormed, audioTokens, frames, numCond);
        using Tensor noiseOut = new(new TensorShape(noisyRows, _dim), DType.F32);
        backend.SliceRows(noiseOut, audioOut, condRows);

        // Output norm with shift and scale of the noisy frames (1 + scale, then + shift), then the gated residual on the noisy rows.
        using Tensor modulated = LongCatOps.NormModulate(backend, noiseOut, modulation, _dim, 0, tokensPerFrame);
        using Tensor noisyX = new(new TensorShape(noisyRows, _dim), DType.F32);
        backend.SliceRows(noisyX, x, condRows);
        using Tensor frameIndex = LongCatOps.FrameRowIndex(noisy, tokensPerFrame);
        Tensor updated = LongCatOps.GatedResidual(backend, noisyX, modulated, modulation, 2 * _dim, _dim, frameIndex);
        if (numCond == 0)
            return updated;

        using (updated)
        {
            using Tensor conditioning = new(new TensorShape(condRows, _dim), DType.F32);
            backend.SliceRows(conditioning, x, 0);
            Tensor joined = new(new TensorShape(frames * tokensPerFrame, _dim), DType.F32);
            backend.Concat(joined, [conditioning, updated], 0);
            return joined;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _attention.Dispose();
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned.Clear();
    }

    private static Tensor Expect(Tensor tensor, long[] shape, string key)
    {
        if (tensor.Shape.Rank != shape.Length)
            throw new InvalidDataException($"{key} must have {shape.Length} dimensions, got {tensor.Shape}.");
        for (int i = 0; i < shape.Length; i++)
            if (tensor.Shape[i] != shape[i])
                throw new InvalidDataException($"{key} must be {string.Join('x', shape)}, got {tensor.Shape}.");
        return tensor;
    }
}
