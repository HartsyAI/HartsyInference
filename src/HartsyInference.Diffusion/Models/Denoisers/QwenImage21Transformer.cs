using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using HartsyInference.Diffusion.Models.TextEncoders;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>Qwen-Image 2.1 denoiser (<c>QwenImage21Transformer2DModel</c>): 32 single-stream blocks over the
/// concatenated <c>[text, image]</c> sequence at hidden 4096. Shares no block code with
/// <see cref="QwenImageTransformer"/> — see <see cref="QwenImage21Config"/> for what actually differs.
///
/// <para><b>The text prefix is evaluated once, not once per step.</b> Text rows modulate from <c>t = 0</c> and
/// attend only to earlier text rows, so nothing about them depends on the timestep or on the image: their per-block
/// K/V are constant for a given prompt. <see cref="BuildPrefix"/> computes them into a
/// <see cref="QwenImage21PrefixCache"/>, and <see cref="Forward"/> then runs image rows alone against that cache
/// each step. ComfyUI arrives at the same arithmetic by evaluating the full sequence and caching the prefix
/// afterwards; splitting it up front means every block call sees a uniform modulation and no row-range splitting
/// is needed. The final <c>norm_out</c> reads target rows only in the reference too, so the prefix hidden states
/// are genuinely dead after their K/V are taken.</para></summary>
public sealed unsafe class QwenImage21Transformer : IDisposable
{
    private readonly QwenImage21Config _config;
    private readonly QwenImage21Block[] _blocks;
    private readonly QwenImage21Rope _rope;
    private readonly DType _act;
    private readonly int _mlpDim;

    private Tensor? _imgIn;
    private Tensor? _txtNormPlusOne;
    private Tensor? _txtInLayer, _txtOutLayer;
    private Tensor? _timeLinear1, _timeLinear2;
    private Tensor? _modulation;
    private Tensor? _normOutLinear;
    private Tensor? _projOut;
    private readonly List<Tensor> _owned = new();

    /// <param name="act">Activation dtype, F32 by default. The reference computes in bf16, but this engine's CUDA
    /// DiT recipe is F32-or-F16: <c>LayerNormNoAffine</c>, <c>RmsNorm</c> and <c>WanRopeInterleaved</c> have no BF16
    /// kernel, so bf16 activations fail at the first norm rather than running slowly. Weights stay bf16 either way —
    /// only the activations widen. F16 is the other served recipe and is why the reference block carries a
    /// <c>clip(±65504)</c>; it is selectable here but unverified on this model.</param>
    public QwenImage21Transformer(QwenImage21Config config, DType? act = null)
    {
        _config = config;
        _act = act ?? DType.F32;
        _mlpDim = config.HiddenSize * config.MlpRatio;
        if (config.NumHeads * config.HeadDim != config.HiddenSize)
            throw new ArgumentException($"numHeads * headDim ({config.NumHeads} * {config.HeadDim}) must equal hiddenSize ({config.HiddenSize}).", nameof(config));
        _rope = new QwenImage21Rope(config.AxesDim, config.RopeTheta);
        if (_rope.HeadDim != config.HeadDim)
            throw new ArgumentException($"RoPE axes sum to {_rope.HeadDim} but headDim is {config.HeadDim}.", nameof(config));
        _blocks = new QwenImage21Block[config.Depth];
        for (int i = 0; i < config.Depth; i++)
            _blocks[i] = new QwenImage21Block(config.HiddenSize, config.NumHeads, config.HeadDim, _mlpDim, config.Eps);
    }

    /// <summary>The blocks, for the streaming/sharding schedulers.</summary>
    public IReadOnlyList<QwenImage21Block> Blocks => _blocks;

    /// <summary>Loads every weight. The checkpoint is entirely bias-free, so no bias lookups are attempted.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        _imgIn = weights["img_in.weight"];
        _txtInLayer = weights["txt_in.in_layer.weight"];
        _txtOutLayer = weights["txt_in.out_layer.weight"];
        _timeLinear1 = weights["time_text_embed.timestep_embedder.linear_1.weight"];
        _timeLinear2 = weights["time_text_embed.timestep_embedder.linear_2.weight"];
        _modulation = weights["modulation.1.weight"];
        _normOutLinear = weights["norm_out.linear.weight"];
        _projOut = weights["proj_out.weight"];

        // ZeroCenteredRMSNorm: the file stores (scale − 1) and the reference adds 1 in fp32 on every call. Adding
        // it once at load is the same value and keeps the per-call path a plain RmsNorm — the same trick
        // LlamaStyleEncoder's RmsNormScalePlusOne uses. Owned, because it is a new tensor, not a checkpoint view.
        Tensor textNorm = TensorCasts.EnsureF32(weights["txt_in.text_norm.weight"]);
        if (ReferenceEquals(textNorm, weights["txt_in.text_norm.weight"]))
            textNorm = CloneF32(textNorm);
        float* p = (float*)textNorm.DataPointer;
        for (long i = 0; i < textNorm.ElementCount; i++) p[i] += 1.0f;
        _txtNormPlusOne = textNorm;
        _owned.Add(textNorm);

        if ((int)_modulation.Shape[0] != 4 * _config.HiddenSize)
        {
            throw new ArgumentException(
                $"modulation.1.weight has {_modulation.Shape[0]} rows; Qwen-Image 2.1 expects 4*{_config.HiddenSize} "
                + "(scale1, gate1, scale2, gate2) shared by every block.");
        }

        for (int i = 0; i < _config.Depth; i++)
            _blocks[i].LoadWeights(weights, $"transformer_blocks.{i}");
    }

    /// <summary>Every weight, for GPU preloading / residency accounting.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor? w in new[] { _imgIn, _txtNormPlusOne, _txtInLayer, _txtOutLayer, _timeLinear1, _timeLinear2, _modulation, _normOutLinear, _projOut })
            if (w is not null) yield return w;
        foreach (QwenImage21Block block in _blocks)
            foreach (Tensor w in block.EnumerateWeights()) yield return w;
    }

    /// <summary>Runs the text prefix through every block at <c>t = 0</c> and returns its per-block K/V. Valid for as
    /// long as the conditioning is unchanged; a different prompt (or the CFG counterpart) needs its own cache.</summary>
    /// <param name="context">Text-encoder hidden states, <c>[1, T, contextDim]</c>.</param>
    public QwenImage21PrefixCache BuildPrefix(IBackend backend, Tensor context)
    {
        int textLen = (int)context.Shape[context.Shape.Rank - 2];
        if ((int)context.Shape[context.Shape.Rank - 1] != _config.ContextDim)
            throw new ArgumentException($"context last dim {context.Shape[context.Shape.Rank - 1]} != contextDim {_config.ContextDim}.", nameof(context));

        using Tensor temb0 = ComputeTimestepEmbedding(backend, 0.0f);
        using QwenImage21Modulation mod0 = ComputeModulation(backend, temb0);
        (Tensor ropeCos, Tensor ropeSin) = _rope.GetOrBuildTextTables(backend, textLen);
        using Tensor causalMaskHost = TextEncoderTensorHelpers.BuildCausalMask(textLen, int.MaxValue);
        using Tensor causalMask = TextEncoderTensorHelpers.MakeResidentCopy(backend, causalMaskHost);

        Tensor hidden = ProjectText(backend, context, textLen);
        Tensor[] keys = new Tensor[_config.Depth];
        Tensor[] values = new Tensor[_config.Depth];
        try
        {
            for (int i = 0; i < _config.Depth; i++)
            {
                Tensor next = _blocks[i].ForwardPrefix(backend, hidden, mod0, ropeCos, ropeSin, causalMask,
                    out keys[i], out values[i]);
                hidden.Dispose();
                hidden = next;
            }
        }
        catch
        {
            foreach (Tensor? t in keys) t?.Dispose();
            foreach (Tensor? t in values) t?.Dispose();
            hidden.Dispose();
            throw;
        }
        // The prefix hidden state is dead here by construction: norm_out reads target rows only.
        hidden.Dispose();
        return new QwenImage21PrefixCache(textLen, keys, values);
    }

    /// <summary>One denoise step. <paramref name="latent"/> is <c>[1, inChannels, H, W]</c>; the VAE does all 16× of
    /// the spatial reduction, so one latent cell is one token and there is no patchify. Returns the velocity in the
    /// same shape and dtype as the input latent.</summary>
    public Tensor Forward(IBackend backend, Tensor latent, QwenImage21PrefixCache prefix, float timestep)
    {
        int channels = (int)latent.Shape[1];
        int h = (int)latent.Shape[2];
        int w = (int)latent.Shape[3];
        if (channels != _config.InChannels)
            throw new ArgumentException($"latent has {channels} channels; expected {_config.InChannels}.", nameof(latent));
        int imgSeq = h * w;

        // The reference rounds t through the compute dtype before the sinusoid, twice: (t*1000).to(dtype)/1000
        // then .to(dtype). Reproduced so the embedding lands on the same value the reference feeds its blocks.
        float rounded = RoundThroughAct(RoundThroughAct(timestep * 1000.0f) / 1000.0f);
        using Tensor temb = ComputeTimestepEmbedding(backend, rounded);
        using QwenImage21Modulation mod = ComputeModulation(backend, temb);
        (Tensor ropeCos, Tensor ropeSin) = _rope.GetOrBuildImageTables(backend, prefix.Length, h, w);

        Tensor tokens = new Tensor(new TensorShape(1, imgSeq, channels), latent.DType);
        backend.PatchifyTokens(tokens, latent, patch: 1, innerChannelFastest: true);
        Tensor tokensAct = Cast(backend, tokens, _act);
        if (!ReferenceEquals(tokensAct, tokens)) tokens.Dispose();

        Tensor hidden = new Tensor(new TensorShape(1, imgSeq, _config.HiddenSize), _act);
        backend.Linear(hidden, tokensAct, _imgIn!, null);
        tokensAct.Dispose();

        for (int i = 0; i < _config.Depth; i++)
        {
            Tensor next = _blocks[i].ForwardTarget(backend, hidden, mod, ropeCos, ropeSin,
                prefix.Keys[i], prefix.Values[i]);
            hidden.Dispose();
            hidden = next;
        }

        Tensor normed = ApplyFinalLayer(backend, hidden, temb, imgSeq);
        hidden.Dispose();

        Tensor projected = new Tensor(new TensorShape(1, imgSeq, _config.OutChannels), _act);
        backend.Linear(projected, normed, _projOut!, null);
        normed.Dispose();

        Tensor projectedOut = Cast(backend, projected, latent.DType);
        if (!ReferenceEquals(projectedOut, projected)) projected.Dispose();
        Tensor output = new Tensor(new TensorShape(1, _config.OutChannels, h, w), latent.DType);
        backend.UnpatchifyTokens(output, projectedOut, _config.OutChannels, h, w, patch: 1, innerChannelFastest: true);
        projectedOut.Dispose();
        return output;
    }

    /// <summary><c>TextProjection</c>: zero-centered RMSNorm, then <c>in_layer</c>, tanh-approximate GELU and
    /// <c>out_layer</c>.</summary>
    private Tensor ProjectText(IBackend backend, Tensor context, int textLen)
    {
        Tensor contextAct = Cast(backend, context, _act);
        Tensor normed = new Tensor(new TensorShape(1, textLen, _config.ContextDim), _act);
        backend.RmsNorm(normed, contextAct, _txtNormPlusOne!, _config.Eps);
        if (!ReferenceEquals(contextAct, context)) contextAct.Dispose();

        TensorShape hiddenShape = new TensorShape(1, textLen, _config.HiddenSize);
        Tensor inner = new Tensor(hiddenShape, _act);
        backend.Linear(inner, normed, _txtInLayer!, null);
        normed.Dispose();
        Tensor activated = new Tensor(hiddenShape, _act);
        backend.Gelu(activated, inner);
        inner.Dispose();
        Tensor projected = new Tensor(hiddenShape, _act);
        backend.Linear(projected, activated, _txtOutLayer!, null);
        activated.Dispose();
        return projected;
    }

    /// <summary>Sinusoidal timestep embedding (cos-then-sin halves, <c>time_factor</c> 1000 folded into the scaled
    /// argument) through the bias-free <c>linear_1 → SiLU → linear_2</c> projector.</summary>
    /// <remarks>Kept in F32 whatever the activation dtype. These are <c>[1, 256]</c> and <c>[1, hidden]</c>
    /// tensors evaluated twice per step, so the width costs nothing, and the modulation they feed goes through
    /// <c>AddScalar</c> and <c>Tanh</c>, which this engine's CUDA backend serves for F32 only.</remarks>
    private Tensor ComputeTimestepEmbedding(IBackend backend, float timestep)
    {
        Tensor sinusoid = new Tensor(new TensorShape(1, 256), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoid, timestep * 1000.0f, batch: 1, embDim: 256);

        TensorShape shape = new TensorShape(1, _config.HiddenSize);
        Tensor first = new Tensor(shape, DType.F32);
        backend.Linear(first, sinusoid, _timeLinear1!, null);
        sinusoid.Dispose();
        Tensor activated = new Tensor(shape, DType.F32);
        backend.Silu(activated, first);
        first.Dispose();
        Tensor temb = new Tensor(shape, DType.F32);
        backend.Linear(temb, activated, _timeLinear2!, null);
        activated.Dispose();
        return temb;
    }

    /// <summary>The one modulation every block reads: <c>Linear(SiLU(temb))</c> chunked into
    /// <c>(scale1, gate1, scale2, gate2)</c>, gates through <c>tanh</c>, scales pre-biased to <c>1 + scale</c> so the
    /// blocks can call a scale-only affine directly.</summary>
    private QwenImage21Modulation ComputeModulation(IBackend backend, Tensor temb)
    {
        int hidden = _config.HiddenSize;
        Tensor activated = new Tensor(new TensorShape(1, hidden), DType.F32);
        backend.Silu(activated, temb);
        Tensor all = new Tensor(new TensorShape(1, 4 * hidden), DType.F32);
        backend.Linear(all, activated, _modulation!, null);
        activated.Dispose();

        TensorShape one = new TensorShape(1, hidden);
        Tensor scale1 = new Tensor(one, DType.F32);
        Tensor gate1 = new Tensor(one, DType.F32);
        Tensor scale2 = new Tensor(one, DType.F32);
        Tensor gate2 = new Tensor(one, DType.F32);
        backend.Split([scale1, gate1, scale2, gate2], all, 1);
        all.Dispose();

        Tensor scale1Plus1 = Widened(backend, scale1, 1.0f);
        Tensor scale2Plus1 = Widened(backend, scale2, 1.0f);
        Tensor gate1Tanh = Tanhed(backend, gate1);
        Tensor gate2Tanh = Tanhed(backend, gate2);
        return new QwenImage21Modulation(scale1Plus1, gate1Tanh, scale2Plus1, gate2Tanh);
    }

    /// <summary><c>1 + x</c>, kept F32. The modulation stays F32 whatever the activation dtype is: this engine's
    /// 16-bit DiT recipe is <b>16-bit activations with an F32 scale/gate</b> — both
    /// <see cref="IBackend.AffineBroadcastLastDim"/> and <see cref="IBackend.GatedResidualLastDim"/> refuse a
    /// 16-bit scale — and these are <c>[1, hidden]</c> tensors, so the width is free.</summary>
    private static Tensor Widened(IBackend backend, Tensor value, float bias)
    {
        Tensor sum = new Tensor(value.Shape, DType.F32);
        backend.AddScalar(sum, value, bias);
        value.Dispose();
        return sum;
    }

    private static Tensor Tanhed(IBackend backend, Tensor value)
    {
        Tensor activated = new Tensor(value.Shape, DType.F32);
        backend.Tanh(activated, value);
        value.Dispose();
        return activated;
    }

    /// <summary><c>LastLayer</c>: scale-only adaLN — <c>LayerNorm(x) · (1 + Linear(SiLU(temb)))</c> with no shift
    /// term at all, unlike the <c>[shift, scale]</c> chunk every other DiT in this repo uses.</summary>
    private Tensor ApplyFinalLayer(IBackend backend, Tensor hidden, Tensor temb, int seq)
    {
        int dim = _config.HiddenSize;
        Tensor activated = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Silu(activated, temb);
        Tensor scale = new Tensor(new TensorShape(1, dim), DType.F32);
        backend.Linear(scale, activated, _normOutLinear!, null);
        activated.Dispose();
        Tensor scalePlus1 = Widened(backend, scale, 1.0f);

        TensorShape shape = new TensorShape(1, seq, dim);
        Tensor normed = new Tensor(shape, _act);
        backend.LayerNormNoAffine(normed, hidden, _config.Eps);
        Tensor output = new Tensor(shape, _act);
        backend.AffineBroadcastLastDim(output, normed, scalePlus1, null);
        normed.Dispose();
        scalePlus1.Dispose();
        return output;
    }

    /// <summary>Rounds a value through the activation dtype, reproducing the reference's deliberate
    /// <c>.to(dtype)</c> on the timestep. A no-op when activations are already F32.</summary>
    /// <remarks>Round-to-nearest-even, not the truncation <see cref="TensorCasts.F32ToBf16Bits"/> performs: this
    /// mirrors a PyTorch <c>.to(bfloat16)</c>, and the two disagree on roughly half of all values.</remarks>
    private float RoundThroughAct(float value)
    {
        if (_act == DType.BF16) return Bf16RoundTrip(value);
        if (_act == DType.F16) return (float)(Half)value;
        return value;
    }

    internal static float Bf16RoundTrip(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        if ((bits & 0x7F80_0000u) == 0x7F80_0000u) return value;   // NaN/Inf survive unchanged
        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(rounded & 0xFFFF_0000u);
    }

    private static Tensor CloneF32(Tensor source)
    {
        Tensor copy = new Tensor(source.Shape, DType.F32);
        source.AsSpan<float>().CopyTo(copy.AsSpan<float>());
        return copy;
    }

    private static Tensor Cast(IBackend backend, Tensor source, DType target)
    {
        if (source.DType == target) return source;
        Tensor cast = new Tensor(source.Shape, target);
        if (target == DType.F32) backend.CastToF32(cast, source);
        else if (target == DType.F16) backend.CastToF16(cast, source);
        else if (target == DType.BF16) backend.CastToBf16(cast, source);
        else
        {
            cast.Dispose();
            throw new NotSupportedException($"Qwen-Image 2.1 activations must be F32, F16 or BF16; got {target}.");
        }
        return cast;
    }

    public void Dispose()
    {
        _rope.Dispose();
        foreach (QwenImage21Block block in _blocks) block.DisposeOwned();
        foreach (Tensor t in _owned) t.Dispose();
        _owned.Clear();
    }
}

/// <summary>Per-block K/V of the text prefix, constant across denoise steps for a given prompt. Sized
/// <c>depth × 2 × T × hidden</c> — about 105 MB at 200 text tokens, which is why it is held resident rather than
/// offloaded the way ComfyUI's optional int8/CPU cache allows.</summary>
public sealed class QwenImage21PrefixCache(int length, Tensor[] keys, Tensor[] values) : IDisposable
{
    /// <summary>Number of text tokens in the prefix.</summary>
    public int Length { get; } = length;

    internal Tensor[] Keys { get; } = keys;

    internal Tensor[] Values { get; } = values;

    public void Dispose()
    {
        foreach (Tensor t in Keys) t.Dispose();
        foreach (Tensor t in Values) t.Dispose();
    }
}
