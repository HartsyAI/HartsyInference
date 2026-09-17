using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Mert2;

/// <summary>One MERT-v2 Conformer layer: half-step feed-forward, RoPE self-attention, a depthwise convolution
/// module, a second half-step feed-forward, and a closing LayerNorm. Every sublayer is pre-normed and residual.
///
/// <para>Attention runs through <see cref="IBackend.ScaledDotProductAttention"/> with <c>allowF16</c>, never
/// <see cref="IBackend.FlashAttention"/>: this is 7500-token encoder attention, which the decode-tuned flash path
/// is not built for. F16 is safe here and measured, not assumed — against the released weights the score bound is
/// ~1.4e3 and the largest value element ~22, both far inside F16's range.</para></summary>
internal sealed class Mert2ConformerLayer
{
    private readonly Mert2Config _config;
    private readonly float _attentionScale;

    private Tensor? _ffn1NormWeight;
    private Tensor? _ffn1NormBias;
    private Tensor? _ffn1UpWeight;
    private Tensor? _ffn1UpBias;
    private Tensor? _ffn1DownWeight;
    private Tensor? _ffn1DownBias;

    private Tensor? _attnNormWeight;
    private Tensor? _attnNormBias;
    private Tensor? _queryWeight;
    private Tensor? _queryBias;
    private Tensor? _keyWeight;
    private Tensor? _keyBias;
    private Tensor? _valueWeight;
    private Tensor? _valueBias;
    private Tensor? _outWeight;
    private Tensor? _outBias;

    private Tensor? _convNormWeight;
    private Tensor? _convNormBias;
    private Tensor? _convGateWeight;
    private Tensor? _convDepthwiseWeight;
    private Tensor? _convInnerNormWeight;
    private Tensor? _convInnerNormBias;
    private Tensor? _convOutWeight;

    private Tensor? _ffn2NormWeight;
    private Tensor? _ffn2NormBias;
    private Tensor? _ffn2UpWeight;
    private Tensor? _ffn2UpBias;
    private Tensor? _ffn2DownWeight;
    private Tensor? _ffn2DownBias;

    private Tensor? _finalNormWeight;
    private Tensor? _finalNormBias;

    public Mert2ConformerLayer(Mert2Config config)
    {
        _config = config;
        _attentionScale = 1f / MathF.Sqrt(config.HeadDim);
    }

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, List<Tensor> owned)
    {
        _ffn1NormWeight = Mert2Ops.Load(weights, $"{prefix}.ffn1_layer_norm.weight", owned);
        _ffn1NormBias = Mert2Ops.Load(weights, $"{prefix}.ffn1_layer_norm.bias", owned);
        _ffn1UpWeight = Mert2Ops.Load(weights, $"{prefix}.ffn1.w_1.weight", owned);
        _ffn1UpBias = Mert2Ops.Load(weights, $"{prefix}.ffn1.w_1.bias", owned);
        _ffn1DownWeight = Mert2Ops.Load(weights, $"{prefix}.ffn1.w_2.weight", owned);
        _ffn1DownBias = Mert2Ops.Load(weights, $"{prefix}.ffn1.w_2.bias", owned);

        _attnNormWeight = Mert2Ops.Load(weights, $"{prefix}.attn_layer_norm.weight", owned);
        _attnNormBias = Mert2Ops.Load(weights, $"{prefix}.attn_layer_norm.bias", owned);
        _queryWeight = Mert2Ops.Load(weights, $"{prefix}.attn.query_proj.weight", owned);
        _queryBias = Mert2Ops.Load(weights, $"{prefix}.attn.query_proj.bias", owned);
        _keyWeight = Mert2Ops.Load(weights, $"{prefix}.attn.key_proj.weight", owned);
        _keyBias = Mert2Ops.Load(weights, $"{prefix}.attn.key_proj.bias", owned);
        _valueWeight = Mert2Ops.Load(weights, $"{prefix}.attn.value_proj.weight", owned);
        _valueBias = Mert2Ops.Load(weights, $"{prefix}.attn.value_proj.bias", owned);
        _outWeight = Mert2Ops.Load(weights, $"{prefix}.attn.out_proj.weight", owned);
        _outBias = Mert2Ops.Load(weights, $"{prefix}.attn.out_proj.bias", owned);

        _convNormWeight = Mert2Ops.Load(weights, $"{prefix}.conv_module.layer_norm.weight", owned);
        _convNormBias = Mert2Ops.Load(weights, $"{prefix}.conv_module.layer_norm.bias", owned);
        _convGateWeight = Mert2Ops.Load(weights, $"{prefix}.conv_module.conv_block.1.weight", owned);
        _convDepthwiseWeight = Mert2Ops.Load(weights, $"{prefix}.conv_module.conv_block.3.weight", owned);
        _convInnerNormWeight = Mert2Ops.Load(weights, $"{prefix}.conv_module.conv_block.4.1.weight", owned);
        _convInnerNormBias = Mert2Ops.Load(weights, $"{prefix}.conv_module.conv_block.4.1.bias", owned);
        _convOutWeight = Mert2Ops.Load(weights, $"{prefix}.conv_module.conv_block.6.weight", owned);

        _ffn2NormWeight = Mert2Ops.Load(weights, $"{prefix}.ffn2_layer_norm.weight", owned);
        _ffn2NormBias = Mert2Ops.Load(weights, $"{prefix}.ffn2_layer_norm.bias", owned);
        _ffn2UpWeight = Mert2Ops.Load(weights, $"{prefix}.ffn2.w_1.weight", owned);
        _ffn2UpBias = Mert2Ops.Load(weights, $"{prefix}.ffn2.w_1.bias", owned);
        _ffn2DownWeight = Mert2Ops.Load(weights, $"{prefix}.ffn2.w_2.weight", owned);
        _ffn2DownBias = Mert2Ops.Load(weights, $"{prefix}.ffn2.w_2.bias", owned);

        _finalNormWeight = Mert2Ops.Load(weights, $"{prefix}.final_layer_norm.weight", owned);
        _finalNormBias = Mert2Ops.Load(weights, $"{prefix}.final_layer_norm.bias", owned);
    }

    /// <summary>Runs the layer from <paramref name="input"/> into <paramref name="output"/>; the two must be
    /// different buffers, and <paramref name="input"/> is only read.</summary>
    public void Forward(IBackend backend, Tensor input, Tensor output, Scratch scratch)
    {
        // The four residual steps alternate between two scratch carriers and only the closing LayerNorm writes the
        // caller's buffer, so no op ever reads and writes the same storage.
        FeedForward(backend, input, scratch.CarrierA, scratch, _ffn1NormWeight!, _ffn1NormBias!,
            _ffn1UpWeight!, _ffn1UpBias!, _ffn1DownWeight!, _ffn1DownBias!);
        Attention(backend, scratch.CarrierA, scratch.CarrierB, scratch);
        Convolution(backend, scratch.CarrierB, scratch.CarrierA, scratch);
        FeedForward(backend, scratch.CarrierA, scratch.CarrierB, scratch, _ffn2NormWeight!, _ffn2NormBias!,
            _ffn2UpWeight!, _ffn2UpBias!, _ffn2DownWeight!, _ffn2DownBias!);
        backend.LayerNorm(output, scratch.CarrierB, _finalNormWeight!, _finalNormBias!, _config.LayerNormEps);
    }

    public IEnumerable<Tensor> EnumerateWeights()
    {
        yield return _ffn1NormWeight!;
        yield return _ffn1NormBias!;
        yield return _ffn1UpWeight!;
        yield return _ffn1UpBias!;
        yield return _ffn1DownWeight!;
        yield return _ffn1DownBias!;
        yield return _attnNormWeight!;
        yield return _attnNormBias!;
        yield return _queryWeight!;
        yield return _queryBias!;
        yield return _keyWeight!;
        yield return _keyBias!;
        yield return _valueWeight!;
        yield return _valueBias!;
        yield return _outWeight!;
        yield return _outBias!;
        yield return _convNormWeight!;
        yield return _convNormBias!;
        yield return _convGateWeight!;
        yield return _convDepthwiseWeight!;
        yield return _convInnerNormWeight!;
        yield return _convInnerNormBias!;
        yield return _convOutWeight!;
        yield return _ffn2NormWeight!;
        yield return _ffn2NormBias!;
        yield return _ffn2UpWeight!;
        yield return _ffn2UpBias!;
        yield return _ffn2DownWeight!;
        yield return _ffn2DownBias!;
        yield return _finalNormWeight!;
        yield return _finalNormBias!;
    }

    /// <summary>Pre-normed feed-forward folded into the residual at half weight, which is what makes a Conformer's
    /// two feed-forwards together equivalent to one full-weight sublayer.</summary>
    private void FeedForward(IBackend backend, Tensor input, Tensor output, Scratch scratch,
        Tensor normWeight, Tensor normBias, Tensor upWeight, Tensor upBias, Tensor downWeight, Tensor downBias)
    {
        backend.LayerNorm(scratch.Normed, input, normWeight, normBias, _config.LayerNormEps);
        backend.Linear(scratch.Hidden, scratch.Normed, upWeight, upBias);
        backend.GeluErf(scratch.Activated, scratch.Hidden);
        backend.Linear(scratch.Projected, scratch.Activated, downWeight, downBias);
        backend.AffineMix(output, input, scratch.Projected, 1f, 0.5f);
    }

    private void Attention(IBackend backend, Tensor input, Tensor output, Scratch scratch)
    {
        int tokens = (int)input.Shape[0];
        int heads = _config.Heads;
        int headDim = _config.HeadDim;
        backend.LayerNorm(scratch.Normed, input, _attnNormWeight!, _attnNormBias!, _config.LayerNormEps);
        backend.Linear(scratch.Query, scratch.Normed, _queryWeight!, _queryBias);
        backend.Linear(scratch.Key, scratch.Normed, _keyWeight!, _keyBias);
        backend.Linear(scratch.Value, scratch.Normed, _valueWeight!, _valueBias);

        // Split-half rotary on Q and K only; V is left alone, as in the released module.
        backend.ApplyRope(scratch.Query, scratch.Key, scratch.RopeCos, scratch.RopeSin);

        backend.Permute0213(scratch.QueryHeads, scratch.Query, tokens, heads, headDim);
        backend.Permute0213(scratch.KeyHeads, scratch.Key, tokens, heads, headDim);
        backend.Permute0213(scratch.ValueHeads, scratch.Value, tokens, heads, headDim);
        backend.ScaledDotProductAttention(scratch.AttentionHeads, scratch.QueryHeads, scratch.KeyHeads,
            scratch.ValueHeads, null, _attentionScale, allowF16: true);
        backend.Permute0213(scratch.AttentionTokens, scratch.AttentionHeads, heads, tokens, headDim);

        backend.Linear(scratch.Attention, scratch.AttentionTokens, _outWeight!, _outBias);
        backend.Add(output, input, scratch.Attention);
    }

    /// <summary>The convolution module: pointwise expand, gated linear unit over the channel axis, a 31-tap
    /// depthwise convolution, an inner LayerNorm, GELU, and a pointwise contraction. All three convolutions are
    /// bias-free in the released weights.</summary>
    private void Convolution(IBackend backend, Tensor input, Tensor output, Scratch scratch)
    {
        int tokens = (int)input.Shape[0];
        int dim = _config.Dim;
        int pad = _config.ConformerConvKernel / 2;
        backend.LayerNorm(scratch.Normed, input, _convNormWeight!, _convNormBias!, _config.LayerNormEps);
        backend.Transpose2D(scratch.ConvChannels, scratch.Normed, tokens, dim);
        backend.Conv1d(scratch.ConvGate, scratch.ConvChannels, _convGateWeight!, null,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);

        // GLU over the channel axis: the first half gated by a sigmoid of the second.
        Span<Tensor> halves = [scratch.ConvGateLeft, scratch.ConvGateRight];
        backend.Split(halves, scratch.ConvGate, dim: 1);
        backend.Sigmoid(scratch.ConvGated, scratch.ConvGateRight);
        backend.Mul(scratch.ConvChannels, scratch.ConvGateLeft, scratch.ConvGated);

        backend.Conv1d(scratch.ConvDepthwise, scratch.ConvChannels, _convDepthwiseWeight!, null,
            stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: dim);
        backend.Transpose2D(scratch.ConvTokens, scratch.ConvDepthwise, dim, tokens);
        backend.LayerNorm(scratch.Normed, scratch.ConvTokens, _convInnerNormWeight!, _convInnerNormBias!,
            _config.LayerNormEps);
        backend.Transpose2D(scratch.ConvChannels, scratch.Normed, tokens, dim);
        backend.GeluErf(scratch.ConvGated, scratch.ConvChannels);
        backend.Conv1d(scratch.ConvDepthwise, scratch.ConvGated, _convOutWeight!, null,
            stride: 1, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        backend.Transpose2D(scratch.ConvTokens, scratch.ConvDepthwise, dim, tokens);
        backend.Add(output, input, scratch.ConvTokens);
    }

    /// <summary>Buffers every Conformer layer reuses. One instance serves all 24 layers, so a forward allocates
    /// this once instead of ~20 tensors per layer.</summary>
    internal sealed class Scratch : IDisposable
    {
        public Scratch(Mert2Config config, int tokens)
        {
            int dim = config.Dim;
            int heads = config.Heads;
            int headDim = config.HeadDim;
            TensorShape tokenMajor = new(tokens, dim);
            TensorShape channelMajor = new(1, dim, tokens);
            TensorShape splitHeads = new(1, tokens, heads, headDim);
            TensorShape headMajor = new(1, heads, tokens, headDim);

            CarrierA = new Tensor(tokenMajor, DType.F32);
            CarrierB = new Tensor(tokenMajor, DType.F32);
            Normed = new Tensor(tokenMajor, DType.F32);
            Projected = new Tensor(tokenMajor, DType.F32);
            Attention = new Tensor(tokenMajor, DType.F32);
            Hidden = new Tensor(new TensorShape(tokens, config.Intermediate), DType.F32);
            Activated = new Tensor(new TensorShape(tokens, config.Intermediate), DType.F32);

            Query = new Tensor(splitHeads, DType.F32);
            Key = new Tensor(splitHeads, DType.F32);
            Value = new Tensor(splitHeads, DType.F32);
            QueryHeads = new Tensor(headMajor, DType.F32);
            KeyHeads = new Tensor(headMajor, DType.F32);
            ValueHeads = new Tensor(headMajor, DType.F32);
            AttentionHeads = new Tensor(headMajor, DType.F32);
            AttentionTokens = new Tensor(splitHeads, DType.F32);

            ConvChannels = new Tensor(channelMajor, DType.F32);
            ConvGate = new Tensor(new TensorShape(1, 2 * dim, tokens), DType.F32);
            ConvGateLeft = new Tensor(channelMajor, DType.F32);
            ConvGateRight = new Tensor(channelMajor, DType.F32);
            ConvGated = new Tensor(channelMajor, DType.F32);
            ConvDepthwise = new Tensor(channelMajor, DType.F32);
            ConvTokens = new Tensor(tokenMajor, DType.F32);

            RopeCos = new Tensor(new TensorShape(1, tokens, headDim), DType.F32);
            RopeSin = new Tensor(new TensorShape(1, tokens, headDim), DType.F32);
            Mert2Ops.BuildRopeTables(RopeCos, RopeSin, tokens, headDim, config.RopeTheta);
        }

        public Tensor CarrierA { get; }
        public Tensor CarrierB { get; }
        public Tensor Normed { get; }
        public Tensor Projected { get; }
        public Tensor Attention { get; }
        public Tensor Hidden { get; }
        public Tensor Activated { get; }
        public Tensor Query { get; }
        public Tensor Key { get; }
        public Tensor Value { get; }
        public Tensor QueryHeads { get; }
        public Tensor KeyHeads { get; }
        public Tensor ValueHeads { get; }
        public Tensor AttentionHeads { get; }
        public Tensor AttentionTokens { get; }
        public Tensor ConvChannels { get; }
        public Tensor ConvGate { get; }
        public Tensor ConvGateLeft { get; }
        public Tensor ConvGateRight { get; }
        public Tensor ConvGated { get; }
        public Tensor ConvDepthwise { get; }
        public Tensor ConvTokens { get; }
        public Tensor RopeCos { get; }
        public Tensor RopeSin { get; }

        public void Dispose()
        {
            CarrierA.Dispose();
            CarrierB.Dispose();
            Normed.Dispose();
            Projected.Dispose();
            Attention.Dispose();
            Hidden.Dispose();
            Activated.Dispose();
            Query.Dispose();
            Key.Dispose();
            Value.Dispose();
            QueryHeads.Dispose();
            KeyHeads.Dispose();
            ValueHeads.Dispose();
            AttentionHeads.Dispose();
            AttentionTokens.Dispose();
            ConvChannels.Dispose();
            ConvGate.Dispose();
            ConvGateLeft.Dispose();
            ConvGateRight.Dispose();
            ConvGated.Dispose();
            ConvDepthwise.Dispose();
            ConvTokens.Dispose();
            RopeCos.Dispose();
            RopeSin.Dispose();
        }
    }
}
