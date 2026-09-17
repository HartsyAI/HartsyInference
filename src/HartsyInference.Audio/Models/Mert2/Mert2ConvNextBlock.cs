using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Mert2;

/// <summary>One stage of MERT-v2's subsampling stack: an optional LayerNorm + strided convolution that changes the
/// channel count and frame rate, then a run of ConvNeXt-V2 layers at the new width. The first stage keeps the mel
/// frame rate (its resampling layer is an identity); the other two halve it, which is where the 4× between mel
/// frames and encoder tokens comes from.</summary>
internal sealed class Mert2ConvNextBlock : IDisposable
{
    private readonly Mert2Config _config;
    private readonly int _inChannels;
    private readonly int _outChannels;
    private readonly int _stride;
    private readonly Layer[] _layers;
    private readonly List<Tensor> _owned = [];

    private Tensor? _resampleNormWeight;
    private Tensor? _resampleNormBias;
    private Tensor? _resampleConvWeight;
    private Tensor? _resampleConvBias;
    private int _disposed;

    public Mert2ConvNextBlock(Mert2Config config, int inChannels, int outChannels, int stride, int depth)
    {
        _config = config;
        _inChannels = inChannels;
        _outChannels = outChannels;
        _stride = stride;
        _layers = new Layer[depth];
        for (int i = 0; i < depth; i++)
        {
            _layers[i] = new Layer(config, outChannels);
        }
    }

    /// <summary>Whether this block resamples; the released module substitutes an identity when the channel count
    /// and frame rate both stay put.</summary>
    public bool Resamples => _inChannels != _outChannels || _stride > 1;

    /// <summary>Frames this block emits for a given input length.</summary>
    public int OutputFrames(int frames)
        => Resamples && _stride > 1 ? (frames - _config.ResampleKernel) / _stride + 1 : frames;

    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        if (Resamples)
        {
            _resampleNormWeight = Mert2Ops.Load(weights, $"{prefix}.resampling_layer.0.weight", _owned);
            _resampleNormBias = Mert2Ops.Load(weights, $"{prefix}.resampling_layer.0.bias", _owned);
            _resampleConvWeight = Mert2Ops.Load(weights, $"{prefix}.resampling_layer.2.weight", _owned);
            _resampleConvBias = Mert2Ops.Load(weights, $"{prefix}.resampling_layer.2.bias", _owned);
        }
        for (int i = 0; i < _layers.Length; i++)
        {
            _layers[i].LoadWeights(weights, $"{prefix}.convnext_layers.{i}", _owned);
        }
    }

    /// <summary>Runs the block over <c>input [frames, inChannels]</c> and returns a freshly owned
    /// <c>[outFrames, outChannels]</c>. The input is only read, never written or disposed.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        int frames = (int)input.Shape[0];
        int outFrames = OutputFrames(frames);
        TensorShape shape = new(outFrames, _outChannels);
        Tensor first = new(shape, DType.F32);
        Tensor second = new(shape, DType.F32);
        using Scratch scratch = new(outFrames, _outChannels);
        try
        {
            Tensor current = input;
            if (Resamples)
            {
                Resample(backend, input, first, frames, outFrames);
                current = first;
            }
            for (int i = 0; i < _layers.Length; i++)
            {
                Tensor destination = ReferenceEquals(current, first) ? second : first;
                _layers[i].Forward(backend, current, destination, scratch);
                current = destination;
            }
            if (ReferenceEquals(current, first))
            {
                second.Dispose();
                return first;
            }
            first.Dispose();
            return second;
        }
        catch
        {
            first.Dispose();
            second.Dispose();
            throw;
        }
    }

    /// <summary>LayerNorm over the channel axis, then a stride-2 kernel-2 convolution that also widens.</summary>
    private void Resample(IBackend backend, Tensor input, Tensor output, int frames, int outFrames)
    {
        using Tensor normed = new(new TensorShape(frames, _inChannels), DType.F32);
        backend.LayerNorm(normed, input, _resampleNormWeight!, _resampleNormBias!, _config.ConvNextNormEps);
        using Tensor channelsFirst = new(new TensorShape(1, _inChannels, frames), DType.F32);
        backend.Transpose2D(channelsFirst, normed, frames, _inChannels);
        using Tensor convolved = new(new TensorShape(1, _outChannels, outFrames), DType.F32);
        backend.Conv1d(convolved, channelsFirst, _resampleConvWeight!, _resampleConvBias,
            stride: _stride, padLeft: 0, padRight: 0, dilation: 1, groups: 1);
        backend.Transpose2D(output, convolved, _outChannels, outFrames);
    }

    /// <summary>Every weight the backend should preload; the GlobalResponseNorm gains are deliberately absent
    /// because they are folded on the host, never handed to a device op.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        if (_resampleNormWeight is not null) yield return _resampleNormWeight;
        if (_resampleNormBias is not null) yield return _resampleNormBias;
        if (_resampleConvWeight is not null) yield return _resampleConvWeight;
        if (_resampleConvBias is not null) yield return _resampleConvBias;
        for (int i = 0; i < _layers.Length; i++)
        {
            foreach (Tensor weight in _layers[i].EnumerateWeights()) yield return weight;
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        for (int i = 0; i < _owned.Count; i++) _owned[i].Dispose();
        _owned.Clear();
    }

    /// <summary>Buffers a block reuses across its ConvNeXt layers; all of them are written by backend ops, so one
    /// allocation per block serves every layer without the host ever mutating a cached tensor.</summary>
    private sealed class Scratch : IDisposable
    {
        public Scratch(int frames, int channels)
        {
            ChannelsFirst = new Tensor(new TensorShape(1, channels, frames), DType.F32);
            Depthwise = new Tensor(new TensorShape(1, channels, frames), DType.F32);
            DepthwiseOut = new Tensor(new TensorShape(frames, channels), DType.F32);
            Normed = new Tensor(new TensorShape(frames, channels), DType.F32);
            Contracted = new Tensor(new TensorShape(frames, channels), DType.F32);
            Hidden = new Tensor(new TensorShape(frames, 4 * channels), DType.F32);
            Activated = new Tensor(new TensorShape(frames, 4 * channels), DType.F32);
            Squared = new Tensor(new TensorShape(frames, 4 * channels), DType.F32);
            Ones = new Tensor(new TensorShape(1, frames), DType.F32);
            Ones.AsSpan<float>().Fill(1f);
        }

        public Tensor ChannelsFirst { get; }
        public Tensor Depthwise { get; }
        public Tensor DepthwiseOut { get; }
        public Tensor Normed { get; }
        public Tensor Contracted { get; }
        public Tensor Hidden { get; }
        public Tensor Activated { get; }
        public Tensor Squared { get; }
        public Tensor Ones { get; }

        public void Dispose()
        {
            ChannelsFirst.Dispose();
            Depthwise.Dispose();
            DepthwiseOut.Dispose();
            Normed.Dispose();
            Contracted.Dispose();
            Hidden.Dispose();
            Activated.Dispose();
            Squared.Dispose();
            Ones.Dispose();
        }
    }

    /// <summary>A ConvNeXt-V2 layer: depthwise 7-tap convolution, then LayerNorm, a 4× inverted bottleneck with
    /// GELU and GlobalResponseNorm, and a residual add.</summary>
    private sealed class Layer
    {
        private readonly Mert2Config _config;
        private readonly int _channels;
        private readonly float[] _responseNormWeight;

        private Tensor? _depthwiseWeight;
        private Tensor? _depthwiseBias;
        private Tensor? _normWeight;
        private Tensor? _normBias;
        private Tensor? _expandWeight;
        private Tensor? _expandBias;
        private Tensor? _responseNormBias;
        private Tensor? _contractWeight;
        private Tensor? _contractBias;

        public Layer(Mert2Config config, int channels)
        {
            _config = config;
            _channels = channels;
            _responseNormWeight = new float[4 * channels];
        }

        public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights, string prefix, List<Tensor> owned)
        {
            _depthwiseWeight = Mert2Ops.Load(weights, $"{prefix}.depthwise_block.1.weight", owned);
            _depthwiseBias = Mert2Ops.Load(weights, $"{prefix}.depthwise_block.1.bias", owned);
            _normWeight = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.0.weight", owned);
            _normBias = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.0.bias", owned);
            _expandWeight = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.1.weight", owned);
            _expandBias = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.1.bias", owned);
            Mert2Ops.Load(weights, $"{prefix}.pointwise_block.3.weight", owned)
                .AsReadOnlySpan<float>().CopyTo(_responseNormWeight);
            _responseNormBias = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.3.bias", owned);
            _contractWeight = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.4.weight", owned);
            _contractBias = Mert2Ops.Load(weights, $"{prefix}.pointwise_block.4.bias", owned);
        }

        public void Forward(IBackend backend, Tensor input, Tensor output, Scratch scratch)
        {
            int frames = (int)input.Shape[0];
            int pad = _config.ConvNextKernel / 2;
            backend.Transpose2D(scratch.ChannelsFirst, input, frames, _channels);
            backend.Conv1d(scratch.Depthwise, scratch.ChannelsFirst, _depthwiseWeight!, _depthwiseBias,
                stride: 1, padLeft: pad, padRight: pad, dilation: 1, groups: _channels);
            backend.Transpose2D(scratch.DepthwiseOut, scratch.Depthwise, _channels, frames);

            backend.LayerNorm(scratch.Normed, scratch.DepthwiseOut, _normWeight!, _normBias!, _config.ConvNextNormEps);
            backend.Linear(scratch.Hidden, scratch.Normed, _expandWeight!, _expandBias);
            backend.GeluErf(scratch.Activated, scratch.Hidden);
            Mert2Ops.GlobalResponseNorm(backend, scratch.Hidden, scratch.Activated, _responseNormWeight,
                _responseNormBias!, scratch.Ones, scratch.Squared);
            backend.Linear(scratch.Contracted, scratch.Hidden, _contractWeight!, _contractBias);
            // The skip spans the whole layer, depthwise convolution included.
            backend.Add(output, scratch.Contracted, input);
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            if (_depthwiseWeight is not null) yield return _depthwiseWeight;
            if (_depthwiseBias is not null) yield return _depthwiseBias;
            if (_normWeight is not null) yield return _normWeight;
            if (_normBias is not null) yield return _normBias;
            if (_expandWeight is not null) yield return _expandWeight;
            if (_expandBias is not null) yield return _expandBias;
            if (_responseNormBias is not null) yield return _responseNormBias;
            if (_contractWeight is not null) yield return _contractWeight;
            if (_contractBias is not null) yield return _contractBias;
        }
    }
}
