using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MiniMax Music 3's flow-matching transformer: it predicts the velocity of Flow-VAE audio latents
/// conditioned on the latent-aligned hidden states the condition encoder produces. Flow-matching time runs from
/// 0 (noise) to 1 (data) and is consumed directly as the <c>timestep</c>.
///
/// <para>Everything between the input and output transposes is token-major <c>[1+length, innerDim]</c>: the
/// pre/post-process convolutions are kernel-1, so they are per-position linears, and that is also the layout the
/// attention and feed-forward projections already emit. The timestep embedding rides as an extra leading token that
/// is dropped after the blocks.</para></summary>
public sealed unsafe class MiniMaxMusic3Dit : IDisposable
{
    private readonly MiniMaxMusic3DitConfig _config;
    private readonly Block[] _blocks;
    private readonly List<Tensor> _owned = [];

    private Tensor? _timeProj;
    private Tensor? _timeLinear1Weight;
    private Tensor? _timeLinear1Bias;
    private Tensor? _timeLinear2Weight;
    private Tensor? _timeLinear2Bias;
    private Tensor? _preprocessConv;
    private Tensor? _projIn;
    private Tensor? _projOut;
    private Tensor? _postprocessConv;
    private Tensor? _ropeCos;
    private Tensor? _ropeSin;
    private int _ropeLength;
    private BlockScratch? _scratch;
    private int _disposed;

    public MiniMaxMusic3Dit(MiniMaxMusic3DitConfig? config = null)
    {
        _config = config ?? MiniMaxMusic3DitConfig.Default;
        _blocks = new Block[_config.NumLayers];
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i] = new Block();
        }
    }

    /// <summary>Reads the checkpoint, reshaping the two kernel-1 convolutions into the rank-2 linears they are.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _timeProj = weights["time_proj.weight"];
        _timeLinear1Weight = weights["time_embed.linear_1.weight"];
        _timeLinear1Bias = weights["time_embed.linear_1.bias"];
        _timeLinear2Weight = weights["time_embed.linear_2.weight"];
        _timeLinear2Bias = weights["time_embed.linear_2.bias"];
        _preprocessConv = AsLinear(weights["preprocess_conv.weight"]);
        _projIn = weights["proj_in.weight"];
        _projOut = weights["proj_out.weight"];
        _postprocessConv = AsLinear(weights["postprocess_conv.weight"]);
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].Load(weights, $"transformer_blocks.{i}");
        }
    }

    /// <summary>Predicts the flow-matching velocity for <paramref name="latents"/> <c>[1, inChannels, length]</c> at
    /// <paramref name="timestep"/> in <c>[0, 1]</c>, conditioned on <paramref name="condition"/>
    /// <c>[1, length, conditionDim]</c>. Pass a zero-filled condition for the unconditional branch — the reference
    /// conditions on zeros rather than re-encoding an empty prompt. Caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor latents, float timestep, Tensor condition)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latents);
        ArgumentNullException.ThrowIfNull(condition);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_projIn is null)
        {
            throw new InvalidOperationException($"{nameof(MiniMaxMusic3Dit)}.{nameof(LoadWeights)} must run before {nameof(Forward)}.");
        }
        int channels = _config.InChannels;
        if (latents.Shape.Rank != 3 || latents.Shape[0] != 1 || latents.Shape[1] != channels)
        {
            throw new ArgumentException($"expected latents [1, {channels}, length], got {latents.Shape}.", nameof(latents));
        }
        int length = (int)latents.Shape[2];
        if (condition.Shape.Rank != 3 || condition.Shape[1] != length || condition.Shape[2] != _config.ConditionDim)
        {
            throw new ArgumentException(
                $"expected condition [1, {length}, {_config.ConditionDim}], got {condition.Shape}.", nameof(condition));
        }

        int concat = _config.ConcatChannels;
        int inner = _config.InnerDim;
        using Tensor tokens = new Tensor(new TensorShape(length, concat), DType.F32);
        BuildTokens(backend, tokens, latents, condition, length);

        // preprocess_conv is kernel-1: a per-position linear plus the residual the reference adds around it.
        using Tensor preprocessed = new Tensor(new TensorShape(length, concat), DType.F32);
        backend.Linear(preprocessed, tokens, _preprocessConv!, null);
        backend.Add(preprocessed, preprocessed, tokens);

        Tensor hidden = new Tensor(new TensorShape(length + 1, inner), DType.F32);
        using (Tensor projected = new Tensor(new TensorShape(length, inner), DType.F32))
        {
            backend.Linear(projected, preprocessed, _projIn!, null);
            using Tensor temb = EmbedTimestep(backend, timestep);
            // Concat on the backend rather than two host copies: reading DataPointer here synced the whole block
            // input back from the device on every forward.
            backend.Concat(hidden, [temb, projected], dim: 0);
        }

        (Tensor cos, Tensor sin) = RopeTables(length + 1);
        BlockScratch scratch = Scratch(length + 1);
        foreach (Block block in _blocks)
        {
            Tensor next = block.Forward(backend, hidden, _config, cos, sin, scratch);
            hidden.Dispose();
            hidden = next;
        }

        using Tensor body = new Tensor(new TensorShape(length, inner), DType.F32);
        backend.SliceRows(body, hidden, rowOffset: 1);
        hidden.Dispose();

        using Tensor output = new Tensor(new TensorShape(length, channels), DType.F32);
        backend.Linear(output, body, _projOut!, null);
        using Tensor post = new Tensor(new TensorShape(1, length, channels), DType.F32);
        backend.Linear(post, output, _postprocessConv!, null);
        backend.Add(post, post, output);

        Tensor velocity = new Tensor(new TensorShape(1, channels, length), DType.F32);
        backend.Transpose2D(velocity, post, length, channels);
        return velocity;
    }

    /// <summary>Runs only the transformer blocks over an already-projected token-major hidden state
    /// <c>[1+length, innerDim]</c>. Exists so a parity run can bisect to the first divergent block instead of
    /// comparing the whole 36-layer stack at once.</summary>
    internal Tensor ForwardBlocks(IBackend backend, Tensor hidden)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hidden);
        (Tensor cos, Tensor sin) = RopeTables((int)hidden.Shape[0]);
        BlockScratch scratch = Scratch((int)hidden.Shape[0]);
        Tensor current = hidden;
        foreach (Block block in _blocks)
        {
            Tensor next = block.Forward(backend, current, _config, cos, sin, scratch);
            if (!ReferenceEquals(current, hidden))
            {
                current.Dispose();
            }
            current = next;
        }
        return current;
    }

    /// <summary>Every loaded weight, for <see cref="IBackend.PreloadWeights"/>/<see cref="IBackend.FreeWeights"/>.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] top =
        [
            _timeProj, _timeLinear1Weight, _timeLinear1Bias, _timeLinear2Weight, _timeLinear2Bias,
            _preprocessConv, _projIn, _projOut, _postprocessConv,
        ];
        foreach (Tensor? tensor in top)
        {
            if (tensor is not null) { yield return tensor; }
        }
        foreach (Block block in _blocks)
        {
            foreach (Tensor tensor in block.EnumerateWeights())
            {
                yield return tensor;
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        foreach (Tensor tensor in _owned)
        {
            tensor.Dispose();
        }
        _owned.Clear();
        _ropeCos?.Dispose();
        _ropeSin?.Dispose();
        _ropeCos = null;
        _ropeSin = null;
        _scratch?.Dispose();
        _scratch = null;
    }

    /// <summary>Lays out <c>[latent, zeros(latent), condition]</c> token-major, the channel concatenation the
    /// reference performs before the kernel-1 preprocess convolution.</summary>
    private void BuildTokens(IBackend backend, Tensor tokens, Tensor latents, Tensor condition, int length)
    {
        int channels = _config.InChannels;
        int concat = _config.ConcatChannels;
        using Tensor latentsTokenMajor = new Tensor(new TensorShape(1, length, channels), DType.F32);
        backend.Transpose2D(latentsTokenMajor, latents, channels, length);

        float* destination = (float*)tokens.DataPointer;
        new Span<float>(destination, length * concat).Clear();
        ReadOnlySpan<float> latentValues = latentsTokenMajor.AsReadOnlySpan<float>();
        ReadOnlySpan<float> conditionValues = condition.AsReadOnlySpan<float>();
        for (int position = 0; position < length; position++)
        {
            long row = (long)position * concat;
            latentValues.Slice(position * channels, channels).CopyTo(new Span<float>(destination + row, channels));
            conditionValues.Slice(position * _config.ConditionDim, _config.ConditionDim)
                .CopyTo(new Span<float>(destination + row + (2 * channels), _config.ConditionDim));
        }
    }

    /// <summary>Random Fourier features of the flow-matching time, then the two-layer timestep MLP.</summary>
    private Tensor EmbedTimestep(IBackend backend, float timestep)
    {
        int fourier = _config.FourierEmbeddingDim;
        int half = fourier / 2;
        using Tensor features = new Tensor(new TensorShape(1, fourier), DType.F32);
        float* destination = (float*)features.DataPointer;
        ReadOnlySpan<float> projection = _timeProj!.AsReadOnlySpan<float>();
        for (int i = 0; i < half; i++)
        {
            float angle = 2f * MathF.PI * timestep * projection[i];
            destination[i] = MathF.Cos(angle);
            destination[half + i] = MathF.Sin(angle);
        }
        using Tensor first = new Tensor(new TensorShape(1, _config.InnerDim), DType.F32);
        backend.Linear(first, features, _timeLinear1Weight!, _timeLinear1Bias);
        backend.Silu(first, first);
        Tensor second = new Tensor(new TensorShape(1, _config.InnerDim), DType.F32);
        backend.Linear(second, first, _timeLinear2Weight!, _timeLinear2Bias);
        return second;
    }

    /// <summary>Partial-rotary cos/sin tables, cached across steps. Only the leading <c>RotaryDim</c> entries of each
    /// head-width row are populated; <see cref="IBackend.ApplyRopeSingle"/> reads no further.</summary>
    private (Tensor Cos, Tensor Sin) RopeTables(int sequenceLength)
    {
        if (_ropeCos is not null && _ropeLength >= sequenceLength)
        {
            return (_ropeCos, _ropeSin!);
        }
        _ropeCos?.Dispose();
        _ropeSin?.Dispose();
        int headDim = _config.AttentionHeadDim;
        int rotary = _config.RotaryDim;
        int half = rotary / 2;
        Tensor cos = new Tensor(new TensorShape(1, sequenceLength, headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(1, sequenceLength, headDim), DType.F32);
        float* cosData = (float*)cos.DataPointer;
        float* sinData = (float*)sin.DataPointer;
        new Span<float>(cosData, sequenceLength * headDim).Clear();
        new Span<float>(sinData, sequenceLength * headDim).Clear();
        for (int position = 0; position < sequenceLength; position++)
        {
            long row = (long)position * headDim;
            for (int i = 0; i < half; i++)
            {
                float inverseFrequency = 1f / MathF.Pow(_config.RopeTheta, 2f * i / rotary);
                float angle = position * inverseFrequency;
                // The reference concatenates the frequency block with itself, so the two halves share an angle.
                cosData[row + i] = MathF.Cos(angle);
                cosData[row + half + i] = cosData[row + i];
                sinData[row + i] = MathF.Sin(angle);
                sinData[row + half + i] = sinData[row + i];
            }
        }
        _ropeCos = cos;
        _ropeSin = sin;
        _ropeLength = sequenceLength;
        return (cos, sin);
    }

    /// <summary>Per-block working tensors, allocated once per sequence length and reused by every block and every
    /// forward. Allocating fourteen device buffers per block, 36 blocks a forward, fragmented the CUDA pool badly
    /// enough to OOM a 12 GB card on longer generations.</summary>
    private BlockScratch Scratch(int rows)
    {
        if (_scratch is not null && _scratch.Rows == rows)
        {
            return _scratch;
        }
        _scratch?.Dispose();
        _scratch = new BlockScratch(rows, _config);
        return _scratch;
    }

    private sealed class BlockScratch : IDisposable
    {
        private readonly List<Tensor> _owned = [];

        internal BlockScratch(int rows, MiniMaxMusic3DitConfig config)
        {
            Rows = rows;
            TensorShape flat = new TensorShape(rows, config.InnerDim);
            TensorShape tokenMajor = new TensorShape(1, rows, config.NumAttentionHeads, config.AttentionHeadDim);
            TensorShape headMajor = new TensorShape(1, config.NumAttentionHeads, rows, config.AttentionHeadDim);
            TensorShape wide = new TensorShape(rows, config.FfInnerDim);
            Normed = Track(flat);
            Query = Track(tokenMajor);
            Key = Track(tokenMajor);
            Value = Track(tokenMajor);
            Attention = Track(tokenMajor);
            HeadQuery = Track(headMajor);
            HeadKey = Track(headMajor);
            HeadValue = Track(headMajor);
            HeadAttention = Track(headMajor);
            Normed2 = Track(flat);
            Gated = Track(new TensorShape(rows, 2 * config.FfInnerDim));
            States = Track(wide);
            Gate = Track(wide);
            Projected = Track(flat);
        }

        internal int Rows { get; }
        internal Tensor Normed { get; }
        internal Tensor Query { get; }
        internal Tensor Key { get; }
        internal Tensor Value { get; }
        internal Tensor Attention { get; }
        internal Tensor HeadQuery { get; }
        internal Tensor HeadKey { get; }
        internal Tensor HeadValue { get; }
        internal Tensor HeadAttention { get; }
        internal Tensor Normed2 { get; }
        internal Tensor Gated { get; }
        internal Tensor States { get; }
        internal Tensor Gate { get; }
        internal Tensor Projected { get; }

        private Tensor Track(TensorShape shape)
        {
            Tensor tensor = new Tensor(shape, DType.F32);
            _owned.Add(tensor);
            return tensor;
        }

        public void Dispose()
        {
            foreach (Tensor tensor in _owned)
            {
                tensor.Dispose();
            }
            _owned.Clear();
        }
    }

    private Tensor AsLinear(Tensor kernelOneConv)
    {
        if (kernelOneConv.Shape.Rank != 3 || kernelOneConv.Shape[2] != 1)
        {
            throw new ArgumentException($"expected a kernel-1 convolution weight [out, in, 1], got {kernelOneConv.Shape}.");
        }
        return kernelOneConv.Reshape(new TensorShape((int)kernelOneConv.Shape[0], (int)kernelOneConv.Shape[1]));
    }

    /// <summary>One transformer block: pre-norm attention with partial rotary, then a gated feed-forward whose gate
    /// is the SECOND half of <c>ff_in</c> — <c>first · silu(second)</c>, the reverse of the usual convention.</summary>
    private sealed class Block
    {
        private Tensor? _norm1Weight;
        private Tensor? _norm1Bias;
        private Tensor? _norm2Weight;
        private Tensor? _norm2Bias;
        private Tensor? _toQ;
        private Tensor? _toK;
        private Tensor? _toV;
        private Tensor? _toOut;
        private Tensor? _ffInWeight;
        private Tensor? _ffInBias;
        private Tensor? _ffOutWeight;
        private Tensor? _ffOutBias;

        public void Load(IReadOnlyDictionary<string, Tensor> weights, string prefix)
        {
            _norm1Weight = weights[$"{prefix}.norm1.weight"];
            _norm1Bias = weights[$"{prefix}.norm1.bias"];
            _norm2Weight = weights[$"{prefix}.norm2.weight"];
            _norm2Bias = weights[$"{prefix}.norm2.bias"];
            _toQ = weights[$"{prefix}.attn.to_q.weight"];
            _toK = weights[$"{prefix}.attn.to_k.weight"];
            _toV = weights[$"{prefix}.attn.to_v.weight"];
            _toOut = weights[$"{prefix}.attn.to_out.0.weight"];
            _ffInWeight = weights[$"{prefix}.ff_in.weight"];
            _ffInBias = weights[$"{prefix}.ff_in.bias"];
            _ffOutWeight = weights[$"{prefix}.ff_out.weight"];
            _ffOutBias = weights[$"{prefix}.ff_out.bias"];
        }

        public Tensor Forward(IBackend backend, Tensor hidden, MiniMaxMusic3DitConfig config, Tensor cos, Tensor sin,
            BlockScratch scratch)
        {
            int rows = (int)hidden.Shape[0];
            int heads = config.NumAttentionHeads;
            int headDim = config.AttentionHeadDim;

            Tensor result = new Tensor(hidden.Shape, DType.F32);
            backend.LayerNorm(scratch.Normed, hidden, _norm1Weight!, _norm1Bias!, config.LayerNormEps);
            backend.Linear(scratch.Query, scratch.Normed, _toQ!, null);
            backend.Linear(scratch.Key, scratch.Normed, _toK!, null);
            backend.Linear(scratch.Value, scratch.Normed, _toV!, null);
            backend.ApplyRopeSingle(scratch.Query, cos, sin, config.RotaryDim);
            backend.ApplyRopeSingle(scratch.Key, cos, sin, config.RotaryDim);
            Attend(backend, scratch, rows, heads, headDim);
            backend.Linear(result, scratch.Attention, _toOut!, null);
            backend.Add(result, result, hidden);

            backend.LayerNorm(scratch.Normed2, result, _norm2Weight!, _norm2Bias!, config.LayerNormEps);
            backend.Linear(scratch.Gated, scratch.Normed2, _ffInWeight!, _ffInBias);
            backend.Split([scratch.States, scratch.Gate], scratch.Gated, dim: 1);
            backend.Silu(scratch.Gate, scratch.Gate);
            backend.Mul(scratch.States, scratch.States, scratch.Gate);
            backend.Linear(scratch.Projected, scratch.States, _ffOutWeight!, _ffOutBias);
            backend.Add(result, result, scratch.Projected);
            return result;
        }

        public IEnumerable<Tensor> EnumerateWeights()
        {
            Tensor?[] all =
            [
                _norm1Weight, _norm1Bias, _norm2Weight, _norm2Bias,
                _toQ, _toK, _toV, _toOut, _ffInWeight, _ffInBias, _ffOutWeight, _ffOutBias,
            ];
            foreach (Tensor? tensor in all)
            {
                if (tensor is not null) { yield return tensor; }
            }
        }

        /// <summary>Head-major attention over rank-4 <c>[1, rows, heads, headDim]</c> tensors. The token-major
        /// entry point is deliberately not used: it wants rank-2, and getting there from the rotary op's rank-4
        /// layout requires a reshape, which is what broke GPU residency.</summary>
        /// <summary>Head-major attention over the shared rank-4 scratch. The token-major entry point is not used:
        /// it wants rank-2, and reaching that from the rotary op's rank-4 layout needs a reshape, which silently
        /// drops GPU residency.</summary>
        private static void Attend(IBackend backend, BlockScratch scratch, int rows, int heads, int headDim)
        {
            backend.Permute0213(scratch.HeadQuery, scratch.Query, rows, heads, headDim);
            backend.Permute0213(scratch.HeadKey, scratch.Key, rows, heads, headDim);
            backend.Permute0213(scratch.HeadValue, scratch.Value, rows, heads, headDim);
            backend.ScaledDotProductAttention(scratch.HeadAttention, scratch.HeadQuery, scratch.HeadKey,
                scratch.HeadValue, null, 1f / MathF.Sqrt(headDim));
            backend.Permute0213(scratch.Attention, scratch.HeadAttention, heads, rows, headDim);
        }
    }
}
