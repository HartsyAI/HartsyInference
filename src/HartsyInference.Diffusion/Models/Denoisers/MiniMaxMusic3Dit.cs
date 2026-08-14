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
            float* destination = (float*)hidden.DataPointer;
            temb.AsReadOnlySpan<float>().CopyTo(new Span<float>(destination, inner));
            projected.AsReadOnlySpan<float>().CopyTo(new Span<float>(destination + inner, length * inner));
        }

        (Tensor cos, Tensor sin) = RopeTables(length + 1);
        foreach (Block block in _blocks)
        {
            Tensor next = block.Forward(backend, hidden, _config, cos, sin);
            hidden.Dispose();
            hidden = next;
        }

        using Tensor body = new Tensor(new TensorShape(length, inner), DType.F32);
        hidden.AsReadOnlySpan<float>()[inner..].CopyTo(new Span<float>((float*)body.DataPointer, length * inner));
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
        Tensor current = hidden;
        foreach (Block block in _blocks)
        {
            Tensor next = block.Forward(backend, current, _config, cos, sin);
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

        public Tensor Forward(IBackend backend, Tensor hidden, MiniMaxMusic3DitConfig config, Tensor cos, Tensor sin)
        {
            int rows = (int)hidden.Shape[0];
            int inner = config.InnerDim;
            int heads = config.NumAttentionHeads;
            int headDim = config.AttentionHeadDim;

            Tensor result = new Tensor(hidden.Shape, DType.F32);
            using (Tensor normed = new Tensor(hidden.Shape, DType.F32))
            {
                backend.LayerNorm(normed, hidden, _norm1Weight!, _norm1Bias!, config.LayerNormEps);
                // Allocated at the rank-4 shape the rotary op needs rather than reshaped into it: Tensor.Reshape
                // reads DataPointer, which syncs a device tensor back to the host and hands out a HOST pointer, so
                // the view loses GPU residency. Roping through such a view left the device copy un-rotated and the
                // attention ran without rotary on CUDA while CPU was correct.
                TensorShape headed = new TensorShape(1, rows, heads, headDim);
                using Tensor query = new Tensor(headed, DType.F32);
                using Tensor key = new Tensor(headed, DType.F32);
                using Tensor value = new Tensor(headed, DType.F32);
                backend.Linear(query, normed, _toQ!, null);
                backend.Linear(key, normed, _toK!, null);
                backend.Linear(value, normed, _toV!, null);

                backend.ApplyRopeSingle(query, cos, sin, config.RotaryDim);
                backend.ApplyRopeSingle(key, cos, sin, config.RotaryDim);

                using Tensor attention = new Tensor(headed, DType.F32);
                Attend(backend, attention, query, key, value, rows, heads, headDim);
                backend.Linear(result, attention, _toOut!, null);
            }
            backend.Add(result, result, hidden);

            using Tensor normed2 = new Tensor(hidden.Shape, DType.F32);
            backend.LayerNorm(normed2, result, _norm2Weight!, _norm2Bias!, config.LayerNormEps);
            using Tensor gated = new Tensor(new TensorShape(rows, 2 * config.FfInnerDim), DType.F32);
            backend.Linear(gated, normed2, _ffInWeight!, _ffInBias);
            using Tensor states = new Tensor(new TensorShape(rows, config.FfInnerDim), DType.F32);
            using Tensor gate = new Tensor(new TensorShape(rows, config.FfInnerDim), DType.F32);
            backend.Split([states, gate], gated, dim: 1);
            backend.Silu(gate, gate);
            backend.Mul(states, states, gate);
            using Tensor projected = new Tensor(hidden.Shape, DType.F32);
            backend.Linear(projected, states, _ffOutWeight!, _ffOutBias);
            backend.Add(result, result, projected);
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
        private static void Attend(IBackend backend, Tensor output, Tensor query, Tensor key, Tensor value,
            int rows, int heads, int headDim)
        {
            float scale = 1f / MathF.Sqrt(headDim);
            TensorShape headMajor = new TensorShape(1, heads, rows, headDim);
            using Tensor q = new Tensor(headMajor, DType.F32);
            using Tensor k = new Tensor(headMajor, DType.F32);
            using Tensor v = new Tensor(headMajor, DType.F32);
            backend.Permute0213(q, query, rows, heads, headDim);
            backend.Permute0213(k, key, rows, heads, headDim);
            backend.Permute0213(v, value, rows, heads, headDim);
            using Tensor attention = new Tensor(headMajor, DType.F32);
            backend.ScaledDotProductAttention(attention, q, k, v, null, scale);
            backend.Permute0213(output, attention, heads, rows, headDim);
        }
    }
}
