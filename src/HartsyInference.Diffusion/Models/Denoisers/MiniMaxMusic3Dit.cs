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
    private int _ropeBatch;
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
        ArgumentNullException.ThrowIfNull(condition);
        int length = ValidateForward(backend, latents, condition);
        int inner = _config.InnerDim;
        using Tensor latentsTokenMajor = TokenMajorLatents(backend, latents, length);
        Tensor hidden = new Tensor(new TensorShape(length + 1, inner), DType.F32);
        using (Tensor projected = ProjectBranch(backend, latentsTokenMajor, condition, length))
        using (Tensor temb = EmbedTimestep(backend, timestep))
        {
            // Concat on the backend rather than two host copies: reading DataPointer here synced the whole block
            // input back from the device on every forward.
            backend.Concat(hidden, [temb, projected], dim: 0);
        }

        Tensor blocks = RunBlocks(backend, hidden, length + 1, batch: 1);
        hidden.Dispose();
        using Tensor body = new Tensor(new TensorShape(length, inner), DType.F32);
        backend.SliceRows(body, blocks, rowOffset: 1);
        blocks.Dispose();
        return ProjectVelocity(backend, body, length);
    }

    /// <summary>Batch-2 twin of <see cref="Forward"/>: the conditional branch and the zero-conditioned unconditional
    /// branch run as one row-stacked pass, so the 36 blocks amortize their weight reads and pay one kernel launch
    /// where the two separate forwards paid two. Returns the two velocities in that order; the caller owns both.
    ///
    /// <para>The unconditional branch is built here rather than taken as an argument precisely so it stays the
    /// reference's ZERO conditioning and can never drift into a re-encoded empty prompt.</para></summary>
    public (Tensor Conditional, Tensor Unconditional) ForwardCfg(IBackend backend, Tensor latents, float timestep, Tensor condition)
    {
        ArgumentNullException.ThrowIfNull(condition);
        int length = ValidateForward(backend, latents, condition);
        int inner = _config.InnerDim;
        int rows = length + 1;
        using Tensor latentsTokenMajor = TokenMajorLatents(backend, latents, length);
        Tensor hidden = new Tensor(new TensorShape(2 * rows, inner), DType.F32);
        using (Tensor conditional = ProjectBranch(backend, latentsTokenMajor, condition, length))
        using (Tensor unconditional = ProjectBranch(backend, latentsTokenMajor, null, length))
        using (Tensor temb = EmbedTimestep(backend, timestep))
        {
            // Both branches share the timestep token, and both need it at their own row 0 — hence four pieces.
            backend.Concat(hidden, [temb, conditional, temb, unconditional], dim: 0);
        }

        Tensor blocks = RunBlocks(backend, hidden, rows, batch: 2);
        hidden.Dispose();
        using Tensor bodyConditional = new Tensor(new TensorShape(length, inner), DType.F32);
        using Tensor bodyUnconditional = new Tensor(new TensorShape(length, inner), DType.F32);
        try
        {
            backend.SliceRows(bodyConditional, blocks, rowOffset: 1);
            backend.SliceRows(bodyUnconditional, blocks, rowOffset: rows + 1);
        }
        finally
        {
            blocks.Dispose();
        }

        Tensor conditionalVelocity = ProjectVelocity(backend, bodyConditional, length);
        try
        {
            return (conditionalVelocity, ProjectVelocity(backend, bodyUnconditional, length));
        }
        catch
        {
            conditionalVelocity.Dispose();
            throw;
        }
    }

    /// <summary>Runs only the transformer blocks over an already-projected token-major hidden state
    /// <c>[1+length, innerDim]</c>. Exists so a parity run can bisect to the first divergent block instead of
    /// comparing the whole 36-layer stack at once.</summary>
    internal Tensor ForwardBlocks(IBackend backend, Tensor hidden)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(hidden);
        return RunBlocks(backend, hidden, (int)hidden.Shape[0], batch: 1);
    }

    /// <summary>Walks the block stack over <paramref name="batch"/> row-blocks of <paramref name="rows"/> tokens each.
    /// Never disposes <paramref name="hidden"/> — the caller owns it.</summary>
    private Tensor RunBlocks(IBackend backend, Tensor hidden, int rows, int batch)
    {
        (Tensor cos, Tensor sin) = RopeTables(rows, batch);
        BlockScratch scratch = Scratch(rows, batch);
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

    private int ValidateForward(IBackend backend, Tensor latents, Tensor condition)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latents);
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
        return length;
    }

    private Tensor TokenMajorLatents(IBackend backend, Tensor latents, int length)
    {
        Tensor tokenMajor = new Tensor(new TensorShape(1, length, _config.InChannels), DType.F32);
        backend.Transpose2D(tokenMajor, latents, _config.InChannels, length);
        return tokenMajor;
    }

    /// <summary>One branch's block input: token concatenation, the kernel-1 preprocess convolution and its residual,
    /// then <c>proj_in</c>. A null <paramref name="condition"/> is the unconditional branch, whose conditioning is
    /// zeros rather than a re-encoded empty prompt.</summary>
    private Tensor ProjectBranch(IBackend backend, Tensor latentsTokenMajor, Tensor? condition, int length)
    {
        int concat = _config.ConcatChannels;
        using Tensor tokens = new Tensor(new TensorShape(length, concat), DType.F32);
        BuildTokens(tokens, latentsTokenMajor, condition, length);
        using Tensor preprocessed = new Tensor(new TensorShape(length, concat), DType.F32);
        backend.Linear(preprocessed, tokens, _preprocessConv!, null);
        backend.Add(preprocessed, preprocessed, tokens);
        Tensor projected = new Tensor(new TensorShape(length, _config.InnerDim), DType.F32);
        backend.Linear(projected, preprocessed, _projIn!, null);
        return projected;
    }

    /// <summary><c>proj_out</c>, the kernel-1 postprocess convolution and its residual, then back to channel-major.</summary>
    private Tensor ProjectVelocity(IBackend backend, Tensor body, int length)
    {
        int channels = _config.InChannels;
        using Tensor output = new Tensor(new TensorShape(length, channels), DType.F32);
        backend.Linear(output, body, _projOut!, null);
        using Tensor post = new Tensor(new TensorShape(1, length, channels), DType.F32);
        backend.Linear(post, output, _postprocessConv!, null);
        backend.Add(post, post, output);
        Tensor velocity = new Tensor(new TensorShape(1, channels, length), DType.F32);
        backend.Transpose2D(velocity, post, length, channels);
        return velocity;
    }

    /// <summary>Lays out <c>[latent, zeros(latent), condition]</c> token-major, the channel concatenation the
    /// reference performs before the kernel-1 preprocess convolution.</summary>
    private void BuildTokens(Tensor tokens, Tensor latentsTokenMajor, Tensor? condition, int length)
    {
        int channels = _config.InChannels;
        int concat = _config.ConcatChannels;
        float* destination = (float*)tokens.DataPointer;
        new Span<float>(destination, length * concat).Clear();
        ReadOnlySpan<float> latentValues = latentsTokenMajor.AsReadOnlySpan<float>();
        ReadOnlySpan<float> conditionValues = condition is null ? default : condition.AsReadOnlySpan<float>();
        for (int position = 0; position < length; position++)
        {
            long row = (long)position * concat;
            latentValues.Slice(position * channels, channels).CopyTo(new Span<float>(destination + row, channels));
            if (condition is not null)
            {
                conditionValues.Slice(position * _config.ConditionDim, _config.ConditionDim)
                    .CopyTo(new Span<float>(destination + row + (2 * channels), _config.ConditionDim));
            }
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
    /// <remarks>The rotary op indexes cos/sin as <c>[batch, seq, headDim]</c>, so a batched pass needs the table
    /// repeated per batch element — and unlike the batch-1 table, a longer cached one is then unusable, because the
    /// second batch element's block starts at the ACTUAL sequence length rather than the cached one.</remarks>
    private (Tensor Cos, Tensor Sin) RopeTables(int sequenceLength, int batch)
    {
        bool reusable = _ropeCos is not null && _ropeBatch == batch
            && (batch == 1 ? _ropeLength >= sequenceLength : _ropeLength == sequenceLength);
        if (reusable)
        {
            return (_ropeCos!, _ropeSin!);
        }
        _ropeCos?.Dispose();
        _ropeSin?.Dispose();
        int headDim = _config.AttentionHeadDim;
        int rotary = _config.RotaryDim;
        int half = rotary / 2;
        Tensor cos = new Tensor(new TensorShape(batch, sequenceLength, headDim), DType.F32);
        Tensor sin = new Tensor(new TensorShape(batch, sequenceLength, headDim), DType.F32);
        float* cosData = (float*)cos.DataPointer;
        float* sinData = (float*)sin.DataPointer;
        long table = (long)sequenceLength * headDim;
        new Span<float>(cosData, (int)(table * batch)).Clear();
        new Span<float>(sinData, (int)(table * batch)).Clear();
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
        for (int element = 1; element < batch; element++)
        {
            new ReadOnlySpan<float>(cosData, (int)table).CopyTo(new Span<float>(cosData + (element * table), (int)table));
            new ReadOnlySpan<float>(sinData, (int)table).CopyTo(new Span<float>(sinData + (element * table), (int)table));
        }
        _ropeCos = cos;
        _ropeSin = sin;
        _ropeLength = sequenceLength;
        _ropeBatch = batch;
        return (cos, sin);
    }

    /// <summary>Per-block working tensors, allocated once per sequence length and reused by every block and every
    /// forward. Allocating fourteen device buffers per block, 36 blocks a forward, fragmented the CUDA pool badly
    /// enough to OOM a 12 GB card on longer generations.</summary>
    private BlockScratch Scratch(int rows, int batch)
    {
        if (_scratch is not null && _scratch.Rows == rows && _scratch.Batch == batch)
        {
            return _scratch;
        }
        _scratch?.Dispose();
        _scratch = new BlockScratch(rows, batch, _config);
        return _scratch;
    }

    private sealed class BlockScratch : IDisposable
    {
        private readonly List<Tensor> _owned = [];

        internal BlockScratch(int rows, int batch, MiniMaxMusic3DitConfig config)
        {
            Rows = rows;
            Batch = batch;
            TensorShape flat = new TensorShape(batch * rows, config.InnerDim);
            TensorShape tokenMajor = new TensorShape(batch, rows, config.NumAttentionHeads, config.AttentionHeadDim);
            TensorShape headMajor = new TensorShape(batch, config.NumAttentionHeads, rows, config.AttentionHeadDim);
            TensorShape wide = new TensorShape(batch * rows, config.FfInnerDim);
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
            Gated = Track(new TensorShape(batch * rows, 2 * config.FfInnerDim));
            States = Track(wide);
            Gate = Track(wide);
            Projected = Track(flat);
        }

        /// <summary>Tokens per batch element, NOT the row count of the flat tensors.</summary>
        internal int Rows { get; }

        internal int Batch { get; }
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
            // Per-batch-element tokens, which is what the permutes need — hidden's row count is batch times this.
            int rows = scratch.Rows;
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
