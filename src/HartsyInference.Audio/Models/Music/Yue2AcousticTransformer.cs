using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.Music;

/// <summary>YuE2's non-autoregressive acoustic stack: the flow-matching velocity field over 64-channel audio
/// latents. It has the same Qwen3 geometry as <see cref="Yue2ArLm"/> but its own weights, no embedding table and no
/// output head — a latent chunk enters through <c>vae2llm</c> and the velocity leaves through <c>llm2vae</c>.</summary>
/// <remarks><para>Two things make it unlike an ordinary decoder, and both are easy to get subtly wrong:</para>
/// <list type="bullet">
///   <item><b>Attention is bidirectional.</b> Every latent position sees every other one, plus the whole AR prefix.
///   There is no causal mask anywhere in this stack.</item>
///   <item><b>Two position schemes run at once.</b> The learned <c>latent_pos_embed</c> is indexed chunk-relative
///   from 0, while RoPE positions continue after the prefix, from <c>arLength</c>. Using one convention for both
///   produces audio that decodes to plausible-sounding noise.</item>
/// </list>
/// <para>The chunk is padded with one zero latent frame at each end before <c>vae2llm</c> and those two positions
/// are dropped from the output, so the stack runs over <c>frames + 2</c> tokens.</para></remarks>
public sealed class Yue2AcousticTransformer : IDisposable
{
    private readonly Yue2Config _config;
    private readonly Layer[] _layers;
    private Tensor? _vae2llmWeight, _vae2llmBias, _llm2vaeWeight, _llm2vaeBias;
    private Tensor? _timeMlp0Weight, _timeMlp0Bias, _timeMlp2Weight, _timeMlp2Bias;
    private Tensor? _positionEmbedding, _finalNorm;
    private int _disposed;

    private sealed class Layer
    {
        public Tensor? QWeight, KWeight, VWeight, OWeight, QNorm, KNorm;
        public Tensor? GateWeight, UpWeight, DownWeight;
        public Tensor? InputNorm, PostAttentionNorm;
    }

    public Yue2AcousticTransformer(Yue2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _layers = new Layer[config.Nar.NumHiddenLayers];
        for (int i = 0; i < _layers.Length; i++) _layers[i] = new Layer();
    }

    public int NumLayers => _config.Nar.NumHiddenLayers;

    /// <summary>Loads the converted <c>model.diffusion_model.</c> stack.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ThrowIfDisposed();
        _vae2llmWeight = weights["vae2llm.weight"];
        _vae2llmBias = weights["vae2llm.bias"];
        _llm2vaeWeight = weights["llm2vae.weight"];
        _llm2vaeBias = weights["llm2vae.bias"];
        _timeMlp0Weight = weights["time_embedder.mlp.0.weight"];
        _timeMlp0Bias = weights["time_embedder.mlp.0.bias"];
        _timeMlp2Weight = weights["time_embedder.mlp.2.weight"];
        _timeMlp2Bias = weights["time_embedder.mlp.2.bias"];
        _positionEmbedding = weights["latent_pos_embed.pe"];
        _finalNorm = weights["model.norm.weight"];
        for (int i = 0; i < _layers.Length; i++)
        {
            string p = $"model.layers.{i}";
            Layer layer = _layers[i];
            layer.QWeight = weights[$"{p}.self_attn.q_proj.weight"];
            layer.KWeight = weights[$"{p}.self_attn.k_proj.weight"];
            layer.VWeight = weights[$"{p}.self_attn.v_proj.weight"];
            layer.OWeight = weights[$"{p}.self_attn.o_proj.weight"];
            layer.QNorm = weights[$"{p}.self_attn.q_norm.weight"];
            layer.KNorm = weights[$"{p}.self_attn.k_norm.weight"];
            layer.GateWeight = weights[$"{p}.mlp.gate_proj.weight"];
            layer.UpWeight = weights[$"{p}.mlp.up_proj.weight"];
            layer.DownWeight = weights[$"{p}.mlp.down_proj.weight"];
            layer.InputNorm = weights[$"{p}.input_layernorm.weight"];
            layer.PostAttentionNorm = weights[$"{p}.post_attention_layernorm.weight"];
        }
    }

    /// <summary>Evaluates the velocity field for one chunk.</summary>
    /// <param name="state">The current ODE state, <c>[frames, 64]</c> row-major (frame-major, as the reference
    /// draws its noise).</param>
    /// <param name="rawTimestep">The solver's timestep, already <c>logit(t)</c> clamped to ±20. The warp back to
    /// <c>t</c> happens here, because the checkpoint's shift is applied on the model side.</param>
    /// <param name="arPrefix">Per-layer post-RoPE K/V from the AR prefill, each <c>[1, kv_heads, arLength, dim]</c>
    /// — the cache must have been sized to exactly <paramref name="arLength"/>, so the buffers carry no unpopulated
    /// tail.</param>
    /// <param name="velocity">Receives <c>[frames, 64]</c>.</param>
    public void Velocity(IBackend backend, ReadOnlySpan<float> state, float rawTimestep,
        (Tensor Key, Tensor Value)[] arPrefix, int arLength, Span<float> velocity)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(arPrefix);
        ThrowIfDisposed();
        if (_vae2llmWeight is null) throw new InvalidOperationException("YuE2 acoustic weights are not loaded.");
        if (arPrefix.Length != NumLayers)
            throw new ArgumentException($"Expected {NumLayers} prefix layers, got {arPrefix.Length}.", nameof(arPrefix));

        int latentDim = _config.LatentDim;
        int frames = state.Length / latentDim;
        if (frames < 1 || state.Length != frames * latentDim)
            throw new ArgumentException($"The ODE state must be a whole number of {latentDim}-channel frames.", nameof(state));
        if (velocity.Length < state.Length)
            throw new ArgumentException("The velocity buffer is smaller than the state.", nameof(velocity));

        int hidden = _config.Nar.HiddenSize;
        int heads = _config.Nar.NumAttentionHeads, kvHeads = _config.Nar.NumKeyValueHeads, dim = _config.Nar.HeadDim;
        int tokens = frames + 2;   // one zero latent frame padded at each end
        if (arLength + tokens > _config.Nar.MaxPositionEmbeddings)
            throw new ArgumentOutOfRangeException(nameof(arLength), "The acoustic chunk plus its prefix exceeds the model context.");

        // ── Input: vae2llm(padded latents) + timestep + learned chunk-relative positions ──
        using Tensor padded = ZeroTensor(new TensorShape(1, tokens, latentDim));
        unsafe
        {
            fixed (float* source = state)
                Buffer.MemoryCopy(source, (float*)padded.DataPointer + latentDim, (long)state.Length * 4, (long)state.Length * 4);
        }
        using Tensor x = new(new TensorShape(1, tokens, hidden), DType.F32);
        backend.Linear(x, padded, _vae2llmWeight!, _vae2llmBias);

        using Tensor timeEmbedding = EmbedTimestep(backend, rawTimestep);
        using Tensor positions = new(new TensorShape(1, tokens, hidden), DType.F32);
        backend.GatherRows(positions, _positionEmbedding!, ChunkPositions(tokens));

        // The timestep is identical at every position; broadcast it into the position tensor BEFORE either has
        // been through a device op, so the whole input assembly is one device-side add.
        BroadcastAddRow(positions, timeEmbedding, tokens, hidden);
        AddInPlace(backend, x, positions);

        // RoPE positions continue after the prefix — unlike the learned embedding above, which restarts at 0.
        using Tensor cos = new(new TensorShape(tokens, dim), DType.F32);
        using Tensor sin = new(new TensorShape(tokens, dim), DType.F32);
        BuildRope(cos, sin, tokens, arLength, dim, _config.Nar.RopeTheta);

        float scale = 1f / MathF.Sqrt(dim);
        float eps = _config.Nar.RmsNormEps;
        for (int i = 0; i < _layers.Length; i++)
        {
            Layer layer = _layers[i];
            using Tensor normed = new(new TensorShape(1, tokens, hidden), DType.F32);
            backend.RmsNorm(normed, x, layer.InputNorm!, eps);

            // Projections are written straight into head layout. Never reshape a tensor a device op has written:
            // the backend caches the device-side result against the tensor OBJECT, so a view sharing the host
            // pointer does not see it, and the stale pre-op host bytes are what the next op uploads.
            using Tensor q = new(new TensorShape(1, tokens, heads, dim), DType.F32);
            using Tensor k = new(new TensorShape(1, tokens, kvHeads, dim), DType.F32);
            using Tensor v = new(new TensorShape(1, tokens, kvHeads, dim), DType.F32);
            backend.Linear(q, normed, layer.QWeight!, null);
            backend.Linear(k, normed, layer.KWeight!, null);
            backend.Linear(v, normed, layer.VWeight!, null);

            // Qwen3 normalises each head independently, before RoPE. RmsNorm reduces over the input's last dim,
            // which in head layout is exactly one head's channels.
            using Tensor qNormed = new(new TensorShape(1, tokens, heads, dim), DType.F32);
            using Tensor kNormed = new(new TensorShape(1, tokens, kvHeads, dim), DType.F32);
            backend.RmsNorm(qNormed, q, layer.QNorm!, eps);
            backend.RmsNorm(kNormed, k, layer.KNorm!, eps);
            backend.ApplyRopeSingle(qNormed, cos, sin, dim);
            backend.ApplyRopeSingle(kNormed, cos, sin, dim);

            // Head-major, then prepend the AR prefix along the key axis.
            using Tensor qHeadMajor = new(new TensorShape(1, heads, tokens, dim), DType.F32);
            using Tensor kHeadMajor = new(new TensorShape(1, kvHeads, tokens, dim), DType.F32);
            using Tensor vHeadMajor = new(new TensorShape(1, kvHeads, tokens, dim), DType.F32);
            backend.Permute0213(qHeadMajor, qNormed, tokens, heads, dim);
            backend.Permute0213(kHeadMajor, kNormed, tokens, kvHeads, dim);
            backend.Permute0213(vHeadMajor, v, tokens, kvHeads, dim);

            int kvLength = arLength + tokens;
            using Tensor keys = new(new TensorShape(1, kvHeads, kvLength, dim), DType.F32);
            using Tensor values = new(new TensorShape(1, kvHeads, kvLength, dim), DType.F32);
            backend.Concat(keys, [arPrefix[i].Key, kHeadMajor], dim: 2);
            backend.Concat(values, [arPrefix[i].Value, vHeadMajor], dim: 2);

            // Every acoustic query row attends over the whole prefix, so this is prefill-shaped: thousands of
            // query rows, not the single row IBackend.FlashAttention's kernel is tuned for. Measured at 2308
            // frames on a 4090, that kernel costs 74.6 ms a layer against 2.8 ms through the general entry,
            // which reaches cuDNN's fused engine — 26x. The fused engine is MHA-only, so the grouped KV is
            // widened to full heads first; that copy is ~3% of what it buys. F16 ingest matches the release,
            // which runs the whole transformer in bfloat16. Since that fix attention is ~43% of this stack,
            // not the ~89% it was, and at 5.3 ms a call it runs near the card's BF16 peak — what is left to win
            // here is the F32 activations in the projections and the per-forward allocation churn, not the
            // attention kernel.
            using Tensor keysFull = new(new TensorShape(1, heads, kvLength, dim), DType.F32);
            using Tensor valuesFull = new(new TensorShape(1, heads, kvLength, dim), DType.F32);
            backend.RepeatKvHeads(keysFull, keys, kvHeads, heads / kvHeads);
            backend.RepeatKvHeads(valuesFull, values, kvHeads, heads / kvHeads);

            using Tensor attention = new(new TensorShape(1, heads, tokens, dim), DType.F32);
            backend.ScaledDotProductAttention(attention, qHeadMajor, keysFull, valuesFull, null, scale, allowF16: true);

            using Tensor tokenMajor = new(new TensorShape(1, tokens, heads * dim), DType.F32);
            backend.Permute0213(tokenMajor, attention, heads, tokens, dim);
            using Tensor projected = new(new TensorShape(1, tokens, hidden), DType.F32);
            backend.Linear(projected, tokenMajor, layer.OWeight!, null);
            AddInPlace(backend, x, projected);

            // The feed-forward is the widest thing here — two [hidden, 6144] projections and a [6144, hidden]
            // — and the weights are already BF16, so running its activations at F16 halves the traffic and lets
            // the GEMM stay 16-bit end to end instead of casting the activation down on every call. Measured at
            // 5900 frames: gate+up 4.00 -> 2.02 ms, down 1.85 -> 0.95, SiLU+Mul 0.79 -> 0.40. The residual stays
            // F32 — it accumulates across 28 layers, and rejoining it costs one 0.08 ms cast.
            DType act = backend.SupportsF16Activations ? DType.F16 : DType.F32;
            using Tensor mlpInput = new(new TensorShape(1, tokens, hidden), act);
            backend.RmsNorm(mlpInput, x, layer.PostAttentionNorm!, eps);
            using Tensor gate = new(new TensorShape(1, tokens, _config.Nar.IntermediateSize), act);
            using Tensor up = new(new TensorShape(1, tokens, _config.Nar.IntermediateSize), act);
            backend.Linear(gate, mlpInput, layer.GateWeight!, null);
            backend.Linear(up, mlpInput, layer.UpWeight!, null);
            backend.Silu(gate, gate);
            backend.Mul(gate, gate, up);
            using Tensor down = new(new TensorShape(1, tokens, hidden), act);
            backend.Linear(down, gate, layer.DownWeight!, null);
            if (act == DType.F32)
            {
                AddInPlace(backend, x, down);
            }
            else
            {
                using Tensor down32 = new(new TensorShape(1, tokens, hidden), DType.F32);
                backend.CastToF32(down32, down);
                AddInPlace(backend, x, down32);
            }
        }

        using Tensor final = new(new TensorShape(1, tokens, hidden), DType.F32);
        backend.RmsNorm(final, x, _finalNorm!, eps);
        using Tensor output = new(new TensorShape(1, tokens, latentDim), DType.F32);
        backend.Linear(output, final, _llm2vaeWeight!, _llm2vaeBias);

        // Drop the two padding frames.
        output.AsReadOnlySpan<float>().Slice(latentDim, frames * latentDim).CopyTo(velocity);
    }

    /// <summary>The sinusoidal timestep embedding through the two-layer MLP, <c>[1, 1, hidden]</c>.</summary>
    /// <remarks>The solver hands over <c>logit(t)</c>, so <c>t</c> is recovered with a sigmoid and then warped by
    /// the checkpoint's <c>timestep_shift</c> — which the release ships at 1.0, making the warp the identity. The
    /// frequency block is cosine-first, which is the opposite of several other DiTs in this tree.</remarks>
    private Tensor EmbedTimestep(IBackend backend, float rawTimestep)
    {
        int width = _config.TimestepEmbedWidth, half = width / 2;
        float sigmoid = 1f / (1f + MathF.Exp(-rawTimestep));
        float shift = _config.TimestepShift;
        float t = shift * sigmoid / (1f + (shift - 1f) * sigmoid);

        using Tensor frequencies = new(new TensorShape(1, 1, width), DType.F32);
        Span<float> buffer = frequencies.AsSpan<float>();
        for (int i = 0; i < half; i++)
        {
            double angle = t * Math.Exp(-Math.Log(10_000d) * i / half);
            buffer[i] = (float)Math.Cos(angle);
            buffer[half + i] = (float)Math.Sin(angle);
        }
        using Tensor firstLayer = new(new TensorShape(1, 1, _config.Nar.HiddenSize), DType.F32);
        backend.Linear(firstLayer, frequencies, _timeMlp0Weight!, _timeMlp0Bias);
        backend.Silu(firstLayer, firstLayer);
        Tensor result = new(new TensorShape(1, 1, _config.Nar.HiddenSize), DType.F32);
        backend.Linear(result, firstLayer, _timeMlp2Weight!, _timeMlp2Bias);
        return result;
    }

    /// <summary>Chunk-relative learned-embedding rows, clamped to the stored table.</summary>
    private int[] ChunkPositions(int tokens)
    {
        int[] rows = new int[tokens];
        for (int i = 0; i < tokens; i++) rows[i] = Math.Min(i, _config.MaxLatentFrames - 1);
        return rows;
    }

    /// <summary>Split-half (NeoX) RoPE tables for absolute positions <c>[posStart, posStart + tokens)</c>. Both
    /// halves of each head carry the same angle, which is what <see cref="IBackend.ApplyRopeSingle"/> expects.</summary>
    private static unsafe void BuildRope(Tensor cos, Tensor sin, int tokens, int posStart, int headDim, float theta)
    {
        int half = headDim / 2;
        float* c = (float*)cos.DataPointer;
        float* s = (float*)sin.DataPointer;
        for (int token = 0; token < tokens; token++)
        {
            long baseOffset = (long)token * headDim;
            int position = posStart + token;
            for (int i = 0; i < half; i++)
            {
                double angle = position / Math.Pow(theta, 2.0 * i / headDim);
                float cosine = (float)Math.Cos(angle), sine = (float)Math.Sin(angle);
                c[baseOffset + i] = cosine; c[baseOffset + i + half] = cosine;
                s[baseOffset + i] = sine; s[baseOffset + i + half] = sine;
            }
        }
    }

    private static void AddInPlace(IBackend backend, Tensor accumulator, Tensor addend)
        => backend.Add(accumulator, accumulator, addend);

    /// <summary>Adds one <c>[1, 1, hidden]</c> row into every token of <paramref name="target"/>, host-side.</summary>
    /// <remarks>Only safe on a tensor no device op has written yet: the backend caches a device-side result against
    /// the tensor object, and a later host write to the same buffer is invisible to it.</remarks>
    private static unsafe void BroadcastAddRow(Tensor target, Tensor row, int tokens, int hidden)
    {
        float* destination = (float*)target.DataPointer;
        float* source = (float*)row.DataPointer;
        for (int token = 0; token < tokens; token++)
        {
            float* line = destination + (long)token * hidden;
            for (int i = 0; i < hidden; i++) line[i] += source[i];
        }
    }

    private static Tensor ZeroTensor(TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32);
        tensor.AsSpan<float>().Clear();
        return tensor;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed != 0, this);

    public void Dispose() => Interlocked.Exchange(ref _disposed, 1);
}
