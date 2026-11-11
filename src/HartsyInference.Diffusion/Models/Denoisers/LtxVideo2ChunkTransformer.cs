using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Gating;
using HartsyInference.Core.Memory.Vram.Streaming;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>The rotation tables of one chunk of a chunk-causal LTX-2 video stream: the chunk's own, at its absolute frames,
/// and the sink's, at the frame the stream's state places it for this chunk. Built once per chunk and read by every pass
/// over it; disposing releases the tables' device copies.</summary>
public sealed class LtxVideo2ChunkScope : IDisposable
{
    private readonly IBackend _backend;
    private readonly Tensor[] _tables;
    private int _disposed;

    internal LtxVideo2ChunkScope(IBackend backend, int frames, int height, int width, int firstFrame, Tensor cos, Tensor sin,
        (Tensor Cos, Tensor Sin)? sink, (Tensor Cos, Tensor Sin)? sinkAlone)
    {
        _backend = backend;
        Frames = frames;
        Height = height;
        Width = width;
        FirstFrame = firstFrame;
        Cos = cos;
        Sin = sin;
        SinkCos = sink?.Cos;
        SinkSin = sink?.Sin;
        SinkCosAlone = sinkAlone?.Cos ?? SinkCos;
        SinkSinAlone = sinkAlone?.Sin ?? SinkSin;
        List<Tensor> tables = [cos, sin];
        if (sink is (Tensor, Tensor) beside)
            tables.AddRange([beside.Cos, beside.Sin]);
        if (sinkAlone is (Tensor, Tensor) alone)
            tables.AddRange([alone.Cos, alone.Sin]);
        _tables = [.. tables];
        // Host-built and read by every attention of every pass: unpinned, each read is a cache miss and a re-upload.
        backend.PreloadWeights(_tables);
    }

    /// <summary>Latent frames of the chunk.</summary>
    public int Frames { get; }

    /// <summary>Latent rows of a frame.</summary>
    public int Height { get; }

    /// <summary>Latent columns of a frame.</summary>
    public int Width { get; }

    /// <summary>The absolute latent frame of the chunk's first frame.</summary>
    public int FirstFrame { get; }

    /// <summary>Tokens of the chunk.</summary>
    public int Tokens => Frames * Height * Width;

    internal Tensor Cos { get; }

    internal Tensor Sin { get; }

    internal Tensor? SinkCos { get; }

    internal Tensor? SinkSin { get; }

    internal Tensor? SinkCosAlone { get; }

    internal Tensor? SinkSinAlone { get; }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _backend.FreeWeights(_tables);
        foreach (Tensor table in _tables)
            table.Dispose();
    }
}

/// <summary>The video stream of the LTX-2.0 transformer (<c>LTX2VideoTransformer3DModel</c> run video-only), with the
/// self-attention prefix that makes it chunk-causal: SANA-WM's streaming refiner (<c>_forward_video_only_with_rope</c>
/// over <c>_forward_video_block</c>). Single sample over packed latent tokens <c>[S, 128]</c> in (frame, row, column)
/// order. Flow: <c>proj_in</c> → the <see cref="LtxVideo2VideoBlock"/>s under one AdaLN-Single modulation for the pass
/// (every token of a chunk shares its timestep) → the AdaLN-Single output layer → <c>proj_out</c>.</summary>
/// <remarks><para>A stream runs three kinds of pass. <see cref="CaptureSink"/> runs the sink frames alone at timestep 0 and
/// stages their key and value projections. <see cref="Predict"/> attends each chunk to
/// <c>[sink ⊕ history ⊕ chunk]</c> and returns the velocity. <see cref="CaptureChunk"/> runs the finished chunk at
/// timestep 0 under the same prefix and stages its rotated keys and values for the history. Without a
/// <see cref="LtxVideo2ChunkState"/>, <see cref="Predict"/> is the plain bidirectional video-only forward.</para>
/// <para>Reads the diffusers key names of an audio-video LTX-2.0 checkpoint and only its video keys
/// (<see cref="IsVideoKey"/>), about 13.0B of the 18.9B parameters. The caption projection lives in this transformer
/// in LTX-2.0 (<see cref="ProjectCaption"/>), not in the text connectors. The blocks are streamable, so
/// <see cref="BlockStreamingScope"/> places them on a device that cannot hold all of them.</para></remarks>
public sealed class LtxVideo2ChunkTransformer : IStreamableDenoiser, IDisposable
{
    private const int ModulationRows = 6;

    private readonly LtxVideo2Config _config;
    private readonly LtxVideo2VideoBlock[] _blocks;
    private readonly LtxVideo2Rope _rope;
    private readonly LtxVideo2AdaLnSingle _timeEmbed;
    private readonly Tensor _ones;

    private Tensor? _projInW, _projInB, _projOutW, _projOutB, _scaleShift;
    private Tensor? _captionW1, _captionB1, _captionW2, _captionB2;
    private int _disposed;

    /// <summary>Creates the transformer for <paramref name="config"/>, of which the video-stream fields are read.</summary>
    public LtxVideo2ChunkTransformer(LtxVideo2Config config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _blocks = new LtxVideo2VideoBlock[config.NumLayers];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new LtxVideo2VideoBlock(config);
        _rope = LtxVideo2Rope.ForVideoSelf(config.InnerDim, config.RopeTheta, config.RopeBaseNumFrames, config.RopeBaseHeight,
            config.RopeBaseWidth, [config.VaeTemporalCompression, config.VaeSpatialCompression, config.VaeSpatialCompression],
            config.CausalOffset, config.RopeType, config.NumHeads, config.HeadDim);
        _timeEmbed = new LtxVideo2AdaLnSingle(config.InnerDim, ModulationRows);
        _ones = DiTUtils.Ones(config.InnerDim);
    }

    /// <summary>The configuration the transformer was built for.</summary>
    public LtxVideo2Config Config => _config;

    /// <summary>Whether <paramref name="key"/> of an audio-video LTX-2.0 checkpoint is read by the video-only forward. The
    /// audio stream, the audio↔video cross-attention and their modulation are not.</summary>
    public static bool IsVideoKey(string key)
    {
        ArgumentNullException.ThrowIfNull(key);
        return !key.Contains("audio", StringComparison.Ordinal) && !key.StartsWith("av_cross_attn", StringComparison.Ordinal)
            && !key.Contains("a2v_cross_attn", StringComparison.Ordinal);
    }

    /// <summary>Binds the checkpoint's tensors, which stay the caller's.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> w)
    {
        ArgumentNullException.ThrowIfNull(w);
        _projInW = w["proj_in.weight"]; w.TryGetValue("proj_in.bias", out _projInB);
        _projOutW = w["proj_out.weight"]; w.TryGetValue("proj_out.bias", out _projOutB);
        _scaleShift = TensorCasts.LoadF32(w, "scale_shift_table");
        _timeEmbed.LoadWeights(w, "time_embed");
        Tensor timeLinear = w["time_embed.linear.weight"];
        if (timeLinear.Shape[0] != (long)ModulationRows * _config.InnerDim)
            throw new InvalidOperationException($"'time_embed.linear.weight' emits {timeLinear.Shape[0]} values; an LTX-2.0 video "
                + $"stream takes {ModulationRows}·{_config.InnerDim}. Nine rows per block is the LTX-2.3 layout.");
        _captionW1 = w["caption_projection.linear_1.weight"]; w.TryGetValue("caption_projection.linear_1.bias", out _captionB1);
        _captionW2 = w["caption_projection.linear_2.weight"]; w.TryGetValue("caption_projection.linear_2.bias", out _captionB2);
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(w, $"transformer_blocks.{i}");
    }

    /// <inheritdoc />
    public int BlockCount => _blocks.Length;

    /// <inheritdoc />
    public IStreamingBlock GetBlock(int idx) => _blocks[idx];

    /// <inheritdoc />
    public Action<int>? BeforeBlockForward { get; set; }

    /// <summary>Parity hook: called with the hidden state after <c>proj_in</c> (index -1) and after each block that ran in
    /// full. Leave null in production; reading the tensor synchronizes a device backend.</summary>
    public Action<int, Tensor>? OnBlockOutput { get; set; }

    /// <inheritdoc />
    public IEnumerable<Tensor> EnumerateSharedWeights()
    {
        foreach (Tensor? t in new[] { _projInW, _projInB, _projOutW, _projOutB, _scaleShift, _captionW1, _captionB1, _captionW2, _captionB2 })
            if (t is not null) yield return t;
        foreach (Tensor t in _timeEmbed.EnumerateWeights()) yield return t;
        yield return _ones;
    }

    /// <summary>Every weight the forward reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        foreach (Tensor t in EnumerateSharedWeights()) yield return t;
        foreach (LtxVideo2VideoBlock block in _blocks)
            foreach (Tensor t in block.EnumerateWeights()) yield return t;
    }

    /// <summary>The activation dtype the block stream runs in on <paramref name="backend"/>; the stream's carried keys and
    /// values are kept in it.</summary>
    public static DType ActivationDType(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        return backend.SupportsF16Activations ? DitDtype.Act : DType.F32;
    }

    /// <summary>The caption the cross-attention reads: <c>caption_projection</c> (linear, gelu-approximate, linear) of the
    /// text connector's output <paramref name="promptEmbeds"/> <c>[L, 3840]</c>. Returns <c>[L, dim]</c> in the activation
    /// dtype; the caller owns it and computes it once per prompt.</summary>
    public Tensor ProjectCaption(IBackend backend, Tensor promptEmbeds)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(promptEmbeds);
        if (_captionW1 is null || _captionW2 is null)
            throw new InvalidOperationException("LtxVideo2ChunkTransformer.LoadWeights has not run.");
        if (promptEmbeds.Shape.Rank != 2 || promptEmbeds.Shape[1] != _captionW1.Shape[_captionW1.Shape.Rank - 1])
            throw new ArgumentException($"The caption projection takes [L, {_captionW1.Shape[_captionW1.Shape.Rank - 1]}] embeddings; "
                + $"got {promptEmbeds.Shape}.", nameof(promptEmbeds));
        int tokens = (int)promptEmbeds.Shape[0];
        DType act = ActivationDType(backend);
        using Tensor hidden = new(new TensorShape(tokens, _config.InnerDim), act);
        backend.LinearFused(hidden, promptEmbeds, _captionW1, _captionB1, LinearFusion.Gelu);
        Tensor projected = new(new TensorShape(tokens, _config.InnerDim), act);
        try
        {
            backend.Linear(projected, hidden, _captionW2, _captionB2);
            return projected;
        }
        catch
        {
            projected.Dispose();
            throw;
        }
    }

    /// <summary>Builds the rotation tables of a chunk of <paramref name="frames"/> latent frames starting at absolute frame
    /// <paramref name="firstFrame"/>, and, when <paramref name="state"/> holds a sink, the sink's tables at the frames that
    /// state places it for this chunk. Dispose the scope when the chunk's passes are done.</summary>
    /// <param name="fps">The video's frame rate, which scales the temporal coordinate.</param>
    public LtxVideo2ChunkScope OpenChunk(IBackend backend, int frames, int height, int width, double fps, int firstFrame,
        LtxVideo2ChunkState? state = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentOutOfRangeException.ThrowIfLessThan(frames, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfNegative(firstFrame);
        if (!(fps > 0))
            throw new ArgumentOutOfRangeException(nameof(fps), fps, "The frame rate must be positive.");
        if (state is not null && state.TokensPerFrame != height * width)
            throw new ArgumentException($"The stream state holds {state.TokensPerFrame} tokens per frame; the chunk has "
                + $"{height * width}.", nameof(state));
        List<Tensor> built = [];
        try
        {
            (Tensor Cos, Tensor Sin) chunk = Track(built, _rope.BuildVideo(frames, height, width, fps, firstFrame));
            (Tensor Cos, Tensor Sin)? sink = null, sinkAlone = null;
            if (state is { SinkCaptured: true })
            {
                sink = Track(built, _rope.BuildVideo(state.SinkFrames, height, width, fps, state.SinkFrameBesideHistory));
                if (state.SinkFrameWithoutHistory != state.SinkFrameBesideHistory)
                    sinkAlone = Track(built, _rope.BuildVideo(state.SinkFrames, height, width, fps, state.SinkFrameWithoutHistory));
            }

            LtxVideo2ChunkScope scope = new(backend, frames, height, width, firstFrame, chunk.Cos, chunk.Sin, sink, sinkAlone);
            built.Clear();
            return scope;
        }
        finally
        {
            foreach (Tensor table in built)
                table.Dispose();
        }
    }

    /// <summary>The velocity of <paramref name="tokens"/> <c>[S, inChannels]</c> at <paramref name="timestep"/> (sigma times
    /// <see cref="LtxVideo2Config.TimestepScaleMultiplier"/>), as host-readable F32 <c>[S, outChannels]</c> the caller owns.
    /// With <paramref name="state"/> the chunk attends to the stream's sink and history; the state is only read.</summary>
    /// <param name="encoder">The projected caption from <see cref="ProjectCaption"/>.</param>
    public Tensor Predict(IBackend backend, Tensor tokens, float timestep, Tensor encoder, LtxVideo2ChunkScope chunk,
        LtxVideo2ChunkState? state = null) =>
        Run(backend, tokens, timestep, encoder, chunk, state, LtxVideo2ChunkPassKind.Predict)
            ?? throw new InvalidOperationException("The prediction pass returned no velocity.");

    /// <summary>Runs the sink frames <paramref name="tokens"/> at timestep 0, attending to themselves only, and stages their
    /// key and value projections on every block. <see cref="LtxVideo2ChunkState.CommitSink"/> makes them current.</summary>
    public void CaptureSink(IBackend backend, Tensor tokens, Tensor encoder, LtxVideo2ChunkScope chunk, LtxVideo2ChunkState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentNullException.ThrowIfNull(chunk);
        if (state.SinkCaptured)
            throw new InvalidOperationException("The stream's sink is already captured.");
        if (chunk.Frames != state.SinkFrames || chunk.FirstFrame != 0)
            throw new ArgumentException($"The sink is frames [0, {state.SinkFrames}); the chunk scope covers "
                + $"[{chunk.FirstFrame}, {chunk.FirstFrame + chunk.Frames}).", nameof(chunk));
        Run(backend, tokens, 0f, encoder, chunk, state, LtxVideo2ChunkPassKind.CaptureSink)?.Dispose();
    }

    /// <summary>Runs the finished chunk <paramref name="tokens"/> at timestep 0 under the prefix its denoising passes read,
    /// and stages its rotated keys and values on every block that keeps the history.
    /// <see cref="LtxVideo2ChunkState.Commit"/> makes them current. Does nothing when the state keeps no history.</summary>
    public void CaptureChunk(IBackend backend, Tensor tokens, Tensor encoder, LtxVideo2ChunkScope chunk, LtxVideo2ChunkState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (state.LastHistoryLayer < 0)
            return;
        Run(backend, tokens, 0f, encoder, chunk, state, LtxVideo2ChunkPassKind.CaptureHistory)?.Dispose();
    }

    private Tensor? Run(IBackend backend, Tensor tokens, float timestep, Tensor encoder, LtxVideo2ChunkScope chunk,
        LtxVideo2ChunkState? state, LtxVideo2ChunkPassKind kind)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(tokens);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(chunk);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_projInW is null)
            throw new InvalidOperationException("LtxVideo2ChunkTransformer.LoadWeights has not run.");
        int dim = _config.InnerDim;
        if (tokens.Shape.Rank != 2 || tokens.Shape[0] != chunk.Tokens || tokens.Shape[1] != _config.InChannels)
            throw new ArgumentException($"The chunk takes [{chunk.Tokens}, {_config.InChannels}] tokens; got {tokens.Shape}.", nameof(tokens));
        if (encoder.Shape.Rank != 2 || encoder.Shape[1] != _config.CrossAttentionDim)
            throw new ArgumentException($"The caption must be [L, {_config.CrossAttentionDim}]; got {encoder.Shape}.", nameof(encoder));
        if (state is not null)
        {
            if (state.TokensPerFrame != chunk.Height * chunk.Width)
                throw new ArgumentException("The stream state and the chunk scope disagree on the frame size.", nameof(state));
            if (kind != LtxVideo2ChunkPassKind.CaptureSink && state.SinkCaptured && chunk.SinkCos is null)
                throw new ArgumentException("The chunk scope was opened without the stream state, so it has no sink tables.", nameof(chunk));
            if (kind != LtxVideo2ChunkPassKind.CaptureSink && chunk.FirstFrame != state.NextFrame)
                throw new ArgumentException($"The stream stands at frame {state.NextFrame}; the chunk scope starts at "
                    + $"{chunk.FirstFrame}.", nameof(chunk));
        }

        int last = kind == LtxVideo2ChunkPassKind.CaptureHistory ? state!.LastHistoryLayer : _blocks.Length - 1;
        Tensor? hidden = null, temb = null, embedded = null;
        try
        {
            // proj_in is where the block stream's activation dtype is chosen; every block inherits it from its input.
            hidden = new Tensor(new TensorShape(chunk.Tokens, dim), ActivationDType(backend));
            backend.Linear(hidden, tokens, _projInW, _projInB);
            (temb, embedded) = _timeEmbed.Forward(backend, timestep);
            LtxVideo2ChunkPass pass = new()
            {
                Kind = kind,
                Frames = chunk.Frames,
                Encoder = encoder,
                Temb = temb,
                Rope = _rope,
                Cos = chunk.Cos,
                Sin = chunk.Sin,
                SinkCos = chunk.SinkCos,
                SinkSin = chunk.SinkSin,
                SinkCosAlone = chunk.SinkCosAlone,
                SinkSinAlone = chunk.SinkSinAlone,
                Ones = _ones,
            };
            OnBlockOutput?.Invoke(-1, hidden);
            for (int i = 0; i <= last; i++)
            {
                BeforeBlockForward?.Invoke(i);
                // The last block of a capture pass only has to stage its keys and values: nothing reads its output.
                bool captureOnly = kind != LtxVideo2ChunkPassKind.Predict && i == last;
                Tensor input = hidden;
                hidden = null;
                hidden = _blocks[i].Forward(backend, input, pass, state?.Layer(i), captureOnly);
                if (!captureOnly)
                    OnBlockOutput?.Invoke(i, hidden);
            }

            if (kind != LtxVideo2ChunkPassKind.Predict)
                return null;
            return LtxVideo2BlockOps.OutputLayer(backend, hidden, embedded, _scaleShift!, _projOutW!, _projOutB, chunk.Tokens, dim,
                _config.OutChannels);
        }
        finally
        {
            hidden?.Dispose();
            temb?.Dispose();
            embedded?.Dispose();
        }
    }

    private static (Tensor Cos, Tensor Sin) Track(List<Tensor> built, (Tensor Cos, Tensor Sin) tables)
    {
        built.Add(tables.Cos);
        built.Add(tables.Sin);
        return tables;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _ones.Dispose();
    }
}
