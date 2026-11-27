using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.FastFrames;

/// <summary>Shape of a FastFrames pixel decoder. <c>Dim</c> is the decoder's own width (1536 in both released models) and
/// <c>InputDim</c> the generator's hidden width (1536 for S; 5120 for L, which reaches the decoder through a bias-free bridge
/// Linear stored as <c>bridge.weight</c>).</summary>
public sealed record FastFramesPixelDecoderConfig(int Dim = 1536, int InputDim = 1536, int PatchT = 4, int PatchH = 16,
    int PatchW = 16, int OutChannels = 3, int LowBlocks = 2, int HighBlocks = 1, int FfnMultiplier = 4, float Eps = 1e-6f)
{
    /// <summary>Whether the generator's width differs from the decoder's, so a bridge projects into it.</summary>
    public bool HasBridge => InputDim != Dim;
}

/// <summary>The pixel decoder of FastFrames (Sec. 3.2 of arXiv 2611.23226; <c>fastframes/models/pixel_decoder.py</c> in
/// ml-fastframes a65c592), which replaces the Wan VAE decoder. It reads the generator's pre-head hidden states of a frame's
/// final denoising step and the head's time embedding, and emits <c>PatchT</c> RGB frames in <c>[-1, 1]</c> per latent frame:
/// adaLN, channel layer norm and a 1×1 mixer, then causal depthwise-ConvNeXt blocks (3×3×3 depthwise conv with <i>left-only</i>
/// temporal padding, layer norm, a 1×1 MLP with exact GELU, residual), then a linear unpatchify.
/// <para>The decoder is causal in time, so it runs frame by frame: each depthwise conv keeps the last two frames of its own
/// input in a <see cref="State"/>, and a clip decoded frame by frame equals the same clip decoded whole. The 3×3×3 depthwise
/// kernel is applied as three spatial 3×3 depthwise convolutions, one per temporal tap, summed. All compute is F32.</para></summary>
public sealed class FastFramesPixelDecoder : IDisposable
{
    private readonly FastFramesPixelDecoderConfig _config;
    private readonly List<Tensor> _owned = [];
    private Tensor? _bridge;
    private bool _loaded;
    private Tensor _modulation = null!, _entryNormW = null!, _entryNormB = null!, _entryW = null!, _entryB = null!;
    private Tensor _outW = null!, _outB = null!;
    private Block[] _blocks = [];
    private int _disposed;

    /// <summary>A decoder of the given shape; call <see cref="LoadWeights"/> before use.</summary>
    public FastFramesPixelDecoder(FastFramesPixelDecoderConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.PatchH != config.PatchW)
            throw new NotSupportedException($"The shared unpatchify needs square patches; this decoder has {config.PatchH}x{config.PatchW}.");
        _config = config;
    }

    /// <summary>The decoder's shape.</summary>
    public FastFramesPixelDecoderConfig Config => _config;

    /// <summary>Reads a decoder's shape from its checkpoint (names with or without the <c>pixel_head.</c> prefix that FastFrames-L
    /// carries): the width from <c>entry_pw</c>, the input width from <c>bridge.weight</c> when there is one, the block counts
    /// from the block names.</summary>
    public static FastFramesPixelDecoderConfig DetectConfig(IReadOnlyDictionary<string, Tensor> weights)
    {
        string prefix = weights.ContainsKey("pixel_head.modulation") ? "pixel_head." : "";
        int dim = (int)CheckpointTensors.Require(weights, "FastFrames decoder", prefix + "modulation").Shape[2];
        int inputDim = weights.TryGetValue("bridge.weight", out Tensor? bridge) ? (int)bridge.Shape[1] : dim;
        int low = 0, high = 0;
        while (weights.ContainsKey($"{prefix}conv_low.{low}.dw.weight"))
            low++;
        while (weights.ContainsKey($"{prefix}conv_high.{high}.dw.weight"))
            high++;
        int ffn = (int)(CheckpointTensors.Require(weights, "FastFrames decoder", $"{prefix}conv_low.0.pw1.weight").Shape[0] / dim);
        int patch = (int)CheckpointTensors.Require(weights, "FastFrames decoder", prefix + "fc_out.weight").Shape[0];
        // fc_out emits PatchT·PatchH·PatchW·OutChannels = 3072 for (4, 16, 16, 3); the patch is the released one.
        FastFramesPixelDecoderConfig defaults = new();
        if (patch != defaults.PatchT * defaults.PatchH * defaults.PatchW * defaults.OutChannels)
            throw new NotSupportedException($"fc_out emits {patch} values per token; only the released (4, 16, 16, 3) patch is supported.");
        return defaults with { Dim = dim, InputDim = inputDim, LowBlocks = low, HighBlocks = high, FfnMultiplier = ffn };
    }

    /// <summary>Every tensor name this decoder reads (with the <c>pixel_head.</c> prefix when <paramref name="prefixed"/>) and its
    /// checkpoint shape, so a checkpoint can be checked against the loader without loading weights.</summary>
    public static IReadOnlyDictionary<string, long[]> ExpectedShapes(FastFramesPixelDecoderConfig c, bool prefixed)
    {
        string prefix = prefixed ? "pixel_head." : "";
        int dim = c.Dim, ffn = dim * c.FfnMultiplier, outValues = c.PatchT * c.PatchH * c.PatchW * c.OutChannels;
        Dictionary<string, long[]> shapes = new()
        {
            [prefix + "modulation"] = [1, 2, dim],
            [prefix + "entry_norm.weight"] = [dim], [prefix + "entry_norm.bias"] = [dim],
            [prefix + "entry_pw.weight"] = [dim, dim, 1, 1, 1], [prefix + "entry_pw.bias"] = [dim],
            [prefix + "fc_out.weight"] = [outValues, dim], [prefix + "fc_out.bias"] = [outValues],
        };
        if (c.HasBridge)
            shapes["bridge.weight"] = [dim, c.InputDim];
        for (int i = 0; i < c.LowBlocks + c.HighBlocks; i++)
        {
            string p = prefix + (i < c.LowBlocks ? $"conv_low.{i}." : $"conv_high.{i - c.LowBlocks}.");
            shapes[p + "dw.weight"] = [dim, 1, 3, 3, 3]; shapes[p + "dw.bias"] = [dim];
            shapes[p + "norm.weight"] = [dim]; shapes[p + "norm.bias"] = [dim];
            shapes[p + "pw1.weight"] = [ffn, dim, 1, 1, 1]; shapes[p + "pw1.bias"] = [ffn];
            shapes[p + "pw2.weight"] = [dim, ffn, 1, 1, 1]; shapes[p + "pw2.bias"] = [dim];
        }
        return shapes;
    }

    /// <summary>Loads the weights. Names may carry the <c>pixel_head.</c> prefix; the tensors in <paramref name="weights"/> are
    /// only read.</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_loaded)
            throw new InvalidOperationException("The decoder's weights are already loaded; a decoder loads once.");
        bool prefixed = weights.ContainsKey("pixel_head.modulation");
        foreach ((string name, long[] shape) in ExpectedShapes(_config, prefixed))
        {
            // Checked before anything is copied, so a checkpoint that fails here can be fixed and loaded again.
            CheckpointTensors.Require(weights, "FastFrames decoder", name, shape);
        }
        string prefix = weights.ContainsKey("pixel_head.modulation") ? "pixel_head." : "";
        int dim = _config.Dim;
        if (_config.HasBridge)
            _bridge = Own(F32(weights, "bridge.weight"));
        _modulation = Own(F32(weights, prefix + "modulation"));
        _entryNormW = Own(F32(weights, prefix + "entry_norm.weight")); _entryNormB = Own(F32(weights, prefix + "entry_norm.bias"));
        _entryW = Own(Matrix(weights, prefix + "entry_pw.weight", dim, dim)); _entryB = Own(F32(weights, prefix + "entry_pw.bias"));
        // The head's rows are reordered once here, so the shared unpatchify reads each token's channel axis with the frame index
        // inside the channel (see Unpatchify).
        using (Tensor rawW = F32(weights, prefix + "fc_out.weight"))
            _outW = Own(ReorderOutputRows(rawW, _config));
        using (Tensor rawB = F32(weights, prefix + "fc_out.bias"))
            _outB = Own(ReorderOutputRows(rawB, _config));
        List<Block> blocks = [];
        for (int i = 0; i < _config.LowBlocks; i++)
            blocks.Add(LoadBlock(weights, $"{prefix}conv_low.{i}."));
        for (int i = 0; i < _config.HighBlocks; i++)
            blocks.Add(LoadBlock(weights, $"{prefix}conv_high.{i}."));
        _blocks = [.. blocks];
        _loaded = true;
    }

    /// <summary>A new decoding state for one clip: no frames seen, so the first frame's temporal padding is zeros.</summary>
    public State CreateState()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new(_blocks.Length);
    }

    /// <summary>Decodes one latent frame. <paramref name="hidden"/> is the generator's pre-head hidden states
    /// <c>[gridH·gridW, InputDim]</c>, <paramref name="temb"/> its head time embedding <c>[1, InputDim]</c>. Returns
    /// <c>[1, PatchT·OutChannels, gridH·PatchH, gridW·PatchW]</c> F32 in about <c>[-1, 1]</c> (not clamped); the caller disposes it. Its
    /// memory is the frame-major <c>[PatchT, OutChannels, H, W]</c>, so the frame count is <c>PatchT</c>, not <c>Shape[0]</c>.
    /// Frames must be passed in order on the same <paramref name="state"/>.</summary>
    public Tensor DecodeFrame(IBackend backend, Tensor hidden, Tensor temb, int gridH, int gridW, State state)
    {
        ArgumentNullException.ThrowIfNull(state);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ObjectDisposedException.ThrowIf(state.IsDisposed, state);
        RequireLoaded();
        int dim = _config.Dim, tokens = gridH * gridW;
        if (hidden.Shape.Rank != 2 || hidden.Shape[0] != tokens || hidden.Shape[1] != _config.InputDim)
            throw new ArgumentException($"hidden is {hidden.Shape}; expected [{tokens}, {_config.InputDim}].", nameof(hidden));
        if (state.Blocks != _blocks.Length)
            throw new ArgumentException("The state belongs to a different decoder.", nameof(state));
        if (temb.Shape.Rank != 2 || temb.Shape[0] != 1 || temb.Shape[1] != _config.InputDim)
            throw new ArgumentException($"temb is {temb.Shape}; expected [1, {_config.InputDim}].", nameof(temb));
        if (hidden.DType != DType.F32 || temb.DType != DType.F32)
            throw new ArgumentException($"hidden and temb must be F32 (got {hidden.DType} and {temb.DType}); cast the features first.", nameof(hidden));
        if (state.Faulted)
            throw new InvalidOperationException("This state faulted on an earlier frame; create a new state for the next clip.");
        if (state.Grid is { } grid && grid != (gridH, gridW))
            throw new ArgumentException($"This state decodes {grid.H}x{grid.W} token grids; this frame is {gridH}x{gridW}.", nameof(state));
        state.Grid = (gridH, gridW);

        Tensor x = ModulatedInput(backend, hidden, temb);
        Tensor entryNormed = FastFramesOps.LayerNormAffine(backend, x, _entryNormW, _entryNormB, _config.Eps);
        x.Dispose();
        x = FastFramesOps.Linear(backend, entryNormed, _entryW, _entryB);
        entryNormed.Dispose();

        Tensor current = x;
        try
        {
            for (int b = 0; b < _blocks.Length; b++)
            {
                Tensor next = RunBlock(backend, _blocks[b], current, gridH, gridW, state, b);
                current.Dispose();
                current = next;
            }
            using Tensor tokensOut = FastFramesOps.Linear(backend, current, _outW, _outB);
            current.Dispose();
            return Unpatchify(backend, tokensOut, gridH, gridW);
        }
        catch
        {
            // The blocks before the failure have already slid their windows: the state no longer matches a clip.
            state.Faulted = true;
            current.Dispose();
            throw;
        }
    }

    /// <summary>The adaLN of the decoder's entry: <c>norm(x) · (1 + scale) + shift</c> with <c>(shift, scale) = chunk(modulation +
    /// temb)</c>, after the bridge when the generator's width differs.</summary>
    private Tensor ModulatedInput(IBackend backend, Tensor hidden, Tensor temb)
    {
        int dim = _config.Dim;
        Tensor? bridgedHidden = null, bridgedTemb = null;
        try
        {
            if (_bridge is not null)
            {
                bridgedHidden = FastFramesOps.Linear(backend, hidden, _bridge, null);
                bridgedTemb = FastFramesOps.Linear(backend, temb, _bridge, null);
            }
            ReadOnlySpan<float> e = (bridgedTemb ?? temb).AsReadOnlySpan<float>();
            ReadOnlySpan<float> m = _modulation.AsReadOnlySpan<float>();   // [1, 2, dim]: shift row, then scale row
            float[] shift = new float[dim], scale = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                shift[i] = m[i] + e[i];
                scale[i] = m[dim + i] + e[i];
            }
            using Tensor shiftT = FastFramesOps.Vector(shift);
            using Tensor scaleT = FastFramesOps.OnePlus(scale);
            return FastFramesOps.Modulate(backend, bridgedHidden ?? hidden, shiftT, scaleT, _config.Eps);
        }
        finally
        {
            bridgedHidden?.Dispose();
            bridgedTemb?.Dispose();
        }
    }

    private Tensor RunBlock(IBackend backend, Block block, Tensor x, int gridH, int gridW, State state, int index)
    {
        int dim = _config.Dim, tokens = gridH * gridW;
        // Every backend input below is the tensor the previous op produced, never a Reshape view of it. A device result is
        // bound to the tensor it was written into, so a view reads the host copy, which is never written on a GPU. Each
        // layout change is therefore a transpose straight into an owned tensor of the shape the next op takes.
        // Channel-first [1, dim, H, W] for the spatial depthwise convs.
        Tensor current = new(new TensorShape(1, dim, gridH, gridW), DType.F32);
        backend.Transpose2D(current, x, tokens, dim);
        Tensor conv = new(new TensorShape(1, dim, gridH, gridW), DType.F32);
        backend.Conv2dDepthwise(conv, current, block.Taps[2], block.DwBias, 1, 1, 1, 1);
        Tensor? previous1 = state.Previous1[index], previous2 = state.Previous2[index];
        conv = AddTap(backend, conv, previous1, block.Taps[1], gridH, gridW, dim);
        conv = AddTap(backend, conv, previous2, block.Taps[0], gridH, gridW, dim);
        // Slide the window: this frame's input becomes t-1, the old t-1 becomes t-2, the old t-2 is dropped.
        previous2?.Dispose();
        state.Previous2[index] = previous1;
        state.Previous1[index] = current;

        using Tensor tokenMajor = new(new TensorShape(tokens, dim), DType.F32);
        backend.Transpose2D(tokenMajor, conv, dim, tokens);
        conv.Dispose();
        using Tensor normed = FastFramesOps.LayerNormAffine(backend, tokenMajor, block.NormW, block.NormB, _config.Eps);
        using Tensor up = FastFramesOps.Linear(backend, normed, block.Pw1W, block.Pw1B);
        using Tensor act = FastFramesOps.GeluErf(backend, up);
        using Tensor down = FastFramesOps.Linear(backend, act, block.Pw2W, block.Pw2B);
        return FastFramesOps.Add(backend, x, down);
    }

    /// <summary>Adds one earlier frame's depthwise conv (no bias) to <paramref name="sum"/>; a frame before the clip is zeros and
    /// contributes nothing.</summary>
    private static Tensor AddTap(IBackend backend, Tensor sum, Tensor? earlierInput, Tensor tap, int gridH, int gridW, int dim)
    {
        if (earlierInput is null)
            return sum;
        // The earlier frame is passed as stored, not as a view (see RunBlock).
        using Tensor tapOutput = new(new TensorShape(1, dim, gridH, gridW), DType.F32);
        backend.Conv2dDepthwise(tapOutput, earlierInput, tap, null, 1, 1, 1, 1);
        Tensor added = new(sum.Shape, DType.F32);
        backend.Add(added, sum, tapOutput);
        sum.Dispose();
        return added;
    }

    /// <summary>Linear unpatchify on the device, through the shared <see cref="PixelShuffles.UnpatchifyTokens"/>. Token <c>(h, w)</c>
    /// holds, in the order the reordered head wrote it, <c>((q·PatchW + r)·PatchT + p)·C + c</c>: frame <c>p</c>, channel <c>c</c> at
    /// patch row <c>q</c>, column <c>r</c>. The shared op reads the channel axis as <c>(p, c)</c> fastest, so the result is
    /// <c>[1, PatchT·C, H, W]</c>, which is the frame-major <c>[PatchT, C, H, W]</c> in memory.</summary>
    private Tensor Unpatchify(IBackend backend, Tensor tokensOut, int gridH, int gridW)
    {
        int pt = _config.PatchT, ph = _config.PatchH, pw = _config.PatchW, c = _config.OutChannels;
        Tensor output = new(new TensorShape(1, pt * c, gridH * ph, gridW * pw), DType.F32);
        PixelShuffles.UnpatchifyTokens(backend, output, tokensOut, pt * c, gridH, gridW, ph, innerChannelFastest: true);
        return output;
    }

    /// <summary>Reorders the head's rows (or bias entries) from the released <c>((p·PatchH + q)·PatchW + r)·C + c</c> to
    /// <c>((q·PatchW + r)·PatchT + p)·C + c</c>, so the frame index sits inside the channel axis.</summary>
    private static Tensor ReorderOutputRows(Tensor source, FastFramesPixelDecoderConfig config)
    {
        int pt = config.PatchT, ph = config.PatchH, pw = config.PatchW, c = config.OutChannels;
        int rowLength = (int)(source.Shape.ElementCount / (pt * ph * pw * c));
        Tensor output = new(source.Shape, DType.F32);
        ReadOnlySpan<float> from = source.AsReadOnlySpan<float>();
        Span<float> to = output.AsSpan<float>();
        for (int q = 0; q < ph; q++)
        {
            for (int r = 0; r < pw; r++)
            {
                for (int p = 0; p < pt; p++)
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        int released = ((p * ph + q) * pw + r) * c + ch;
                        int reordered = ((q * pw + r) * pt + p) * c + ch;
                        from.Slice(released * rowLength, rowLength).CopyTo(to.Slice(reordered * rowLength, rowLength));
                    }
                }
            }
        }
        return output;
    }

    private Block LoadBlock(IReadOnlyDictionary<string, Tensor> weights, string p)
    {
        int dim = _config.Dim, ffn = dim * _config.FfnMultiplier;
        using Tensor dw = F32(weights, p + "dw.weight");   // [C, 1, 3, 3, 3]
        Tensor[] taps = new Tensor[3];
        ReadOnlySpan<float> kernel = dw.AsReadOnlySpan<float>();
        for (int t = 0; t < 3; t++)
        {
            taps[t] = Own(new Tensor(new TensorShape(dim, 1, 3, 3), DType.F32));
            Span<float> tap = taps[t].AsSpan<float>();
            for (int c = 0; c < dim; c++)
                kernel.Slice((c * 3 + t) * 9, 9).CopyTo(tap.Slice(c * 9, 9));
        }
        return new Block(taps, Own(F32(weights, p + "dw.bias")), Own(F32(weights, p + "norm.weight")), Own(F32(weights, p + "norm.bias")),
            Own(Matrix(weights, p + "pw1.weight", ffn, dim)), Own(F32(weights, p + "pw1.bias")),
            Own(Matrix(weights, p + "pw2.weight", dim, ffn)), Own(F32(weights, p + "pw2.bias")));
    }

    /// <summary>A 1×1 Conv3d weight <c>[out, in, 1, 1, 1]</c> as the <c>[out, in]</c> Linear weight (the same bytes).</summary>
    private static Tensor Matrix(IReadOnlyDictionary<string, Tensor> weights, string key, int outDim, int inDim) =>
        FastFramesOps.CopyF32(CheckpointTensors.Require(weights, "FastFrames decoder", key), new TensorShape(outDim, inDim));

    /// <summary>An owned F32 copy of a checkpoint tensor, read through <see cref="CheckpointTensors"/> so a missing key names the family.
    /// Shapes were checked in <see cref="LoadWeights"/> before any copy.</summary>
    private static Tensor F32(IReadOnlyDictionary<string, Tensor> weights, string key)
    {
        Tensor source = CheckpointTensors.Require(weights, "FastFrames decoder", key);
        return FastFramesOps.CopyF32(source, source.Shape);
    }

    /// <summary>The decoder's owned device-uploadable weights, for a caller that places them before a clip.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        RequireLoaded();
        return _owned.AsReadOnly();
    }

    private void RequireLoaded()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (!_loaded)
            throw new InvalidOperationException("The decoder has no weights yet; call LoadWeights first.");
    }

    private Tensor Own(Tensor t)
    {
        _owned.Add(t);
        return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        foreach (Tensor t in _owned)
            t.Dispose();
        _owned.Clear();
    }

    /// <summary>The last two frames of every depthwise conv's input, channel-first; a null entry is a frame before the clip
    /// (zeros). One state decodes one clip, in order.</summary>
    public sealed class State(int blocks) : IDisposable
    {
        /// <summary>Blocks the state covers.</summary>
        public int Blocks { get; } = blocks;

        /// <summary>Whether this state has been disposed.</summary>
        public bool IsDisposed { get; private set; }

        /// <summary>Set when a frame failed partway: the blocks before the failure have already advanced, so the state no longer
        /// matches a clip and refuses further frames.</summary>
        internal bool Faulted { get; set; }

        /// <summary>The token grid of the frames decoded so far, height and width; null before the first frame.</summary>
        internal (int H, int W)? Grid { get; set; }

        internal Tensor?[] Previous1 { get; } = new Tensor?[blocks];

        internal Tensor?[] Previous2 { get; } = new Tensor?[blocks];

        /// <inheritdoc/>
        public void Dispose()
        {
            if (IsDisposed)
                return;
            IsDisposed = true;
            for (int i = 0; i < Blocks; i++)
            {
                Previous1[i]?.Dispose();
                Previous2[i]?.Dispose();
                Previous1[i] = Previous2[i] = null;
            }
        }
    }

    private sealed record Block(Tensor[] Taps, Tensor DwBias, Tensor NormW, Tensor NormB, Tensor Pw1W, Tensor Pw1B, Tensor Pw2W, Tensor Pw2B);
}
