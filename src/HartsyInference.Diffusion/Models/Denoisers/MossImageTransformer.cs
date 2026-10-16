using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

namespace HartsyInference.Diffusion.Models.Denoisers;

/// <summary>MOSS-Image diffusion transformer (OpenMOSS; ComfyUI <c>comfyui/moss-image/model.py</c>, SGLang
/// <c>runtime/models/dits/moss_image.py</c>). The prompt's Qwen3.5 hidden states go through a projection and
/// <see cref="MossImageTextBlock"/>s on their own, then one <see cref="MossImageBlock"/> stack runs over
/// <c>[image ; text]</c> under a single modulation shared by every block, and the image rows leave through a modulated
/// RMSNorm head. The text side and the RoPE tables depend only on the prompt and resolution, so
/// <see cref="PrepareConditioning"/> builds them once per generation.
///
/// <para><see cref="ForwardBranches"/> runs a step's guidance branches (conditional, unconditional with its skipped
/// layers, an optional skip-layer-guidance branch) together. When the latent is the transformer's fixed buffer (see
/// <see cref="PrepareGraphLatent"/>) on a backend with step graphs, the whole branch set is captured once and replayed
/// with a single launch per step; only the timestep modulation is computed outside the graph.</para></summary>
public sealed class MossImageTransformer : IDisposable
{
    private readonly MossImageConfig _config;
    private readonly MossImageTextBlock[] _textBlocks;
    private readonly MossImageBlock[] _blocks;
    private readonly DitStepGraphSession _graph = new("MOSS-Image");
    private Tensor? _videoEmbed, _videoEmbedBias;
    private Tensor? _textProj, _textProjBias;
    private Tensor? _time1, _time1Bias, _time2, _time2Bias;
    private Tensor? _blockModulation, _blockModulationBias;
    private Tensor? _headModulation, _headModulationBias;
    private Tensor? _finalNorm, _finalLinear;
    private Tensor? _latentFixed, _blockRowsFixed, _headRowsFixed;
    private Tensor[] _velocityFixed = [];
    private bool _hasBlockQuantizedWeights;
    private int _disposed;

    /// <summary>Creates an unloaded transformer for <paramref name="config"/>.</summary>
    public MossImageTransformer(MossImageConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _textBlocks = new MossImageTextBlock[config.NumTextBlocks];
        for (int i = 0; i < _textBlocks.Length; i++)
            _textBlocks[i] = new MossImageTextBlock(config.HiddenSize, config.NumHeads, config.FfnHidden, config.Eps);
        _blocks = new MossImageBlock[config.NumLayers];
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i] = new MossImageBlock(config.HiddenSize, config.NumHeads, config.FfnHidden, config.Eps);
    }

    /// <summary>The configuration this transformer was built for.</summary>
    public MossImageConfig Config => _config;

    /// <summary>Binds the converted weights (see <c>MossImageCheckpointConverter</c>).</summary>
    public void LoadWeights(IReadOnlyDictionary<string, Tensor> weights)
    {
        ArgumentNullException.ThrowIfNull(weights);
        _videoEmbed = weights["video_embedder.weight"];
        _videoEmbedBias = TensorCasts.LoadF32(weights, "video_embedder.bias");
        _textProj = weights["text_embedder.proj.weight"];
        _textProjBias = TensorCasts.LoadF32(weights, "text_embedder.proj.bias");
        _time1 = weights["time_embedding.0.weight"];
        _time1Bias = TensorCasts.LoadF32(weights, "time_embedding.0.bias");
        _time2 = weights["time_embedding.2.weight"];
        _time2Bias = TensorCasts.LoadF32(weights, "time_embedding.2.bias");
        _blockModulation = weights["shared_block_modulation.weight"];
        _blockModulationBias = TensorCasts.LoadF32(weights, "shared_block_modulation.bias");
        _headModulation = weights["video_head_modulation.weight"];
        _headModulationBias = TensorCasts.LoadF32(weights, "video_head_modulation.bias");
        _finalNorm = TensorCasts.LoadF32(weights, "final_norm_video.weight");
        _finalLinear = weights["final_linear_video.weight"];
        for (int i = 0; i < _textBlocks.Length; i++)
            _textBlocks[i].LoadWeights(weights, $"text_embedder.blocks.{i}");
        for (int i = 0; i < _blocks.Length; i++)
            _blocks[i].LoadWeights(weights, $"layers.{i}");
        _hasBlockQuantizedWeights = EnumerateWeights().Any(static w => w.DType.IsQuantized);
    }

    /// <summary>Every weight the transformer reads.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor?[] singles =
        [
            _videoEmbed, _videoEmbedBias, _textProj, _textProjBias, _time1, _time1Bias, _time2, _time2Bias,
            _blockModulation, _blockModulationBias, _headModulation, _headModulationBias, _finalNorm, _finalLinear,
        ];
        foreach (Tensor? w in singles)
        {
            if (w is not null) yield return w;
        }
        foreach (MossImageTextBlock block in _textBlocks)
            foreach (Tensor w in block.EnumerateWeights()) yield return w;
        foreach (MossImageBlock block in _blocks)
            foreach (Tensor w in block.EnumerateWeights()) yield return w;
    }

    /// <summary>Runs the text projection and refiner over one prompt's Qwen3.5 features <c>[1, L, textDim]</c> and builds
    /// the RoPE tables for an <paramref name="hPacked"/>×<paramref name="wPacked"/> token grid.</summary>
    public unsafe MossImageConditioning PrepareConditioning(IBackend backend, Tensor textFeatures, int hPacked, int wPacked)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(textFeatures);
        ThrowIfDisposed();
        if (textFeatures.Shape.Rank != 3 || textFeatures.Shape[0] != 1 || textFeatures.Shape[2] != _config.TextDim || textFeatures.Shape[1] < 1)
            throw new ArgumentException($"MOSS-Image text features must be [1, L >= 1, {_config.TextDim}], got {textFeatures.Shape}.", nameof(textFeatures));
        if (hPacked < 1 || wPacked < 1)
            throw new ArgumentOutOfRangeException(nameof(hPacked), $"MOSS-Image needs a positive token grid, got {hPacked}x{wPacked}.");
        int length = (int)textFeatures.Shape[1];
        Tensor text = new(new TensorShape(1, length, _config.HiddenSize), DType.F32);
        (Tensor textCos, Tensor textSin) = MossImageRope.Text(length, _config.HeadDim, _config.RopeBase);
        try
        {
            backend.Linear(text, textFeatures, _textProj!, _textProjBias);
            foreach (MossImageTextBlock block in _textBlocks)
            {
                Tensor next = block.Forward(backend, text, textCos, textSin);
                text.Dispose();
                text = next;
            }
            // Materialize to host so the refined tokens outlive the activation sweeps between steps, then pin them.
            _ = text.DataPointer;
        }
        catch
        {
            text.Dispose();
            throw;
        }
        finally
        {
            textCos.Dispose();
            textSin.Dispose();
        }
        (Tensor cos, Tensor sin) = MossImageRope.Joint(hPacked, wPacked, length, _config.HeadDim, _config.RopeBase);
        return new MossImageConditioning(backend, text, cos, sin, hPacked, wPacked);
    }

    /// <summary>Predicts the flow velocity <c>[1, H·W, C]</c> for the packed latent <c>[1, H·W, C]</c> at
    /// <paramref name="timestep"/> (the 0–1000 scale), skipping <paramref name="skipLayers"/>. The caller owns the result.</summary>
    public Tensor Forward(IBackend backend, Tensor packedLatent, float timestep, MossImageConditioning conditioning,
        IReadOnlyCollection<int>? skipLayers = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ValidateLatent(packedLatent, conditioning);
        using TimeRows time = ComputeTimeRows(backend, timestep);
        using MossImageModulation modulation = MossImageModulation.Split(backend, time.Block, time.Head, _config.HiddenSize);
        return ForwardCore(backend, packedLatent, conditioning, modulation, skipLayers ?? []);
    }

    /// <summary>Runs every branch of one step at <paramref name="timestep"/>, sharing its modulation. When the latent is
    /// the fixed buffer from <see cref="PrepareGraphLatent"/> and step graphs are enabled, the branch set is captured and
    /// replayed, and the returned velocities are the transformer's fixed buffers (<c>CallerOwns</c> false: read them,
    /// never dispose them, never touch their <c>DataPointer</c>); otherwise each is a fresh caller-owned tensor.</summary>
    public (Tensor[] Velocities, bool CallerOwns) ForwardBranches(IBackend backend, Tensor packedLatent, float timestep,
        IReadOnlyList<MossImageBranch> branches)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(branches);
        if (branches.Count == 0)
            throw new ArgumentException("A MOSS-Image step needs at least one branch.", nameof(branches));
        foreach (MossImageBranch branch in branches) ValidateLatent(packedLatent, branch.Conditioning);

        using TimeRows time = ComputeTimeRows(backend, timestep);
        if (!UseGraph(backend, packedLatent))
        {
            using MossImageModulation modulation = MossImageModulation.Split(backend, time.Block, time.Head, _config.HiddenSize);
            Tensor[] eager = new Tensor[branches.Count];
            try
            {
                for (int i = 0; i < branches.Count; i++)
                    eager[i] = ForwardCore(backend, packedLatent, branches[i].Conditioning, modulation, branches[i].SkipLayers);
            }
            catch
            {
                foreach (Tensor? done in eager) done?.Dispose();
                throw;
            }
            return (eager, true);
        }

        _blockRowsFixed ??= new Tensor(time.Block.Shape, DType.F32);
        _headRowsFixed ??= new Tensor(time.Head.Shape, DType.F32);
        backend.CopyInto(_blockRowsFixed, time.Block);
        backend.CopyInto(_headRowsFixed, time.Head);
        EnsureVelocityBuffers(branches.Count, packedLatent.Shape);
        _graph.Run(backend, this, BranchSignature(branches), trimBeforeCapture: _hasBlockQuantizedWeights, step: () =>
        {
            using MossImageModulation modulation = MossImageModulation.Split(backend, _blockRowsFixed, _headRowsFixed, _config.HiddenSize);
            for (int i = 0; i < branches.Count; i++)
            {
                using Tensor velocity = ForwardCore(backend, packedLatent, branches[i].Conditioning, modulation, branches[i].SkipLayers);
                backend.CopyInto(_velocityFixed[i], velocity);
            }
        });
        return (_velocityFixed[..branches.Count], false);
    }

    /// <summary>Routes a fresh packed latent into the fixed buffer the step graph reads and the in-place sampler update
    /// writes, and returns it. The transformer owns it: never dispose it or read its <c>DataPointer</c>; take a
    /// <see cref="SnapshotGraphLatent"/> instead. A shape change drops the captured graph.</summary>
    public Tensor PrepareGraphLatent(IBackend backend, Tensor freshLatent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(freshLatent);
        ThrowIfDisposed();
        if (_latentFixed is not null && _latentFixed.Shape != freshLatent.Shape)
        {
            InvalidateStepGraph(backend);
            _latentFixed.Dispose();
            _latentFixed = null;
            DisposeVelocityBuffers();
        }
        _latentFixed ??= new Tensor(freshLatent.Shape, DType.F32);
        backend.CopyInto(_latentFixed, freshLatent);
        return _latentFixed;
    }

    /// <summary>A caller-owned device copy of the fixed latent.</summary>
    public Tensor SnapshotGraphLatent(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (_latentFixed is null)
            throw new InvalidOperationException("No graph latent has been prepared.");
        Tensor snapshot = new(_latentFixed.Shape, DType.F32);
        backend.CopyInto(snapshot, _latentFixed);
        return snapshot;
    }

    /// <summary>Drops the captured step graph. Call before the transformer's weights are freed: the graph bakes their
    /// device addresses.</summary>
    public void InvalidateStepGraph(IBackend backend) => _graph.Invalidate(backend, this);

    /// <summary>True when <see cref="ForwardBranches"/> would capture on <paramref name="backend"/> given the fixed latent:
    /// on by default on CUDA for every weight format, opt-in (<c>numerics.ditGraph</c>) elsewhere. Block-quantized weights
    /// (GGUF, nvfp4) dequantize transiently per GEMM, so their capture reserves that high-water in the graph pool; Flux.2
    /// keeps them opt-in because at 32B on 24 GB that reservation OOMs, but MOSS-Image's largest transient is one 5B-model
    /// projection, so the pool is trimmed before capture instead (and a failed capture still falls back to eager).</summary>
    public bool GraphEligible(IBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (!backend.StepGraphSupported || _graph.Dead)
            return false;
        bool defaultOn = backend.Device.IsCuda && DitStepGraph.EnabledDefaultOn;
        return defaultOn || DitStepGraph.Enabled;
    }

    private bool UseGraph(IBackend backend, Tensor packedLatent) =>
        ReferenceEquals(packedLatent, _latentFixed) && GraphEligible(backend);

    private Tensor ForwardCore(IBackend backend, Tensor packedLatent, MossImageConditioning conditioning, MossImageModulation modulation,
        IReadOnlyCollection<int> skipLayers)
    {
        int imageTokens = conditioning.Height * conditioning.Width;
        DType act = backend.SupportsF16Activations ? DitDtype.Act : DType.F32;
        Tensor? video = null;
        Tensor? x = null;
        Tensor? image = null;
        Tensor? head = null;
        try
        {
            video = new Tensor(new TensorShape(1, imageTokens, _config.HiddenSize), DType.F32);
            backend.Linear(video, packedLatent, _videoEmbed!, _videoEmbedBias);
            x = new Tensor(new TensorShape(1, imageTokens + conditioning.TextLength, _config.HiddenSize), DType.F32);
            backend.Concat(x, [video, conditioning.Text], 1);
            video.Dispose();
            video = null;
            for (int i = 0; i < _blocks.Length; i++)
            {
                if (skipLayers.Contains(i))
                    continue;
                Tensor next = _blocks[i].Forward(backend, x, modulation, conditioning.Cos, conditioning.Sin, act);
                x.Dispose();
                x = next;
            }
            image = new Tensor(new TensorShape(1, imageTokens, _config.HiddenSize), DType.F32);
            backend.SliceRows(image, x, 0);
            x.Dispose();
            x = null;
            head = new Tensor(image.Shape, DType.F32);
            backend.RmsNormModulate(head, image, _finalNorm!, modulation.ScaleHead, modulation.ShiftHead, _config.Eps);
            Tensor velocity = new(new TensorShape(1, imageTokens, _config.LatentChannels), DType.F32);
            backend.Linear(velocity, head, _finalLinear!, null);
            return velocity;
        }
        finally
        {
            video?.Dispose();
            x?.Dispose();
            image?.Dispose();
            head?.Dispose();
        }
    }

    /// <summary>The step's <c>shared_block_modulation</c> and <c>video_head_modulation</c> rows. The sinusoid is built
    /// in F64 as training did (<c>[cos, sin]</c> over <c>10000^(−i/half)</c>); the MLP runs on the backend.</summary>
    private TimeRows ComputeTimeRows(IBackend backend, float timestep)
    {
        int hidden = _config.HiddenSize;
        using Tensor sinusoid = new(new TensorShape(1, _config.FreqDim), DType.F32);
        DiTUtils.SinusoidalTimestepEmbedding(sinusoid, timestep, 1, _config.FreqDim, doublePrecision: true);
        using Tensor first = new(new TensorShape(1, hidden), DType.F32);
        backend.Linear(first, sinusoid, _time1!, _time1Bias);
        backend.Silu(first, first);
        using Tensor time = new(new TensorShape(1, hidden), DType.F32);
        backend.Linear(time, first, _time2!, _time2Bias);
        Tensor block = new(new TensorShape(1, 6 * hidden), DType.F32);
        Tensor head = new(new TensorShape(1, 2 * hidden), DType.F32);
        backend.Linear(block, time, _blockModulation!, _blockModulationBias);
        backend.Linear(head, time, _headModulation!, _headModulationBias);
        return new TimeRows(block, head);
    }

    private void ValidateLatent(Tensor packedLatent, MossImageConditioning conditioning)
    {
        ArgumentNullException.ThrowIfNull(packedLatent);
        ArgumentNullException.ThrowIfNull(conditioning);
        ThrowIfDisposed();
        int tokens = conditioning.Height * conditioning.Width;
        if (packedLatent.Shape.Rank != 3 || packedLatent.Shape[0] != 1 || packedLatent.Shape[1] != tokens
            || packedLatent.Shape[2] != _config.LatentChannels)
        {
            throw new ArgumentException(
                $"MOSS-Image expects a packed latent [1, {tokens}, {_config.LatentChannels}] for its {conditioning.Height}x{conditioning.Width} conditioning, got {packedLatent.Shape}.",
                nameof(packedLatent));
        }
    }

    private void EnsureVelocityBuffers(int count, TensorShape shape)
    {
        if (_velocityFixed.Length >= count && _velocityFixed[0].Shape == shape)
            return;
        DisposeVelocityBuffers();
        _velocityFixed = new Tensor[count];
        for (int i = 0; i < count; i++) _velocityFixed[i] = new Tensor(shape, DType.F32);
    }

    private void DisposeVelocityBuffers()
    {
        foreach (Tensor t in _velocityFixed) t.Dispose();
        _velocityFixed = [];
    }

    /// <summary>Identity of a branch set: each conditioning's identity and grid plus its skipped layers. Any change in
    /// it changes the captured op sequence.</summary>
    private static long BranchSignature(IReadOnlyList<MossImageBranch> branches)
    {
        HashCode hash = new();
        foreach (MossImageBranch branch in branches)
        {
            hash.Add(System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(branch.Conditioning));
            hash.Add(branch.Conditioning.Height);
            hash.Add(branch.Conditioning.Width);
            foreach (int layer in branch.SkipLayers.Order()) hash.Add(layer);
            hash.Add(-1);
        }
        return hash.ToHashCode();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _latentFixed?.Dispose();
        _blockRowsFixed?.Dispose();
        _headRowsFixed?.Dispose();
        DisposeVelocityBuffers();
    }

    /// <summary>A step's modulation projections, owned until disposed.</summary>
    private readonly record struct TimeRows(Tensor Block, Tensor Head) : IDisposable
    {
        public void Dispose()
        {
            Block.Dispose();
            Head.Dispose();
        }
    }
}

/// <summary>One guidance branch of a MOSS-Image step: a prompt's conditioning and the joint blocks it skips (the
/// reference skips layer 22 on the unconditional branch, and a skip-layer-guidance branch skips its own set).</summary>
public sealed record MossImageBranch(MossImageConditioning Conditioning, IReadOnlyCollection<int> SkipLayers);
