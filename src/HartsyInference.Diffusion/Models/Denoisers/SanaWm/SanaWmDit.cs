using HartsyInference.Core.Backends;
using HartsyInference.Core.Backends.Layout;
using HartsyInference.Core.Backends.Norm;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The SANA-WM stage-1 DiT (<c>SanaMSVideoCamCtrl_1600M_P1_D20</c>, flow_euler_ltx path). Patch size is 1, so the
/// 1×1×1 embedders run as per-token linears over T-major tokens. The header's <c>pos_embed</c> is unused with
/// <c>wan_rope</c>, so it is not loaded. Timesteps are truncated to integers before the sinusoidal embedding.</summary>
/// <remarks>Weights are borrowed from the checkpoint in their stored dtype and preloaded on a device backend; no F32 copy of a
/// weight matrix is made. A CFG batch runs one item at a time through the whole stack, which is sequential CFG inside one call:
/// the same arithmetic as the joint batch, with the activation peak of a single item.</remarks>
public sealed class SanaWmDit : IDisposable
{
    private const float FinalNormEpsilon = 1e-6f;
    private const int FinalShiftSlot = 0, FinalScaleSlot = 1, FinalSlots = 2;

    private readonly IBackend _backend;
    private readonly SanaWmConfig _config;
    private readonly List<Tensor> _owned = [];
    private readonly SanaWmBlock[] _blocks;
    private readonly Tensor _patchWeight;
    private readonly Tensor _patchBias;
    private readonly Tensor _pluckerWeight;
    private readonly Tensor _pluckerBias;
    private readonly SanaWmTimeWeights _time;
    private readonly SanaWmCaptionWeights _caption;
    private readonly Tensor _finalWeight;
    private readonly Tensor _finalBias;
    private readonly Tensor _finalTable;
    private readonly Tensor _finalTableOnes;
    private int _disposed;

    /// <param name="backend">Backend the DiT runs on.</param>
    /// <param name="checkpoint">Converted checkpoint. Its tensors must outlive the DiT.</param>
    public SanaWmDit(IBackend backend, SanaWmCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(checkpoint);
        SanaWmConfig config = checkpoint.Config;
        IReadOnlyDictionary<string, Tensor> weights = checkpoint.Weights;
        if (!config.HasPluckerBranch)
            throw new ArgumentException("The SANA-WM DiT needs the plucker branch; the checkpoint has none.", nameof(checkpoint));

        _backend = backend;
        _config = config;
        long channels = config.HiddenSize;
        bool built = false;
        try
        {
            _patchWeight = Own(weights["x_embedder.proj.weight"].Reshape(new TensorShape([channels, config.InChannels])));
            _patchBias = Vector(weights["x_embedder.proj.bias"]);
            _pluckerWeight = Own(weights["plucker_embedder.proj.weight"].Reshape(new TensorShape([channels, config.PluckerChannels])));
            _pluckerBias = Vector(weights["plucker_embedder.proj.bias"]);
            _time = new SanaWmTimeWeights(
                weights["t_embedder.mlp.0.weight"], Vector(weights["t_embedder.mlp.0.bias"]),
                weights["t_embedder.mlp.2.weight"], Vector(weights["t_embedder.mlp.2.bias"]),
                weights["t_block.1.weight"], Vector(weights["t_block.1.bias"]));
            _caption = new SanaWmCaptionWeights(
                weights["y_embedder.y_proj.fc1.weight"], Vector(weights["y_embedder.y_proj.fc1.bias"]),
                weights["y_embedder.y_proj.fc2.weight"], Vector(weights["y_embedder.y_proj.fc2.bias"]),
                Vector(weights["attention_y_norm.weight"]));
            _finalWeight = weights["final_layer.linear.weight"];
            _finalBias = Vector(weights["final_layer.linear.bias"]);
            _finalTable = Vector(weights["final_layer.scale_shift_table"]);
            _finalTableOnes = Own(SanaWmTensors.Filled(new TensorShape(FinalSlots * channels), 1f));
            _blocks = new SanaWmBlock[config.Depth];
            for (int i = 0; i < _blocks.Length; i++)
                _blocks[i] = new SanaWmBlock(backend, config, i, weights);
            if (!backend.Device.IsCpu)
                backend.PreloadWeights(EnumerateWeights());
            built = true;
        }
        finally
        {
            if (!built)
                Dispose();
        }
    }

    /// <summary>Every tensor a forward reads from the checkpoint or the DiT's own parameters, each once.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        Tensor[] own = [_patchWeight, _patchBias, _pluckerWeight, _pluckerBias, _time.FirstWeight, _time.FirstBias, _time.SecondWeight,
            _time.SecondBias, _time.BlockWeight, _time.BlockBias, _caption.FirstWeight, _caption.FirstBias, _caption.SecondWeight,
            _caption.SecondBias, _caption.Norm, _finalWeight, _finalBias, _finalTable, _finalTableOnes];
        HashSet<Tensor> seen = new(ReferenceEqualityComparer.Instance);
        foreach (Tensor tensor in own.Concat(_blocks.SelectMany(block => block.EnumerateWeights())))
        {
            if (seen.Add(tensor))
                yield return tensor;
        }
    }

    /// <summary>Predicts the flow velocity for one CFG batch. Returns a new F32 <c>[B, OutChannels, T, H, W]</c> tensor.</summary>
    /// <param name="x">F32 latents <c>[B, InChannels, T, H, W]</c>. Batch 1 is the sequential-CFG form.</param>
    /// <param name="timesteps">One timestep per batch item and frame, <c>B·T</c> values in batch-major order; truncated to integers.</param>
    /// <param name="text">Caption features <c>[B, 1, L, TextWidth]</c>, F32, as the text path produces them.</param>
    /// <param name="mask">Caption validity <c>[B, L]</c>, F32 with 1 for valid tokens and 0 for padding; read on the host.</param>
    /// <param name="cameraConditions">Per frame, 16 camera-to-world floats then fx, fy, cx, cy: <c>B·T·20</c> floats.</param>
    /// <param name="chunkPlucker">F32 plucker features <c>[B, PluckerChannels, T, H, W]</c>.</param>
    public Tensor Forward(Tensor x, ReadOnlySpan<float> timesteps, Tensor text, Tensor mask, ReadOnlySpan<float> cameraConditions,
        Tensor chunkPlucker)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(mask);
        ArgumentNullException.ThrowIfNull(chunkPlucker);
        if (x.DType != DType.F32 || x.Shape.Rank != 5 || x.Shape[1] != _config.InChannels)
            throw new ArgumentException($"x must be F32 [B, {_config.InChannels}, T, H, W]; got {x.DType} {x.Shape}.", nameof(x));

        int batch = (int)x.Shape[0];
        int frames = (int)x.Shape[2];
        int height = (int)x.Shape[3];
        int width = (int)x.Shape[4];
        int tokens = checked(frames * height * width);
        long textTokens = text.Shape.Rank == 4 ? text.Shape[2] : -1;
        SanaWmTensors.Require(text, nameof(text), DType.F32, [batch, 1, textTokens, _config.TextWidth]);
        SanaWmTensors.Require(mask, nameof(mask), DType.F32, [batch, textTokens]);
        SanaWmTensors.Require(chunkPlucker, nameof(chunkPlucker), DType.F32,
            [batch, _config.PluckerChannels, frames, height, width]);
        if (timesteps.Length != batch * frames)
            throw new ArgumentException($"timesteps must hold {batch * frames} values (B·T); got {timesteps.Length}.", nameof(timesteps));
        int itemConditions = frames * SanaWmCamRays.ConditionWidth;
        if (cameraConditions.Length != batch * itemConditions)
            throw new ArgumentException($"camera conditions must hold {batch * itemConditions} floats.", nameof(cameraConditions));

        ReadOnlySpan<float> maskValues = mask.AsReadOnlySpan<float>();
        Grid grid = new(frames, height, width, tokens, textTokens);
        if (batch == 1)
            return ForwardItem(x, timesteps, text, maskValues, cameraConditions, chunkPlucker, grid);

        long latentElements = (long)_config.InChannels * tokens;
        long velocityElements = (long)_config.OutChannels * tokens;
        long textElements = textTokens * _config.TextWidth;
        long pluckerElements = (long)_config.PluckerChannels * tokens;
        using SanaWmScratch scratch = new();
        Tensor output = scratch.New(new TensorShape([batch, _config.OutChannels, frames, height, width]));
        for (int b = 0; b < batch; b++)
        {
            using SanaWmScratch item = new();
            Tensor itemX = item.Track(SanaWmTensors.Slice(_backend, x, b * latentElements,
                new TensorShape([1, _config.InChannels, frames, height, width])));
            Tensor itemText = item.Track(SanaWmTensors.Slice(_backend, text, b * textElements,
                new TensorShape(1, 1, textTokens, _config.TextWidth)));
            Tensor itemPlucker = item.Track(SanaWmTensors.Slice(_backend, chunkPlucker, b * pluckerElements,
                new TensorShape([1, _config.PluckerChannels, frames, height, width])));
            Tensor velocity = item.Track(ForwardItem(itemX, timesteps.Slice(b * frames, frames), itemText,
                maskValues.Slice((int)(b * textTokens), (int)textTokens), cameraConditions.Slice(b * itemConditions, itemConditions),
                itemPlucker, grid));
            SanaWmTensors.Write(_backend, output, velocity, b * velocityElements);
        }

        return scratch.Detach(output);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (_blocks is not null)
        {
            foreach (SanaWmBlock? block in _blocks)
                block?.Dispose();
        }

        SanaWmTensors.DisposeAll(_owned);
    }

    // One batch item through the embedders, every block and the final layer; returns [1, OutChannels, T, H, W].
    private Tensor ForwardItem(Tensor x, ReadOnlySpan<float> timesteps, Tensor text, ReadOnlySpan<float> mask,
        ReadOnlySpan<float> cameraConditions, Tensor chunkPlucker, Grid grid)
    {
        List<Tensor> conditioning = [];
        Tensor? hidden = null;
        try
        {
            Tensor tokenInput = SanaWmConditioning.ChannelFirstToTokens(_backend, conditioning, x, _config.InChannels, grid.Tokens);
            hidden = new Tensor(new TensorShape(1, grid.Tokens, _config.HiddenSize), DType.F32);
            _backend.Linear(hidden, tokenInput, _patchWeight, _patchBias);
            conditioning.Remove(tokenInput);
            tokenInput.Dispose();

            Tensor plucker = SanaWmConditioning.Plucker(_backend, _pluckerWeight, _pluckerBias, chunkPlucker, conditioning);
            (Tensor time, Tensor blockTime) = SanaWmConditioning.Time(_backend, _time, _config.TimestepWidth, timesteps, conditioning);
            Tensor memory = SanaWmConditioning.Caption(_backend, _caption, text, 1, grid.TextTokens, conditioning);
            Tensor memoryBias = SanaWmConditioning.MaskBias(_backend, mask, 1, grid.TextTokens, conditioning);
            using SanaWmForwardContext context = SanaWmForwardContext.Create(_backend, _config, 1, grid.Frames, grid.Height,
                grid.Width, blockTime, memory, memoryBias, plucker, cameraConditions);

            foreach (SanaWmBlock block in _blocks)
            {
                Tensor input = hidden;
                hidden = null;
                hidden = block.ForwardOwned(input, context);
            }

            using Tensor velocity = FinalLayer(hidden, time, grid);
            hidden.Dispose();
            hidden = null;
            return SanaWmConditioning.TokensToChannelFirst(_backend, velocity, 1, _config.OutChannels, grid.Frames, grid.Height,
                grid.Width);
        }
        finally
        {
            hidden?.Dispose();
            SanaWmTensors.DisposeAll(conditioning);
        }
    }

    // Final layer: LayerNorm, modulate by (table + t) per frame, then the linear to OutChannels. Returns token rows [N, out].
    private Tensor FinalLayer(Tensor tokens, Tensor time, Grid grid)
    {
        int channels = _config.HiddenSize;
        long width = (long)FinalSlots * channels;
        using SanaWmScratch scratch = new();
        Tensor timeSlots = scratch.New(grid.Frames, width);
        ReadOnlySpan<long> dims = [grid.Frames, FinalSlots, channels];
        ReadOnlySpan<long> strides = [channels, 0, 1];
        _backend.CopyStrided(timeSlots, time, StridedCopySpec.Gather(new StridedView(dims, strides, 0)));
        Tensor modulation = scratch.New(grid.Frames, width);
        _backend.AffineBroadcastLastDim(modulation, timeSlots, _finalTableOnes, _finalTable);
        scratch.Release(timeSlots);

        Tensor modulated = scratch.New(grid.Tokens, channels);
        ModulationSpec perFrame = ModulationSpec.Packed(modulation, width, (long)FinalScaleSlot * channels,
            (long)FinalShiftSlot * channels, grid.Height * grid.Width, ScaleMode.OnePlus);
        _backend.NormModulate(NormModulateCall.Rows(modulated, tokens, NormKind.LayerNorm, FinalNormEpsilon).WithModulation(perFrame));
        Tensor output = scratch.New(grid.Tokens, _config.OutChannels);
        _backend.Linear(output, modulated, _finalWeight, _finalBias);
        return scratch.Detach(output);
    }

    private Tensor Vector(Tensor source) => SanaWmTensors.HostF32(source, _owned);

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }

    private readonly record struct Grid(int Frames, int Height, int Width, int Tokens, long TextTokens);
}
