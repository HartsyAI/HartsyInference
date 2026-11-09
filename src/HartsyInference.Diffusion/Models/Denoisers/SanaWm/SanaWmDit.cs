using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>The SANA-WM stage-1 DiT (<c>SanaMSVideoCamCtrl_1600M_P1_D20</c>, flow_euler_ltx path). Patch size is 1, so the
/// 1×1×1 embedders run as per-token linears over T-major tokens. The header's <c>pos_embed</c> is unused with
/// <c>wan_rope</c>, so it is not loaded. Timesteps are truncated to integers before the sinusoidal embedding.</summary>
public sealed class SanaWmDit : IDisposable
{
    private const float FinalNormEpsilon = 1e-6f;
    private const int FinalShiftSlot = 0, FinalScaleSlot = 1;
    private const int CameraChannels = 20;

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
            _patchWeight = Owned(weights["x_embedder.proj.weight"]).Reshape(new TensorShape([channels, config.InChannels]));
            _patchBias = Owned(weights["x_embedder.proj.bias"]);
            _pluckerWeight = Owned(weights["plucker_embedder.proj.weight"]).Reshape(new TensorShape([channels, config.PluckerChannels]));
            _pluckerBias = Owned(weights["plucker_embedder.proj.bias"]);
            _time = new SanaWmTimeWeights(
                weights["t_embedder.mlp.0.weight"], weights["t_embedder.mlp.0.bias"],
                weights["t_embedder.mlp.2.weight"], weights["t_embedder.mlp.2.bias"],
                weights["t_block.1.weight"], weights["t_block.1.bias"]);
            _caption = new SanaWmCaptionWeights(
                Owned(weights["y_embedder.y_proj.fc1.weight"]), weights["y_embedder.y_proj.fc1.bias"],
                Owned(weights["y_embedder.y_proj.fc2.weight"]), weights["y_embedder.y_proj.fc2.bias"],
                weights["attention_y_norm.weight"]);
            _finalWeight = Owned(weights["final_layer.linear.weight"]);
            _finalBias = weights["final_layer.linear.bias"];
            _finalTable = weights["final_layer.scale_shift_table"];
            _blocks = new SanaWmBlock[config.Depth];
            for (int i = 0; i < _blocks.Length; i++)
                _blocks[i] = new SanaWmBlock(backend, config, i, weights);
            built = true;
        }
        finally
        {
            if (!built)
                Dispose();
        }
    }

    /// <summary>Predicts the flow velocity for one CFG batch. Returns a new F32 <c>[B, OutChannels, T, H, W]</c> tensor.</summary>
    /// <param name="x">F32 latents <c>[B, InChannels, T, H, W]</c>.</param>
    /// <param name="timesteps">One timestep per batch item and frame, <c>B·T</c> values in batch-major order; truncated to integers.</param>
    /// <param name="text">Caption features <c>[B, 1, L, TextWidth]</c>, F32, as the text path produces them.</param>
    /// <param name="mask">Caption validity <c>[B, L]</c>, F32 with 1 for valid tokens and 0 for padding.</param>
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

        long batch = x.Shape[0];
        int frames = (int)x.Shape[2];
        int height = (int)x.Shape[3];
        int width = (int)x.Shape[4];
        int tokensPerFrame = checked(height * width);
        int tokens = checked(frames * tokensPerFrame);
        long textTokens = text.Shape.Rank == 4 ? text.Shape[2] : -1;
        SanaWmTensors.Require(text, nameof(text), DType.F32, [batch, 1, textTokens, _config.TextWidth]);
        SanaWmTensors.Require(mask, nameof(mask), DType.F32, [batch, textTokens]);
        SanaWmTensors.Require(chunkPlucker, nameof(chunkPlucker), DType.F32,
            [batch, _config.PluckerChannels, frames, height, width]);
        if (timesteps.Length != batch * frames)
            throw new ArgumentException($"timesteps must hold {batch * frames} values (B·T); got {timesteps.Length}.", nameof(timesteps));
        if (cameraConditions.Length != batch * frames * CameraChannels)
            throw new ArgumentException($"camera conditions must hold {batch * frames * CameraChannels} floats.", nameof(cameraConditions));

        List<Tensor> scratch = [];
        Tensor? produced = null;
        try
        {
            int channels = _config.HiddenSize;
            Tensor tokenInput = SanaWmConditioning.ChannelFirstToTokens(_backend, scratch, x, _config.InChannels, tokens);
            Tensor embedded = NewTensor(scratch, new TensorShape([batch * tokens, channels]));
            _backend.Linear(embedded, tokenInput, _patchWeight, _patchBias);
            Tensor plucker = SanaWmConditioning.Plucker(_backend, _pluckerWeight, _pluckerBias, chunkPlucker, scratch);

            (Tensor time, Tensor blockTime) = SanaWmConditioning.Time(_backend, _time, _config.TimestepWidth, timesteps, scratch);
            Tensor memory = SanaWmConditioning.Caption(_backend, _caption, text, batch, textTokens, scratch);
            Tensor memoryBias = SanaWmConditioning.MaskBias(_backend, mask, batch, textTokens, scratch);
            (Tensor cos, Tensor sin, Tensor cameraCos, Tensor cameraSin) =
                SanaWmConditioning.Rope(_backend, _config.HeadDim, frames, height, width, scratch);

            Tensor input = embedded.Reshape(new TensorShape([batch, tokens, channels]));
            foreach (SanaWmBlock block in _blocks)
            {
                Tensor next = block.Forward(input, memory, memoryBias, blockTime, plucker, cameraConditions, frames, height, width,
                    cos, sin, cameraCos, cameraSin);
                produced?.Dispose();
                produced = next;
                input = next;
            }

            Tensor velocity = FinalLayer(scratch, input, time, batch, frames, tokensPerFrame);
            return SanaWmConditioning.TokensToChannelFirst(_backend, velocity, batch, _config.OutChannels, frames, height, width);
        }
        finally
        {
            produced?.Dispose();
            SanaWmTensors.DisposeAll(scratch);
        }
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

    private Tensor Owned(Tensor source) => SanaWmTensors.ToF32(_backend, source, _owned);

    private Tensor NewTensor(List<Tensor> scratch, TensorShape shape)
    {
        Tensor tensor = new(shape, DType.F32, _backend.Device);
        scratch.Add(tensor);
        return tensor;
    }

    // Final layer: LayerNorm, modulate by (table + t) per frame, then the linear to OutChannels. Returns token rows [B·N, out].
    private Tensor FinalLayer(List<Tensor> scratch, Tensor tokens, Tensor time, long batch, int frames, int tokensPerFrame)
    {
        int channels = _config.HiddenSize;
        using Tensor timeHost = SanaWmTensors.Readback(_backend, time);
        using Tensor tableHost = SanaWmTensors.Readback(_backend, _finalTable);
        using Tensor modulationHost = new(new TensorShape([batch * frames * 2, channels]), DType.F32, DeviceKind.Cpu);
        ReadOnlySpan<float> timeValues = timeHost.AsReadOnlySpan<float>();
        ReadOnlySpan<float> table = tableHost.AsReadOnlySpan<float>();
        Span<float> destination = modulationHost.AsSpan<float>();
        for (long row = 0; row < batch * frames; row++)
        {
            for (int slot = 0; slot < 2; slot++)
            {
                float factor = slot == FinalScaleSlot ? 1f : 0f;
                for (int c = 0; c < channels; c++)
                    destination[(int)((row * 2 + slot) * channels) + c] = table[slot * channels + c] + timeValues[(int)row * channels + c] + factor;
            }
        }

        Tensor modulation = SanaWmTensors.Upload(_backend, modulationHost);
        scratch.Add(modulation);
        long totalTokens = batch * tokensPerFrame * frames;
        TensorShape rows = new([totalTokens, channels]);
        Tensor normed = NewTensor(scratch, rows);
        _backend.LayerNormNoAffine(normed, tokens.Reshape(rows), FinalNormEpsilon);
        Tensor modulated = NewTensor(scratch, rows);
        for (long b = 0; b < batch; b++)
        {
            for (int f = 0; f < frames; f++)
            {
                long row = b * frames * tokensPerFrame + (long)f * tokensPerFrame;
                long modulationRow = (b * frames + f) * 2;
                Tensor input = normed.SliceRows(row, tokensPerFrame);
                Tensor target = modulated.SliceRows(row, tokensPerFrame);
                Tensor shift = modulation.SliceRows(modulationRow + FinalShiftSlot, 1).Reshape(new TensorShape([channels]));
                Tensor scale = modulation.SliceRows(modulationRow + FinalScaleSlot, 1).Reshape(new TensorShape([channels]));
                _backend.AffineBroadcastLastDim(target, input, scale, shift);
            }
        }

        Tensor output = NewTensor(scratch, new TensorShape([totalTokens, _config.OutChannels]));
        _backend.Linear(output, modulated, _finalWeight, _finalBias);
        return output;
    }
}
