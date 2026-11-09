using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.SanaWm;

/// <summary>What every SANA-WM block of one forward reads, built once per forward instead of once per block: the latent grid,
/// the per-frame time projection <c>t_block(t)</c>, the text memory and its mask bias, the plucker tokens, the main 3D RoPE
/// tables, the camera RoPE tables and per-token ray matrices, and the token-to-frame row index of the per-frame gates.</summary>
/// <remarks>The time projection, memory, mask bias and plucker tokens are borrowed from the caller; the tables, rays and row
/// index are built here and owned.</remarks>
public sealed class SanaWmForwardContext : IDisposable
{
    private readonly List<Tensor> _owned = [];
    private SanaWmCameraInputs? _camera;
    private int _disposed;

    private SanaWmForwardContext(int batch, int frames, int height, int width, long textTokens, Tensor blockTime, Tensor memory,
        Tensor? memoryBias, Tensor? plucker)
    {
        Batch = batch;
        Frames = frames;
        Height = height;
        Width = width;
        TextTokens = textTokens;
        BlockTime = blockTime;
        Memory = memory;
        MemoryBias = memoryBias;
        Plucker = plucker;
    }

    /// <summary>Batch items.</summary>
    public int Batch { get; }

    /// <summary>Latent frames.</summary>
    public int Frames { get; }

    /// <summary>Latent height.</summary>
    public int Height { get; }

    /// <summary>Latent width.</summary>
    public int Width { get; }

    /// <summary>Tokens per frame.</summary>
    public int Spatial => Height * Width;

    /// <summary>Tokens per batch item, frame-major.</summary>
    public int Tokens => Frames * Spatial;

    /// <summary>Caption tokens per batch item.</summary>
    public long TextTokens { get; }

    /// <summary>F32 <c>[B·F, 6C]</c> output of <c>t_block</c>, one row per batch item and frame.</summary>
    public Tensor BlockTime { get; }

    /// <summary>F32 caption tokens after the projection and norm, <c>[B, L, C]</c>.</summary>
    public Tensor Memory { get; }

    /// <summary>F32 additive cross-attention bias <c>[B, 1, 1, L]</c>, or null for no padding.</summary>
    public Tensor? MemoryBias { get; }

    /// <summary>F32 projected plucker tokens <c>[B, N, C]</c>, or null without a plucker term.</summary>
    public Tensor? Plucker { get; }

    /// <summary>Main 3D RoPE cosines <c>[N, D]</c>, interleaved pairs.</summary>
    public Tensor RopeCos { get; private set; } = null!;

    /// <summary>Main 3D RoPE sines in the same layout.</summary>
    public Tensor RopeSin { get; private set; } = null!;

    /// <summary>I32 <c>[B·N]</c>: the <see cref="BlockTime"/> row (<c>b·F + f</c>) of every token, for the per-frame gates.</summary>
    public Tensor RowIndex { get; private set; } = null!;

    internal SanaWmCameraInputs? Camera => _camera;

    /// <summary>Builds the context for one forward over a <c>frames × height × width</c> latent grid.</summary>
    /// <param name="cameraConditions">Per batch item and frame, 16 camera-to-world floats then fx, fy, cx, cy in latent pixels; read
    /// only when the checkpoint has the camera branch.</param>
    public static SanaWmForwardContext Create(IBackend backend, SanaWmConfig config, int batch, int frames, int height, int width,
        Tensor blockTime, Tensor memory, Tensor? memoryBias, Tensor? plucker, ReadOnlySpan<float> cameraConditions)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(blockTime);
        ArgumentNullException.ThrowIfNull(memory);
        if (batch <= 0 || frames <= 0 || height <= 0 || width <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "Batch and latent grid extents must be positive.");
        int tokens = checked(frames * height * width);
        long textTokens = memory.Shape.Rank == 3 ? memory.Shape[1] : -1;
        int channels = config.HiddenSize;
        SanaWmTensors.Require(blockTime, nameof(blockTime), DType.F32, [(long)batch * frames, SanaWmConditioning.ModulationSlots * channels]);
        SanaWmTensors.Require(memory, nameof(memory), DType.F32, [batch, textTokens, channels]);
        if (memoryBias is not null)
            SanaWmTensors.Require(memoryBias, nameof(memoryBias), DType.F32, [batch, 1, 1, textTokens]);
        if (plucker is not null)
            SanaWmTensors.Require(plucker, nameof(plucker), DType.F32, [batch, tokens, channels]);
        if (config.HasCameraBranch && cameraConditions.Length != batch * frames * SanaWmCamRays.ConditionWidth)
            throw new ArgumentException($"Camera conditions must hold {batch * frames * SanaWmCamRays.ConditionWidth} floats.", nameof(cameraConditions));

        SanaWmForwardContext context = new(batch, frames, height, width, textTokens, blockTime, memory, memoryBias, plucker);
        bool built = false;
        List<Tensor> tables = [];
        try
        {
            (Tensor cos, Tensor sin, Tensor cameraCos, Tensor cameraSin) =
                SanaWmConditioning.Rope(backend, config.HeadDim, frames, height, width, tables);
            tables.Remove(cos);
            tables.Remove(sin);
            context.RopeCos = context.Own(cos);
            context.RopeSin = context.Own(sin);
            if (config.HasCameraBranch)
            {
                context._camera = SanaWmCameraInputs.Create(backend, cameraCos, cameraSin, config.HeadDim, cameraConditions, batch,
                    frames, height, width);
            }

            Tensor rowIndex = new(new TensorShape((long)batch * tokens), DType.I32);
            Span<int> rows = rowIndex.AsSpan<int>();
            int spatial = height * width;
            for (int i = 0; i < rows.Length; i++)
                rows[i] = i / spatial;
            context.RowIndex = context.Own(SanaWmTensors.Stage(backend, rowIndex));
            built = true;
            return context;
        }
        finally
        {
            SanaWmTensors.DisposeAll(tables);
            if (!built)
                context.Dispose();
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _camera?.Dispose();
        SanaWmTensors.DisposeAll(_owned);
    }

    private Tensor Own(Tensor tensor)
    {
        _owned.Add(tensor);
        return tensor;
    }
}
