using System.Runtime.ExceptionServices;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Patch embedding for SD3 / SD3.5 MMDiT: converts a latent <c>[B,C,H,W]</c> into <c>[B,numPatches,embedDim]</c> via a strided projection plus center-cropped learned 2D positions.</summary>
public sealed class PatchEmbed : IDisposable
{
    private readonly int _patchSize;
    private readonly int _inChannels;
    private readonly int _embedDim;
    private readonly object _stateLock = new();

    private Tensor? _projWeight;
    private Tensor? _projBias;
    private Tensor? _posEmbed;
    private Tensor? _posEmbedSource;
    private Tensor? _ownedPosEmbed;
    private int _posEmbedMaxSize;

    // The denoising loop repeatedly uses one resolution. Retaining its small host index table avoids rebuilding
    // it every step; GatherRows still performs the bounded synchronous upload required by its backend contract.
    private int _cachedIndexBatch;
    private int _cachedIndexGridH;
    private int _cachedIndexGridW;
    private int[]? _cachedPositionRows;
    private bool _disposed;

    /// <summary>Creates a patch embedding layer.</summary>
    public PatchEmbed(int patchSize, int inChannels, int embedDim)
    {
        if (patchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(patchSize), patchSize, "Patch size must be positive.");
        if (inChannels <= 0)
            throw new ArgumentOutOfRangeException(nameof(inChannels), inChannels, "Input channel count must be positive.");
        if (embedDim <= 0)
            throw new ArgumentOutOfRangeException(nameof(embedDim), embedDim, "Embedding dimension must be positive.");

        _patchSize = patchSize;
        _inChannels = inChannels;
        _embedDim = embedDim;
    }

    /// <summary>Loads borrowed projection tensors and an optional positional embedding shaped <c>[1,maxGrid*maxGrid,embedDim]</c>; a non-F32 positional tensor is converted once and owned by this layer, other caller-provided tensors remain caller-owned.</summary>
    public void LoadWeights(Tensor projWeight, Tensor projBias, Tensor? posEmbed = null)
    {
        ArgumentNullException.ThrowIfNull(projWeight);
        ArgumentNullException.ThrowIfNull(projBias);

        lock (_stateLock)
        {
            ThrowIfDisposed();
        }

        ValidateProjectionWeight(projWeight);
        ValidateProjectionBias(projBias);
        int posEmbedMaxSize = ValidatePositionEmbedding(posEmbed);

        Tensor? supersededOwned = null;
        lock (_stateLock)
        {
            ThrowIfDisposed();

            Tensor? replacementPos;
            Tensor? replacementOwned;
            if (posEmbed is null)
            {
                replacementPos = null;
                replacementOwned = null;
            }
            else if ((ReferenceEquals(posEmbed, _posEmbedSource) || ReferenceEquals(posEmbed, _ownedPosEmbed)) &&
                     _posEmbed is not null)
            {
                // Idempotent reload of the same loader tensor (or of our preloading snapshot) must neither
                // allocate another potentially huge cast nor accidentally dispose the active one as "superseded".
                replacementPos = _posEmbed;
                replacementOwned = _ownedPosEmbed;
            }
            else if (posEmbed.DType == DType.F32)
            {
                replacementPos = posEmbed;
                replacementOwned = null;
            }
            else
            {
                replacementOwned = posEmbed.CastTo(DType.F32);
                replacementPos = replacementOwned;
            }

            if (!ReferenceEquals(_ownedPosEmbed, replacementOwned))
                supersededOwned = _ownedPosEmbed;

            _projWeight = projWeight;
            _projBias = projBias;
            _posEmbed = replacementPos;
            if (posEmbed is null)
                _posEmbedSource = null;
            else if (!ReferenceEquals(posEmbed, replacementOwned))
                _posEmbedSource = posEmbed;
            _ownedPosEmbed = replacementOwned;
            _posEmbedMaxSize = posEmbedMaxSize;
            InvalidatePositionIndexCache();
        }

        // The replacement is already published, so even a backend cleanup failure cannot leave the layer
        // pointing at the disposed tensor. Tensor.Dispose is idempotent and releases every registered backend.
        supersededOwned?.Dispose();
    }

    /// <summary>Returns a stable snapshot of all tensors that a backend should preload.</summary>
    public IEnumerable<Tensor> EnumerateWeights()
    {
        lock (_stateLock)
        {
            ThrowIfDisposed();
            if (_projWeight is null || _projBias is null)
                return Array.Empty<Tensor>();
            return _posEmbed is null
                ? new[] { _projWeight, _projBias }
                : new[] { _projWeight, _projBias, _posEmbed };
        }
    }

    /// <summary>Patch-embeds a latent entirely through backend operations: the convolution result is physically <c>[B,C,gridH,gridW]</c>, so treating its contiguous spatial plane as <c>P=gridH*gridW</c> lets <see cref="IBackend.Permute0213"/> produce <c>[B,P,C]</c> without a host readback.</summary>
    public Tensor Forward(IBackend backend, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(input);

        lock (_stateLock)
        {
            ThrowIfDisposed();
            Tensor projWeight = _projWeight
                ?? throw new InvalidOperationException("PatchEmbed weights have not been loaded.");
            Tensor projBias = _projBias
                ?? throw new InvalidOperationException("PatchEmbed weights have not been loaded.");

            (int batch, int gridH, int gridW, int numPatches) = ValidateForwardInput(input);
            Tensor? posEmbed = _posEmbed;
            if (posEmbed is not null && (gridH > _posEmbedMaxSize || gridW > _posEmbedMaxSize))
            {
                throw new ArgumentException(
                    $"Patch grid {gridH}x{gridW} exceeds the stored positional grid " +
                    $"{_posEmbedMaxSize}x{_posEmbedMaxSize}.", nameof(input));
            }

            Tensor? convOut = null;
            Tensor? tokens = null;
            Tensor? positionRows = null;
            Tensor? result = null;
            Tensor? returned = null;
            Exception? operationError = null;

            try
            {
                convOut = new Tensor(new TensorShape(batch, _embedDim, gridH, gridW), DType.F32);
                backend.Conv2D(convOut, input, projWeight, projBias, _patchSize, _patchSize, 0, 0);

                TensorShape sequenceShape = new(batch, numPatches, _embedDim);
                tokens = new Tensor(sequenceShape, DType.F32);
                backend.Permute0213(tokens, convOut, _embedDim, numPatches, 1);

                // Queue the device free immediately after its only consumer to keep the live activation set small.
                convOut.Dispose();
                convOut = null;

                if (posEmbed is null)
                {
                    returned = tokens;
                    tokens = null;
                    return returned;
                }

                int[] rowIndices = GetPositionRowIndices(batch, gridH, gridW, numPatches);
                positionRows = new Tensor(sequenceShape, DType.F32);
                backend.GatherRows(positionRows, posEmbed, rowIndices);

                result = new Tensor(sequenceShape, DType.F32);
                backend.Add(result, tokens, positionRows);
                returned = result;
                result = null;
                return returned;
            }
            catch (Exception error)
            {
                operationError = error;
                throw;
            }
            finally
            {
                Exception? cleanupError = null;
                cleanupError = TryDispose(result, cleanupError);
                cleanupError = TryDispose(positionRows, cleanupError);
                cleanupError = TryDispose(tokens, cleanupError);
                cleanupError = TryDispose(convOut, cleanupError);

                if (operationError is null && cleanupError is not null)
                {
                    // A return value cannot be handed out after one of its input lifetimes failed to close cleanly.
                    cleanupError = TryDispose(returned, cleanupError);
                    ExceptionDispatchInfo.Capture(cleanupError!).Throw();
                }
            }
        }
    }

    /// <summary>Returns the exact patch grid for a positive, patch-aligned spatial resolution.</summary>
    public (int gridH, int gridW) GetGridSize(int height, int width)
    {
        lock (_stateLock)
        {
            ThrowIfDisposed();
            ValidateSpatialDimension(height, nameof(height));
            ValidateSpatialDimension(width, nameof(width));
            int gridH = height / _patchSize;
            int gridW = width / _patchSize;
            if (_posEmbed is not null && gridH > _posEmbedMaxSize)
            {
                throw new ArgumentOutOfRangeException(nameof(height),
                    $"Patch-grid height {gridH} exceeds the stored positional-grid height {_posEmbedMaxSize}.");
            }
            if (_posEmbed is not null && gridW > _posEmbedMaxSize)
            {
                throw new ArgumentOutOfRangeException(nameof(width),
                    $"Patch-grid width {gridW} exceeds the stored positional-grid width {_posEmbedMaxSize}.");
            }
            return (gridH, gridW);
        }
    }

    /// <summary>Releases only positional storage derived by this layer; loader tensors remain caller-owned.</summary>
    public void Dispose()
    {
        Tensor? ownedPosEmbed;
        lock (_stateLock)
        {
            if (_disposed)
                return;

            _disposed = true;
            ownedPosEmbed = _ownedPosEmbed;
            _ownedPosEmbed = null;
            _posEmbed = null;
            _posEmbedSource = null;
            _projWeight = null;
            _projBias = null;
            InvalidatePositionIndexCache();
        }

        try
        {
            ownedPosEmbed?.Dispose();
        }
        finally
        {
            GC.SuppressFinalize(this);
        }
    }

    private (int batch, int gridH, int gridW, int numPatches) ValidateForwardInput(Tensor input)
    {
        if (input.DType != DType.F32)
            throw new NotSupportedException($"PatchEmbed input must be F32; got {input.DType}.");
        if (input.Shape.Rank != 4)
            throw new ArgumentException($"PatchEmbed input must have shape [B,C,H,W]; got {input.Shape}.", nameof(input));

        long batch = input.Shape[0];
        long channels = input.Shape[1];
        long height = input.Shape[2];
        long width = input.Shape[3];
        if (batch <= 0 || batch > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(input), $"Batch dimension must be in [1,{int.MaxValue}]; got {batch}.");
        if (channels != _inChannels)
            throw new ArgumentException($"PatchEmbed expected {_inChannels} input channels; got {channels}.", nameof(input));
        if (height <= 0 || height > int.MaxValue || width <= 0 || width > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(input), $"Spatial dimensions must be positive 32-bit values; got {height}x{width}.");
        if (height % _patchSize != 0 || width % _patchSize != 0)
            throw new ArgumentException(
                $"Input spatial dimensions {height}x{width} must be divisible by patch size {_patchSize}.", nameof(input));

        long gridH = height / _patchSize;
        long gridW = width / _patchSize;
        long numPatches;
        long tokenElements;
        long inputElements;
        try
        {
            numPatches = checked(gridH * gridW);
            tokenElements = checked(checked(batch * numPatches) * _embedDim);
            inputElements = checked(checked(checked(batch * channels) * height) * width);
        }
        catch (OverflowException error)
        {
            throw new ArgumentException("PatchEmbed dimensions overflow supported indexing.", nameof(input), error);
        }

        if (numPatches > int.MaxValue || tokenElements > int.MaxValue || inputElements > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(nameof(input),
                $"PatchEmbed requires 32-bit kernel indexing; patches={numPatches}, tokens={tokenElements}, input={inputElements}.");
        }
        if (input.ElementCount != inputElements)
            throw new ArgumentException("PatchEmbed input shape has an invalid element count.", nameof(input));

        return ((int)batch, (int)gridH, (int)gridW, (int)numPatches);
    }

    private void ValidateProjectionWeight(Tensor weight)
    {
        if (weight.Shape.Rank != 4 ||
            weight.Shape[0] != _embedDim ||
            weight.Shape[1] != _inChannels ||
            weight.Shape[2] != _patchSize ||
            weight.Shape[3] != _patchSize)
        {
            throw new ArgumentException(
                $"Projection weight must have shape [{_embedDim},{_inChannels},{_patchSize},{_patchSize}]; got {weight.Shape}.",
                nameof(weight));
        }
        if (!IsSupportedProjectionWeightDType(weight.DType))
            throw new NotSupportedException($"PatchEmbed projection weight dtype {weight.DType} is not supported.");
        if (weight.ElementCount <= 0 || weight.ElementCount > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(weight), "Projection weight must fit 32-bit kernel indexing.");
        if (weight.DType.IsQuantized && weight.ElementCount % weight.DType.BlockElementCount != 0)
        {
            throw new ArgumentException(
                $"Projection weight element count {weight.ElementCount} is not aligned to {weight.DType}'s " +
                $"{weight.DType.BlockElementCount}-element blocks.", nameof(weight));
        }
        if (!float.IsFinite(weight.Fp8ScaleFactor) || weight.Fp8ScaleFactor <= 0f)
            throw new ArgumentOutOfRangeException(nameof(weight), "Projection weight scale must be finite and positive.");
    }

    private void ValidateProjectionBias(Tensor bias)
    {
        if (bias.Shape.Rank != 1 || bias.Shape[0] != _embedDim)
            throw new ArgumentException($"Projection bias must have shape [{_embedDim}]; got {bias.Shape}.", nameof(bias));
        if (bias.DType != DType.F32 && bias.DType != DType.F16 && bias.DType != DType.BF16)
            throw new NotSupportedException($"PatchEmbed projection bias must be F32, F16, or BF16; got {bias.DType}.");
    }

    private int ValidatePositionEmbedding(Tensor? posEmbed)
    {
        if (posEmbed is null)
            return 0;
        if (posEmbed.Shape.Rank != 3 || posEmbed.Shape[0] != 1 || posEmbed.Shape[2] != _embedDim)
        {
            throw new ArgumentException(
                $"Positional embedding must have shape [1,maxGrid*maxGrid,{_embedDim}]; got {posEmbed.Shape}.",
                nameof(posEmbed));
        }
        if (!posEmbed.DType.IsFloatingPoint)
            throw new NotSupportedException($"PatchEmbed positional embedding must use a floating-point dtype; got {posEmbed.DType}.");
        if (!float.IsFinite(posEmbed.Fp8ScaleFactor) || posEmbed.Fp8ScaleFactor <= 0f)
            throw new ArgumentOutOfRangeException(nameof(posEmbed), "Positional embedding scale must be finite and positive.");

        long storedRows = posEmbed.Shape[1];
        if (storedRows <= 0 || storedRows > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(posEmbed), $"Positional row count must be in [1,{int.MaxValue}]; got {storedRows}.");
        int maxSize = (int)Math.Sqrt(storedRows);
        if ((long)maxSize * maxSize != storedRows)
        {
            throw new ArgumentException(
                $"Positional embedding rows must form a square grid; got {storedRows} rows.", nameof(posEmbed));
        }
        return maxSize;
    }

    private int[] GetPositionRowIndices(int batch, int gridH, int gridW, int numPatches)
    {
        if (_cachedPositionRows is not null &&
            _cachedIndexBatch == batch && _cachedIndexGridH == gridH && _cachedIndexGridW == gridW)
        {
            return _cachedPositionRows;
        }

        int[] rows = new int[checked(batch * numPatches)];
        int startH = (_posEmbedMaxSize - gridH) / 2;
        int startW = (_posEmbedMaxSize - gridW) / 2;
        for (int r = 0; r < gridH; r++)
        {
            int sourceRow = (startH + r) * _posEmbedMaxSize + startW;
            int targetRow = r * gridW;
            for (int c = 0; c < gridW; c++)
                rows[targetRow + c] = sourceRow + c;
        }
        for (int b = 1; b < batch; b++)
            Array.Copy(rows, 0, rows, b * numPatches, numPatches);

        _cachedIndexBatch = batch;
        _cachedIndexGridH = gridH;
        _cachedIndexGridW = gridW;
        _cachedPositionRows = rows;
        return rows;
    }

    private void ValidateSpatialDimension(int value, string parameterName)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(parameterName, value, "Spatial dimension must be positive.");
        if (value % _patchSize != 0)
            throw new ArgumentException($"Spatial dimension {value} must be divisible by patch size {_patchSize}.", parameterName);
    }

    private void InvalidatePositionIndexCache()
    {
        _cachedIndexBatch = 0;
        _cachedIndexGridH = 0;
        _cachedIndexGridW = 0;
        _cachedPositionRows = null;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(PatchEmbed));
    }

    private static bool IsSupportedProjectionWeightDType(DType dtype) =>
        dtype == DType.F32 || dtype == DType.F16 || dtype == DType.BF16 || dtype == DType.F8E4M3 ||
        dtype == DType.Q8_0 || dtype == DType.Q4_0 || dtype == DType.Q5_0 ||
        dtype == DType.Q4_K || dtype == DType.Q5_K || dtype == DType.Q6_K;

    private static Exception? TryDispose(Tensor? tensor, Exception? firstError)
    {
        if (tensor is null)
            return firstError;
        try
        {
            tensor.Dispose();
        }
        catch (Exception error)
        {
            firstError ??= error;
        }
        return firstError;
    }
}
