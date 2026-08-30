using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Cuda;

/// <summary>CUDA-owned generation session for the fixed MiniMax-H3 B1/H56/D128 VSA contract.</summary>
internal sealed unsafe class CudaVideoSparseAttentionSession : IVideoSparseAttentionSession
{
    private const int Heads = 56;
    private const int HeadDim = 128;
    private const int MaximumBlocks = 2048;

    private readonly CudaBackend _backend;
    private readonly int _sequence;
    private readonly int _blocks;
    private readonly int _paddedBlocks;
    private readonly int _prefixBlocks;
    private readonly int _keepBlocks;
    private readonly int _maxRoutes;
    private CudaModule? _module;
    private nint _centroidsKernel;
    private nint _keyMeansKernel;
    private nint _routesKernel;
    private nint _attentionKernel;
    private ulong _blockOffsets;
    private ulong _sourceIndices;
    private ulong _sourceBlock;
    private ulong _qCentroids;
    private ulong _kCentroids;
    private ulong _vCentroids;
    private ulong _keyMeans;
    private ulong _routeCounts;
    private ulong _routeBlocks;
    private int _disposed;

    /// <summary>Validates and uploads one immutable routing plan. No metadata transfer occurs during execution.</summary>
    internal CudaVideoSparseAttentionSession(CudaBackend backend, VideoSparseAttentionPlan plan, string ptxPath)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(plan);
        plan.Validate();
        if (plan.Profile is not VideoSparseAttentionProfileKind.ComfySol64V1
            and not VideoSparseAttentionProfileKind.FastVideoVsa64V1)
        {
            throw new NotSupportedException($"CUDA VSA does not implement profile value {(int)plan.Profile}.");
        }
        _backend = backend;
        _sequence = plan.SequenceLength;
        _blocks = plan.BlockOffsets.Length - 1;
        if (_blocks > MaximumBlocks)
        {
            throw new NotSupportedException(
                $"CUDA MiniMax-H3 VSA supports at most {MaximumBlocks} sparse blocks; this layout needs {_blocks}.");
        }
        _paddedBlocks = NextPowerOfTwo(_blocks);
        _prefixBlocks = plan.PrefixSinkBlocks;
        int routeableBlocks = _blocks - _prefixBlocks;
        _keepBlocks = routeableBlocks == 0
            ? 0
            : Math.Clamp((int)MathF.Ceiling(routeableBlocks * plan.KeepFraction), 1, routeableBlocks);
        _maxRoutes = Math.Min(_blocks, checked(_keepBlocks + _prefixBlocks + 3));
        Profile = plan.Profile;

        backend.BindAmbient();
        try
        {
            _module = CudaModule.LoadFromFile(ptxPath);
            _centroidsKernel = _module.GetFunction("h3_vsa_centroids_f32");
            _keyMeansKernel = _module.GetFunction("h3_vsa_key_means_f32");
            _routesKernel = _module.GetFunction("h3_vsa_routes_f32");
            _attentionKernel = _module.GetFunction("h3_vsa_attention_f32");

            int[] sourceBlock = new int[_sequence];
            Array.Fill(sourceBlock, -1);
            for (int block = 0; block < _blocks; block++)
            {
                for (int i = plan.BlockOffsets[block]; i < plan.BlockOffsets[block + 1]; i++)
                {
                    sourceBlock[plan.SourceIndices[i]] = block;
                }
            }

            _blockOffsets = AllocateAndCopy(plan.BlockOffsets);
            _sourceIndices = AllocateAndCopy(plan.SourceIndices);
            _sourceBlock = AllocateAndCopy(sourceBlock);
            nuint centroidBytes = checked((nuint)_blocks * Heads * HeadDim * sizeof(float));
            _qCentroids = CudaMemory.AllocatePersistent(centroidBytes);
            _kCentroids = CudaMemory.AllocatePersistent(centroidBytes);
            _vCentroids = CudaMemory.AllocatePersistent(centroidBytes);
            _keyMeans = CudaMemory.AllocatePersistent(Heads * HeadDim * sizeof(float));
            _routeCounts = CudaMemory.AllocatePersistent(checked((nuint)_blocks * Heads * sizeof(int)));
            _routeBlocks = CudaMemory.AllocatePersistent(
                checked((nuint)_blocks * Heads * (nuint)_maxRoutes * sizeof(int)));
        }
        catch
        {
            Release(synchronize: false);
            throw;
        }
    }

    /// <inheritdoc/>
    public VideoSparseAttentionProfileKind Profile { get; }

    /// <inheritdoc/>
    public void Execute(Tensor output, Tensor query, Tensor key, Tensor value, Tensor gate)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        ValidateTensors(output, query, key, value, gate);
        _backend.BindAmbient();

        ulong pOutput = 0;
        ulong pQuery = 0;
        ulong pKey = 0;
        ulong pValue = 0;
        ulong pGate = 0;
        bool cachedOutput = false;
        try
        {
            pQuery = GpuTransferHelper.CopyToDevice(query);
            pKey = GpuTransferHelper.CopyToDevice(key);
            pValue = GpuTransferHelper.CopyToDevice(value);
            pGate = GpuTransferHelper.CopyToDevice(gate);
            nuint outputBytes = GpuTransferHelper.ByteSize(output);
            pOutput = GpuTransferHelper.AllocateDevice(outputBytes);

            LaunchCentroids(pQuery, pKey, pValue);
            LaunchKeyMeans();
            LaunchRoutes();
            LaunchAttention(pOutput, pQuery, pKey, pValue, pGate);

            GpuTransferHelper.CacheActivation(output, pOutput, outputBytes);
            cachedOutput = true;
        }
        finally
        {
            if (!cachedOutput) GpuTransferHelper.FreeDevice(pOutput);
            GpuTransferHelper.FreeDevice(pQuery);
            GpuTransferHelper.FreeDevice(pKey);
            GpuTransferHelper.FreeDevice(pValue);
            GpuTransferHelper.FreeDevice(pGate);
        }
    }

    private void LaunchKeyMeans()
    {
        ulong keyMeans = _keyMeans;
        ulong kCentroids = _kCentroids;
        int blocks = _blocks;
        void** args = stackalloc void*[3];
        args[0] = &keyMeans;
        args[1] = &kCentroids;
        args[2] = &blocks;
        CudaDriverApi.cuLaunchKernel(_keyMeansKernel, Heads, 1, 1, HeadDim, 1, 1, 0,
            _backend.Stream.Handle, (nint)args, 0).ThrowOnError();
    }

    private void LaunchCentroids(ulong query, ulong key, ulong value)
    {
        ulong qCentroids = _qCentroids;
        ulong kCentroids = _kCentroids;
        ulong vCentroids = _vCentroids;
        ulong blockOffsets = _blockOffsets;
        ulong sourceIndices = _sourceIndices;
        int sequence = _sequence;
        int blocks = _blocks;
        void** args = stackalloc void*[10];
        args[0] = &qCentroids;
        args[1] = &kCentroids;
        args[2] = &vCentroids;
        args[3] = &query;
        args[4] = &key;
        args[5] = &value;
        args[6] = &blockOffsets;
        args[7] = &sourceIndices;
        args[8] = &sequence;
        args[9] = &blocks;
        CudaDriverApi.cuLaunchKernel(_centroidsKernel, (uint)_blocks, Heads, 1, HeadDim, 1, 1, 0,
            _backend.Stream.Handle, (nint)args, 0).ThrowOnError();
    }

    private void LaunchRoutes()
    {
        ulong routeCounts = _routeCounts;
        ulong routeBlocks = _routeBlocks;
        ulong qCentroids = _qCentroids;
        ulong kCentroids = _kCentroids;
        ulong keyMeans = _keyMeans;
        int blocks = _blocks;
        int paddedBlocks = _paddedBlocks;
        int prefixBlocks = _prefixBlocks;
        int keepBlocks = _keepBlocks;
        int maxRoutes = _maxRoutes;
        int profile = (int)Profile;
        void** args = stackalloc void*[11];
        args[0] = &routeCounts;
        args[1] = &routeBlocks;
        args[2] = &qCentroids;
        args[3] = &kCentroids;
        args[4] = &keyMeans;
        args[5] = &blocks;
        args[6] = &paddedBlocks;
        args[7] = &prefixBlocks;
        args[8] = &keepBlocks;
        args[9] = &maxRoutes;
        args[10] = &profile;
        uint sharedBytes = checked((uint)_paddedBlocks * 4u * sizeof(int));
        CudaDriverApi.cuLaunchKernel(_routesKernel, (uint)_blocks, Heads, 1, 256, 1, 1, sharedBytes,
            _backend.Stream.Handle, (nint)args, 0).ThrowOnError();
    }

    private void LaunchAttention(ulong output, ulong query, ulong key, ulong value, ulong gate)
    {
        ulong qCentroids = _qCentroids;
        ulong kCentroids = _kCentroids;
        ulong vCentroids = _vCentroids;
        ulong blockOffsets = _blockOffsets;
        ulong sourceIndices = _sourceIndices;
        ulong sourceBlock = _sourceBlock;
        ulong routeCounts = _routeCounts;
        ulong routeBlocks = _routeBlocks;
        int sequence = _sequence;
        int blocks = _blocks;
        int maxRoutes = _maxRoutes;
        void** args = stackalloc void*[16];
        args[0] = &output;
        args[1] = &query;
        args[2] = &key;
        args[3] = &value;
        args[4] = &gate;
        args[5] = &qCentroids;
        args[6] = &kCentroids;
        args[7] = &vCentroids;
        args[8] = &blockOffsets;
        args[9] = &sourceIndices;
        args[10] = &sourceBlock;
        args[11] = &routeCounts;
        args[12] = &routeBlocks;
        args[13] = &sequence;
        args[14] = &blocks;
        args[15] = &maxRoutes;
        CudaDriverApi.cuLaunchKernel(_attentionKernel, (uint)_sequence, Heads, 1, 32, 1, 1, 0,
            _backend.Stream.Handle, (nint)args, 0).ThrowOnError();
    }

    private void ValidateTensors(Tensor output, Tensor query, Tensor key, Tensor value, Tensor gate)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(query);
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        ArgumentNullException.ThrowIfNull(gate);
        TensorShape expected = new TensorShape(1, Heads, _sequence, HeadDim);
        if (query.Shape != expected || key.Shape != expected || value.Shape != expected
            || gate.Shape != expected || output.Shape != expected)
        {
            throw new ArgumentException($"CUDA MiniMax-H3 VSA requires Q/K/V/gate/output [1,{Heads},{_sequence},{HeadDim}].");
        }
        if (query.DType != DType.F32 || key.DType != DType.F32 || value.DType != DType.F32
            || gate.DType != DType.F32 || output.DType != DType.F32)
        {
            throw new NotSupportedException("CUDA MiniMax-H3 VSA requires F32 Q/K/V/gate/output tensors.");
        }
    }

    private static int NextPowerOfTwo(int value)
    {
        int result = 1;
        while (result < value)
        {
            result <<= 1;
        }
        return result;
    }

    private static ulong AllocateAndCopy(int[] values)
    {
        nuint bytes = checked((nuint)values.Length * sizeof(int));
        ulong device = CudaMemory.AllocatePersistent(bytes);
        try
        {
            fixed (int* source = values)
            {
                CudaMemory.CopyHostToDevice(device, source, bytes);
            }
            return device;
        }
        catch
        {
            CudaMemory.Free(device);
            throw;
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _backend.BindAmbient();
        Release(synchronize: true);
        GC.SuppressFinalize(this);
    }

    private void Release(bool synchronize)
    {
        if (synchronize)
        {
            _backend.Stream.Synchronize();
        }
        Free(ref _routeBlocks);
        Free(ref _routeCounts);
        Free(ref _keyMeans);
        Free(ref _vCentroids);
        Free(ref _kCentroids);
        Free(ref _qCentroids);
        Free(ref _sourceBlock);
        Free(ref _sourceIndices);
        Free(ref _blockOffsets);
        _module?.Dispose();
        _module = null;
    }

    private static void Free(ref ulong pointer)
    {
        ulong value = pointer;
        pointer = 0;
        if (value != 0)
        {
            CudaMemory.Free(value);
        }
    }
}
