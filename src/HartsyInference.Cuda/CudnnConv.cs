using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text;
using HartsyInference.Core.Configuration;
using static HartsyInference.Cuda.CudnnApi;

namespace HartsyInference.Cuda;

/// <summary>Cumulative cuDNN convolution plan-cache counters for one backend; all zero before its first cuDNN conv.
/// A plan is built once per distinct shape with the time extent included. Without length buckets every new length runs
/// the heuristic query for every conv shape it uses; with them (1D convs) a new length only finalizes plans from its
/// bucket's engine choice. Read a delta around a call to see what it built.</summary>
/// <param name="Executions">Convolutions run through cuDNN.</param>
/// <param name="PlanBuilds">Execution plans built for exact shapes (cache misses), however the engine was chosen.</param>
/// <param name="BucketPlanBuilds">Part of <paramref name="PlanBuilds"/>: planned from a length bucket's engine choice,
/// with no heuristic query.</param>
/// <param name="BucketFallbacks">Plans whose bucket choice did not finalize for their length, so that length ran its own
/// heuristic (counted in <paramref name="PlanBuilds"/> too).</param>
/// <param name="ReferenceBuilds">Heuristic queries at a bucket's reference length: one per 1D conv family and bucket.</param>
/// <param name="CachedPlans">Plans held in the cache now.</param>
/// <param name="BuildMs">Wall time spent building plans, reference builds included.</param>
/// <param name="GraphMs">Part of <paramref name="BuildMs"/>: the tensor, operation and graph descriptors.</param>
/// <param name="HeuristicMs">Part of <paramref name="BuildMs"/>: the heuristic query.</param>
/// <param name="FinalizeMs">Part of <paramref name="BuildMs"/>: execution-plan finalize, rejected candidates included.</param>
/// <param name="ConfigsTried">Engine configs finalized by heuristic builds.</param>
/// <param name="RuntimeCompiledBuilds">Heuristic builds whose chosen engine is runtime-compiled.</param>
public readonly record struct CudnnConvPlanStats(long Executions, long PlanBuilds, long BucketPlanBuilds, long BucketFallbacks,
    long ReferenceBuilds, int CachedPlans, double BuildMs, double GraphMs, double HeuristicMs, double FinalizeMs, long ConfigsTried,
    long RuntimeCompiledBuilds);

/// <summary>Convolution forward via cuDNN's backend graph API. Replaces the im2col→cuBLAS GEMM path for F16/BF16 NCHW convolutions: cuDNN's heuristics pick tensor-core implicit-GEMM/Winograd engines that never materialize the im2col matrix (an extra kH·kW-times-input-sized HBM write+read per conv — the dominant conv cost in the SDXL UNet, which runs ~50 convolutions per step). Graph = a single CONVOLUTION_FORWARD op (X ⊛ W → Y) or CONVOLUTION_BACKWARD_DATA op (transposed convolution: DY ⊛ W → DX) — cross-correlation, fp32 accumulate — over NCHW strided tensors, alpha 1 / beta 0. Bias stays a separate kernel in the caller — same numerics as the GEMM path's bias add. Execution plans (heuristics + JIT) are cached by shape+dtype; workspace comes from the stream-ordered pool per execution (capped per plan). Instances are per <see cref="CudaBackend"/> (one cuDNN handle bound to the compute stream). Any failure is caught by the caller, which self-disables the route for the session and falls back to im2col — a wrong shape costs one warning, never a session kill.</summary>
/// <remarks>Plans are shape-exact, the time extent included, and building one is almost all heuristic query: on the 3060
/// about 2 ms of heuristic against 0.01 ms of finalize per plan, and a new Kokoro sentence length needs about 39 of them.
/// So a 1D conv (H = 1) takes its engine configuration from its length bucket. The heuristic runs once per conv family and
/// power-of-two length bucket, at the bucket's reference length, and every length in the bucket finalizes its plan from
/// that configuration (engine and knobs). The reference is odd (the bucket's top length minus one), so the heuristic picks
/// a configuration that handles a ragged final tile and therefore fits every length in the bucket; at a power of two it
/// picked edge-free tiles that only fit aligned lengths. The choice depends on the length alone, never on which length
/// came first. A configuration that does not finalize for a length falls back to that length's own heuristic.
/// <c>numerics.audioConvLengthBuckets=false</c> runs the heuristic for every new length, as before.</remarks>
internal sealed class CudnnConv : IDisposable
{
    // Engine configs demanding more scratch than this are skipped in favor of the next candidate — the audio
    // codecs cache a plan per (shape, dilation) pair, so an FFT-class engine with a multi-GB workspace on one
    // waveform-length conv would otherwise dominate device memory.
    private const long MaxWorkspaceBytes = 512L * 1024 * 1024;

    private readonly nint _handle;
    private readonly nint _stream;
    private readonly ConcurrentDictionary<string, Plan> _plans = new();
    private bool _disposed;

    // Plan-cache accounting, read through Stats / DescribeFamilies. A family is a plan key without its time extent.
    private long _executions;
    private long _planBuilds;
    private long _bucketPlanBuilds;
    private long _bucketFallbacks;
    private long _referenceBuilds;
    private long _buildTicks;
    private long _graphTicks;
    private long _heuristicTicks;
    private long _finalizeTicks;
    private long _configsTried;
    private long _runtimeCompiledBuilds;
    private readonly ConcurrentDictionary<string, FamilyStats> _families = new();

    // Engine choice per (1D conv family, length bucket). Lazy so concurrent first users of a bucket run its heuristic
    // once, the way CudnnSdpa guards its plans; a null value means the reference build failed and that bucket's lengths
    // keep their own heuristic. The exact-shape plan cache above stays a plain GetOrAdd, as before.
    private readonly ConcurrentDictionary<string, Lazy<EngineChoice?>> _choices = new();

    private sealed class FamilyStats
    {
        public long Builds;
        public long BucketBuilds;
        public long References;
        public long Ticks;
        public long HeuristicTicks;
        public long FinalizeTicks;
        public long ConfigsTried;
        public readonly SortedSet<long> Engines = new();
        public bool RuntimeCompiled;
    }

    /// <summary>A 1D (H = 1) convolution with its time extent left out: what the plans of one length bucket share. The
    /// input is the op's data input, X for a forward conv and DY for a transposed one (convolution-backward-data).</summary>
    private readonly record struct Conv1dGeometry(bool BackwardData, long N, long InChannels, long OutChannels, long Kernel,
        long Stride, long PadPre, long PadPost, long Dilation, int DataType)
    {
        /// <summary>The output length for input length <paramref name="length"/>.</summary>
        public long OutputLength(long length) => BackwardData
            ? (length - 1) * Stride + Dilation * (Kernel - 1) + 1 - PadPre - PadPost
            : (length + PadPre + PadPost - Dilation * (Kernel - 1) - 1) / Stride + 1;
    }

    /// <summary>Where a 1D plan build takes its engine choice from: the geometry, the length bucket, and the family it is
    /// accounted under.</summary>
    private sealed record BucketRequest(Conv1dGeometry Geometry, long Bucket, string Family)
    {
        /// <summary>The length the bucket's heuristic runs at; see <see cref="CudnnConv.ReferenceLength"/>.</summary>
        public long ReferenceLength => CudnnConv.ReferenceLength(Bucket);
    }

    private sealed class Plan
    {
        public nint Execution;
        public long WorkspaceBytes;
    }

    public CudnnConv(nint stream)
    {
        int st = cudnnCreate(out nint handle);
        if (st != CUDNN_STATUS_SUCCESS)
            throw new InvalidOperationException($"cudnnCreate failed: {ErrorString(st)}");
        try
        {
            Check(cudnnSetStream(handle, stream), "cudnnSetStream");
        }
        catch (Exception setupFailure)
        {
            int cleanupStatus = cudnnDestroy(handle);
            if (cleanupStatus != CUDNN_STATUS_SUCCESS)
            {
                throw new AggregateException(
                    "cuDNN convolution setup and handle rollback both failed.", setupFailure,
                    new CudnnStatusException(cleanupStatus, "cudnnDestroy after setup failure"));
            }
            throw;
        }
        _handle = handle;
        _stream = stream;
    }

    private const long UidX = 1, UidW = 2, UidY = 3;

    /// <summary>Runs X[n,c,h,w] ⊛ W[k,c,r,s] → Y[n,k,outH,outW]. All pointers are device buffers of <paramref name="dataType"/> (CUDNN_DATA_HALF / CUDNN_DATA_BFLOAT16), contiguous NCHW. W-padding may be asymmetric (<paramref name="padWPre"/> zeros before, <paramref name="padWPost"/> after) — the backend graph API keeps PRE/POST paddings as separate attributes, which lets causal (left-padded) 1D convs run without an explicit pad-copy.</summary>
    public unsafe void Execute(ulong x, ulong w, ulong y,
        long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH = 1, long dilationW = 1)
    {
        string key = $"f|{n},{c},{h},{wIn}|{k},{r},{s}|{strideH},{strideW},{padH},{padWPre},{padWPost}|{dilationH},{dilationW}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ =>
        {
            string family = $"f|{n},{c},{h}|{k},{r},{s}|{strideH},{strideW},{padH},{padWPre},{padWPost}|{dilationH},{dilationW}|{dataType}";
            BucketRequest? bucket = Is1d(h, r, outH, strideH, padH, dilationH)
                ? BucketFor(new Conv1dGeometry(false, n, c, k, s, strideW, padWPre, padWPost, dilationW, dataType), wIn, outW, family)
                : null;
            return BuildPlan(backwardData: false, n, c, h, wIn, k, r, s, outH, outW, strideH, strideW, padH, padWPre, padWPost,
                dataType, dilationH, dilationW, family, bucket);
        });
        Run(plan, x, w, y);
    }

    /// <summary>Forward convolution over CHANNELS-LAST buffers, 2 or 3 spatial dims: X <c>[N, spatial…, C]</c>,
    /// W <c>[K, kernel…, C]</c>, Y <c>[N, outSpatial…, K]</c>, symmetric padding. cuDNN's tensor-core engines are built
    /// for this layout: at 3×3, 256-512 channels and BF16 they run ~155-165 TFLOPS on a 4090, where the same conv over
    /// NCHW gets ~40. The dims arrays are in logical NC… order (<c>[N, C, spatial…]</c>, <c>[K, C, kernel…]</c>).</summary>
    public void ExecuteChannelsLast(ulong x, ulong w, ulong y, long[] xDim, long[] wDim, long[] yDim,
        long[] strides, long[] pads, int dataType)
    {
        string key = $"cl|{string.Join(',', xDim)}|{string.Join(',', wDim)}|{string.Join(',', strides)}|{string.Join(',', pads)}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ =>
        {
            long[] dil = new long[strides.Length];
            Array.Fill(dil, 1L);
            return BuildPlanNd(backwardData: false, xDim, ChannelsLastStrides(xDim), wDim, ChannelsLastStrides(wDim),
                yDim, ChannelsLastStrides(yDim), strides, pads, pads, dil, dataType, family: key, bucket: null);
        });
        Run(plan, x, w, y);
    }

    /// <summary>Element strides of a channels-last buffer for logical dims <c>[N, C, s1, …, sd]</c>.</summary>
    private static long[] ChannelsLastStrides(long[] dims)
    {
        long[] strides = new long[dims.Length];
        long inner = dims[1];                    // channels are innermost
        strides[1] = 1;
        for (int i = dims.Length - 1; i >= 2; i--)
        {
            strides[i] = inner;
            inner *= dims[i];
        }
        strides[0] = inner;
        return strides;
    }

    /// <summary>Transposed convolution as cuDNN convolution-backward-data: DY[n,k,h,wIn] (the transpose-conv input) ⊛ W[k,c,r,s] → DX[n,c,outH,outW]. Geometry attributes describe the corresponding FORWARD conv (DX is the conv input), so the pads crop the full transposed output: outW = (wIn−1)·strideW + dilationW·(s−1) + 1 − padWPre − padWPost.</summary>
    public unsafe void ExecuteBackwardData(ulong dy, ulong w, ulong dx,
        long n, long k, long c, long h, long wIn, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH = 1, long dilationW = 1)
    {
        string key = $"d|{n},{c},{h},{wIn}|{k},{r},{s}|{strideH},{strideW},{padH},{padWPre},{padWPost}|{dilationH},{dilationW}|{dataType}";
        Plan plan = _plans.GetOrAdd(key, _ =>
        {
            string family = $"d|{n},{c},{h}|{k},{r},{s}|{strideH},{strideW},{padH},{padWPre},{padWPost}|{dilationH},{dilationW}|{dataType}";
            // The transposed conv's data input is DY (k channels, length wIn); its output DX has c channels.
            BucketRequest? bucket = Is1d(h, r, outH, strideH, padH, dilationH)
                ? BucketFor(new Conv1dGeometry(true, n, k, c, s, strideW, padWPre, padWPost, dilationW, dataType), wIn, outW, family)
                : null;
            return BuildPlan(backwardData: true, n, c, outH, outW, k, r, s, h, wIn, strideH, strideW, padH, padWPre, padWPost,
                dataType, dilationH, dilationW, family, bucket);
        });
        Run(plan, dx, w, dy);
    }

    // xPtr/yPtr are the X/Y-slot device pointers of the plan (forward: input/output; dgrad: DX/DY).
    // Workspace is taken from the stream-ordered pool per run and freed right after — dozens of cached plans
    // (audio codecs build one per shape/dilation) must not each pin a resident workspace for the session.
    private unsafe void Run(Plan plan, ulong xPtr, ulong w, ulong yPtr)
    {
        Interlocked.Increment(ref _executions);
        nint vp = 0;
        ulong workspace = 0;
        try
        {
            if (plan.WorkspaceBytes > 0)
                workspace = CudaMemory.AllocateAsync((nuint)plan.WorkspaceBytes, _stream);
            Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, out vp), "variant pack create");
            long* uids = stackalloc long[3] { UidX, UidW, UidY };
            void** ptrs = stackalloc void*[3] { (void*)xPtr, (void*)w, (void*)yPtr };
            void* ws = (void*)workspace;
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, uids);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, ptrs);
            SetAttr(vp, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &ws);
            Check(cudnnBackendFinalize(vp), "variant pack finalize");
            Check(cudnnBackendExecute(_handle, plan.Execution, vp), "cudnnBackendExecute");
        }
        finally
        {
            if (vp != 0) cudnnBackendDestroyDescriptor(vp);
            if (workspace != 0) CudaMemory.FreeAsync(workspace, _stream);
        }
    }

    // X-slot dims are (h, wIn) and Y-slot dims (outH, outW): for forward X is the conv input and Y the output;
    // for backwardData X is DX (the large transposed output) and Y is DY (the small input) — the conv descriptor
    // always describes the forward geometry, so callers pass the slot dims accordingly.
    private Plan BuildPlan(bool backwardData, long n, long c, long h, long wIn, long k, long r, long s,
        long outH, long outW, long strideH, long strideW, long padH, long padWPre, long padWPost, int dataType,
        long dilationH, long dilationW, string family, BucketRequest? bucket)
        => BuildPlanNd(backwardData,
            [n, c, h, wIn], Packed([n, c, h, wIn]),
            [k, c, r, s], Packed([k, c, r, s]),
            [n, k, outH, outW], Packed([n, k, outH, outW]),
            [strideH, strideW], [padH, padWPre], [padH, padWPost], [dilationH, dilationW], dataType, family, bucket);

    /// <summary>Element strides of a contiguous rank-4 NCHW buffer.</summary>
    private static long[] Packed(long[] dims) => [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];

    /// <summary>Whether a conv is 1D in the time axis (H = 1 throughout), the only case length buckets apply to.</summary>
    private static bool Is1d(long h, long r, long outH, long strideH, long padH, long dilationH) =>
        h == 1 && r == 1 && outH == 1 && strideH == 1 && padH == 0 && dilationH == 1;

    /// <summary>The power-of-two length bucket of <paramref name="length"/>: the smallest power of two at or above it, which
    /// is also the length the bucket's heuristic runs at.</summary>
    internal static long LengthBucket(long length) => length <= 1 ? 1 : (long)BitOperations.RoundUpToPowerOf2((ulong)length);

    /// <summary>The length a bucket's heuristic runs at: the bucket's top length minus one. An odd length is not a multiple of
    /// any tile, so the configuration the heuristic picks there handles a ragged final tile and finalizes for every length
    /// in the bucket. Buckets 1 and 2 use their own length.</summary>
    internal static long ReferenceLength(long bucket) => bucket > 2 ? bucket - 1 : bucket;

    /// <summary>The bucket a 1D conv of input length <paramref name="length"/> plans from, or null when buckets are off or
    /// the caller's output length is not the one the geometry implies (that shape keeps its own heuristic).</summary>
    private static BucketRequest? BucketFor(Conv1dGeometry geometry, long length, long outLength, string family)
    {
        if (!EngineKnobs.AudioConvLengthBuckets.Value || length < 1 || outLength != geometry.OutputLength(length))
            return null;
        return new BucketRequest(geometry, LengthBucket(length), family);
    }

    /// <summary>The engine choice of <paramref name="bucket"/>'s family at its reference length, made on first use.</summary>
    private EngineChoice? ChoiceFor(BucketRequest bucket) =>
        _choices.GetOrAdd($"{bucket.Family}|b{bucket.Bucket}",
            _ => new Lazy<EngineChoice?>(() => ComputeChoice(bucket), LazyThreadSafetyMode.ExecutionAndPublication)).Value;

    /// <summary>Runs the heuristic for the bucket's family at the bucket's reference length and keeps the configuration it
    /// picks (engine and knobs). The reference plan itself is dropped: a conv at that length plans from the choice like
    /// every other length in the bucket. Null when no configuration was found or its knobs could not be read,
    /// so the bucket's lengths keep their own heuristic.</summary>
    private EngineChoice? ComputeChoice(BucketRequest bucket)
    {
        Conv1dGeometry g = bucket.Geometry;
        long length = bucket.ReferenceLength;
        long outLength = g.OutputLength(length);
        if (outLength < 1)
            return null;
        long[] xDim, wDim, yDim;
        if (g.BackwardData)
        {
            // Convolution-backward-data's descriptors describe the forward conv: X is DX (the output), Y is DY (the input).
            xDim = [g.N, g.OutChannels, 1, outLength];
            wDim = [g.InChannels, g.OutChannels, 1, g.Kernel];
            yDim = [g.N, g.InChannels, 1, length];
        }
        else
        {
            xDim = [g.N, g.InChannels, 1, length];
            wDim = [g.OutChannels, g.InChannels, 1, g.Kernel];
            yDim = [g.N, g.OutChannels, 1, outLength];
        }
        long start = Stopwatch.GetTimestamp();
        List<nint> owned = new();
        try
        {
            nint graph = BuildGraph(owned, g.BackwardData, xDim, Packed(xDim), wDim, Packed(wDim), yDim, Packed(yDim),
                [1, g.Stride], [0, g.PadPre], [0, g.PadPost], [1, g.Dilation], g.DataType);
            long graphTicks = Stopwatch.GetTimestamp() - start;
            PlanBuildProbe probe = new() { CaptureKnobs = true };
            (nint exec, long _) = CudnnPlanSearch.BuildExecutionPlan(_handle, graph, owned, MaxWorkspaceBytes, "conv", probe);
            cudnnBackendDestroyDescriptor(exec);
            RecordReference(bucket.Family, Stopwatch.GetTimestamp() - start, graphTicks, probe);
            return EngineChoice.From(probe);
        }
        catch (Exception ex) when (ex is CudnnStatusException or InvalidOperationException)
        {
            // No configuration at the reference length: the bucket's lengths keep their own heuristic.
            return null;
        }
        finally
        {
            foreach (nint d in owned)
                cudnnBackendDestroyDescriptor(d);
        }
    }

    /// <summary>One convolution op over N spatial dims (dims/strides are rank N+2, the rest rank N). With a
    /// <paramref name="bucket"/>, the plan is finalized from the bucket's engine choice and the heuristic runs only when
    /// that choice does not fit this shape.</summary>
    private Plan BuildPlanNd(bool backwardData, long[] xDim, long[] xStr, long[] wDim, long[] wStr,
        long[] yDim, long[] yStr, long[] strides, long[] prePads, long[] postPads, long[] dilations, int dataType,
        string family, BucketRequest? bucket)
    {
        // Taken before this build's clock starts, so a first-use reference build is not counted twice.
        EngineChoice? choice = bucket is null ? null : ChoiceFor(bucket);
        long buildStart = Stopwatch.GetTimestamp();
        List<nint> owned = new();
        try
        {
            nint graph = BuildGraph(owned, backwardData, xDim, xStr, wDim, wStr, yDim, yStr, strides, prePads, postPads,
                dilations, dataType);
            long graphTicks = Stopwatch.GetTimestamp() - buildStart;
            if (choice is not null)
            {
                long finalizeStart = Stopwatch.GetTimestamp();
                if (CudnnPlanSearch.TryPlanFromChoice(_handle, graph, owned, choice, MaxWorkspaceBytes,
                        out nint chosen, out long chosenWorkspace))
                {
                    long now = Stopwatch.GetTimestamp();
                    RecordBucketBuild(family, now - buildStart, graphTicks, now - finalizeStart, choice);
                    return new Plan
                    {
                        Execution = chosen,
                        WorkspaceBytes = chosenWorkspace,
                    };
                }
                Interlocked.Increment(ref _bucketFallbacks);
            }
            PlanBuildProbe probe = new();
            (nint exec, long wsBytes) = CudnnPlanSearch.BuildExecutionPlan(_handle, graph, owned, MaxWorkspaceBytes, "conv", probe);
            RecordBuild(family, Stopwatch.GetTimestamp() - buildStart, graphTicks, probe);
            return new Plan
            {
                Execution = exec,
                WorkspaceBytes = wsBytes,
            };
        }
        finally
        {
            foreach (nint d in owned)
                cudnnBackendDestroyDescriptor(d);
        }
    }

    /// <summary>The tensor, convolution, operation and operation-graph descriptors of one conv, finalized; every descriptor
    /// it creates goes into <paramref name="owned"/>.</summary>
    private unsafe nint BuildGraph(List<nint> owned, bool backwardData, long[] xDim, long[] xStr, long[] wDim, long[] wStr,
        long[] yDim, long[] yStr, long[] strides, long[] prePads, long[] postPads, long[] dilations, int dataType)
    {
        nint tX = Tensor(owned, UidX, xDim, xStr, dataType);
        nint tW = Tensor(owned, UidW, wDim, wStr, dataType);
        nint tY = Tensor(owned, UidY, yDim, yStr, dataType);

        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, out nint conv), "conv desc create");
        owned.Add(conv);
        long spatial = strides.Length;
        SetAttr(conv, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial);
        int comp = CUDNN_DATA_FLOAT;
        SetAttr(conv, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp);
        int mode = CUDNN_CROSS_CORRELATION;
        SetAttr(conv, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
        fixed (long* dil = dilations, str = strides, pre = prePads, post = postPads)
        {
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, spatial, dil);
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, spatial, str);
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, spatial, pre);
            SetAttr(conv, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, spatial, post);
        }
        Check(cudnnBackendFinalize(conv), "conv desc finalize");

        int opDescType = backwardData ? CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR
            : CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
        Check(cudnnBackendCreateDescriptor(opDescType, out nint op), "conv op create");
        owned.Add(op);
        float alpha = 1.0f, beta = 0.0f;
        SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
            CUDNN_TYPE_FLOAT, 1, &alpha);
        SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
            CUDNN_TYPE_FLOAT, 1, &beta);
        void* cp = (void*)conv, xp = (void*)tX, wp = (void*)tW, yp = (void*)tY;
        SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cp);
        SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xp);
        SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wp);
        SetAttr(op, backwardData ? CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY : CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
            CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yp);
        Check(cudnnBackendFinalize(op), "conv op finalize");

        nint graph;
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, out graph), "graph create");
        owned.Add(graph);
        void* hp = (void*)_handle;
        nint* ops = stackalloc nint[1] { op };
        SetAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &hp);
        SetAttr(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops);
        Check(cudnnBackendFinalize(graph), "graph finalize");
        return graph;
    }


    private unsafe nint Tensor(List<nint> owned, long uid, long[] dims, long[] strides, int dtype)
    {
        Check(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, out nint t), "tensor create");
        owned.Add(t);
        int dt = dtype;
        SetAttr(t, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dt);
        fixed (long* d = dims, st = strides)
        {
            SetAttr(t, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims.Length, d);
            SetAttr(t, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, strides.Length, st);
        }
        long id = uid;
        SetAttr(t, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id);
        long align = 16;
        SetAttr(t, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &align);
        byte v = 0;
        SetAttr(t, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &v);
        Check(cudnnBackendFinalize(t), "tensor finalize");
        return t;
    }


    /// <summary>The cumulative counters; see <see cref="CudnnConvPlanStats"/>.</summary>
    internal CudnnConvPlanStats Stats => new(
        Interlocked.Read(ref _executions), Interlocked.Read(ref _planBuilds), Interlocked.Read(ref _bucketPlanBuilds),
        Interlocked.Read(ref _bucketFallbacks), Interlocked.Read(ref _referenceBuilds), _plans.Count,
        TicksToMs(Interlocked.Read(ref _buildTicks)), TicksToMs(Interlocked.Read(ref _graphTicks)),
        TicksToMs(Interlocked.Read(ref _heuristicTicks)), TicksToMs(Interlocked.Read(ref _finalizeTicks)),
        Interlocked.Read(ref _configsTried), Interlocked.Read(ref _runtimeCompiledBuilds));

    /// <summary>One line per conv family (a plan key without its time extent), costliest first: plans built and how many of
    /// them came from a bucket's choice, reference builds, mean build time with its heuristic and finalize parts, configs
    /// tried per heuristic build, and every engine the family used.</summary>
    internal string DescribeFamilies(int top)
    {
        StringBuilder text = new();
        foreach (KeyValuePair<string, FamilyStats> entry in _families.ToArray()
            .OrderByDescending(e => Interlocked.Read(ref e.Value.Ticks)).Take(top))
        {
            FamilyStats f = entry.Value;
            lock (f)
            {
                double builds = Math.Max(1, f.Builds + f.References);
                double heuristicBuilds = Math.Max(1, f.Builds - f.BucketBuilds + f.References);
                text.Append(entry.Key).Append(": plans=").Append(f.Builds)
                    .Append(" (from bucket ").Append(f.BucketBuilds).Append(") references=").Append(f.References)
                    .Append(" mean=").Append(Format(TicksToMs(f.Ticks) / builds))
                    .Append(" ms (heuristic ").Append(Format(TicksToMs(f.HeuristicTicks) / builds))
                    .Append(", finalize ").Append(Format(TicksToMs(f.FinalizeTicks) / builds))
                    .Append(", configs ").Append((f.ConfigsTried / heuristicBuilds).ToString("F1", CultureInfo.InvariantCulture))
                    .Append(") engines=").Append(string.Join(',', f.Engines))
                    .Append(f.RuntimeCompiled ? " runtime-compiled" : "")
                    .AppendLine();
            }
        }
        return text.ToString();
    }

    /// <summary>A plan built for an exact shape by that shape's own heuristic.</summary>
    private void RecordBuild(string family, long ticks, long graphTicks, PlanBuildProbe probe)
    {
        Interlocked.Increment(ref _planBuilds);
        RecordHeuristic(family, ticks, graphTicks, probe, reference: false);
    }

    /// <summary>A heuristic query at a bucket's reference length.</summary>
    private void RecordReference(string family, long ticks, long graphTicks, PlanBuildProbe probe)
    {
        Interlocked.Increment(ref _referenceBuilds);
        RecordHeuristic(family, ticks, graphTicks, probe, reference: true);
    }

    private void RecordHeuristic(string family, long ticks, long graphTicks, PlanBuildProbe probe, bool reference)
    {
        Interlocked.Add(ref _buildTicks, ticks);
        Interlocked.Add(ref _graphTicks, graphTicks);
        Interlocked.Add(ref _heuristicTicks, probe.HeuristicTicks);
        Interlocked.Add(ref _finalizeTicks, probe.FinalizeTicks);
        Interlocked.Add(ref _configsTried, probe.ConfigsTried);
        if (probe.RuntimeCompiled) Interlocked.Increment(ref _runtimeCompiledBuilds);
        FamilyStats f = _families.GetOrAdd(family, _ => new FamilyStats());
        lock (f)
        {
            if (reference) f.References++;
            else f.Builds++;
            f.Ticks += ticks;
            f.HeuristicTicks += probe.HeuristicTicks;
            f.FinalizeTicks += probe.FinalizeTicks;
            f.ConfigsTried += probe.ConfigsTried;
            if (probe.EngineGlobalIndex >= 0) f.Engines.Add(probe.EngineGlobalIndex);
            f.RuntimeCompiled |= probe.RuntimeCompiled;
        }
    }

    /// <summary>A plan built for an exact shape from its bucket's engine choice.</summary>
    private void RecordBucketBuild(string family, long ticks, long graphTicks, long finalizeTicks, EngineChoice choice)
    {
        Interlocked.Increment(ref _planBuilds);
        Interlocked.Increment(ref _bucketPlanBuilds);
        Interlocked.Add(ref _buildTicks, ticks);
        Interlocked.Add(ref _graphTicks, graphTicks);
        Interlocked.Add(ref _finalizeTicks, finalizeTicks);
        FamilyStats f = _families.GetOrAdd(family, _ => new FamilyStats());
        lock (f)
        {
            f.Builds++;
            f.BucketBuilds++;
            f.Ticks += ticks;
            f.FinalizeTicks += finalizeTicks;
            f.Engines.Add(choice.GlobalIndex);
        }
    }

    private static double TicksToMs(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

    private static string Format(double ms) => ms.ToString("F2", CultureInfo.InvariantCulture);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (Plan p in _plans.Values)
        {
            if (p.Execution != 0) cudnnBackendDestroyDescriptor(p.Execution);
        }
        _plans.Clear();
        if (_handle != 0) cudnnDestroy(_handle);
    }
}
