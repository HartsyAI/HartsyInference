using HartsyInference.Core.Moe;

namespace HartsyInference.Cuda;

/// <summary>
/// The expert FFN kernels (<c>Kernels/moe/expert_f32.cu</c>, shipped as <c>Ptx/expert_f32.ptx</c>), loaded as their own module.
/// It does not go through <see cref="CudaKernels"/>, whose bundle needs every sm_80 module to load before any kernel runs.
/// </summary>
/// <remarks>
/// <see cref="RunExpert"/> is the whole expert on device-resident buffers: gate and up GEMMs, the activation and clamp epilogue,
/// then the down GEMM. It is plain F32 and reads nothing but the buffers it is given, so the caller decides where the weights
/// come from. The module loads into the context current on the calling thread.
/// </remarks>
public sealed class CudaExpertKernels : IDisposable
{
    /// <summary>File name of the PTX module, without directory.</summary>
    public const string PtxFileName = "expert_f32.ptx";

    private const uint BlockSize = 128;

    private readonly CudaModule _module;
    private readonly nint _gemm;
    private readonly nint _activation;
    private bool _disposed;

    /// <summary>Loads <see cref="PtxFileName"/> from <paramref name="ptxDir"/>.</summary>
    /// <exception cref="FileNotFoundException">The module is not in <paramref name="ptxDir"/>.</exception>
    public CudaExpertKernels(string ptxDir)
    {
        ArgumentException.ThrowIfNullOrEmpty(ptxDir);
        string path = Path.Combine(ptxDir, PtxFileName);
        if (!File.Exists(path)) throw new FileNotFoundException("The expert FFN PTX module is missing.", path);
        _module = CudaModule.LoadFromFile(path);
        _gemm = _module.GetFunction("expert_gemm_f32");
        _activation = _module.GetFunction("expert_act_f32");
    }

    /// <summary>Device buffers of one expert run: the three resident weights, then the input, the gate, up and hidden scratch,
    /// and the output.</summary>
    /// <param name="W1">Gate weight, row-major <c>[I, H]</c>.</param>
    /// <param name="W3">Up weight, row-major <c>[I, H]</c>.</param>
    /// <param name="W2">Down weight, row-major <c>[H, I]</c>.</param>
    /// <param name="X"><c>rows × H</c> input.</param>
    /// <param name="Gate"><c>rows × I</c> scratch.</param>
    /// <param name="Up"><c>rows × I</c> scratch.</param>
    /// <param name="Hidden"><c>rows × I</c> scratch.</param>
    /// <param name="Y"><c>rows × H</c> output.</param>
    public readonly record struct Buffers(ulong W1, ulong W3, ulong W2, ulong X, ulong Gate, ulong Up, ulong Hidden, ulong Y);

    /// <summary>Activation code of an <see cref="ExpertActivation"/>, as the kernel reads it.</summary>
    public static int ActivationCode(ExpertActivation activation) => activation switch
    {
        ExpertActivation.Silu => 0,
        ExpertActivation.GeluTanh => 1,
        ExpertActivation.Relu => 2,
        ExpertActivation.ReluSquared => 3,
        _ => throw new ArgumentOutOfRangeException(nameof(activation), activation, "Unknown activation."),
    };

    /// <summary>Runs one expert over <paramref name="rows"/> rows on <paramref name="stream"/>:
    /// <c>Y = Down · (act(clampGate(Gate·X)) * clampUp(Up·X))</c>.</summary>
    /// <remarks>Enqueues the work and returns; <c>Y</c> is complete once <paramref name="stream"/> has drained.</remarks>
    public void RunExpert(in Buffers buffers, int rows, int hidden, int intermediate, ExpertProgram program, nint stream)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(program);
        program.Validated();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hidden);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(intermediate);
        Gemm(buffers.Gate, buffers.X, buffers.W1, rows, intermediate, hidden, stream);
        Gemm(buffers.Up, buffers.X, buffers.W3, rows, intermediate, hidden, stream);
        Activate(buffers.Hidden, buffers.Gate, buffers.Up, (long)rows * intermediate, program, stream);
        Gemm(buffers.Y, buffers.Hidden, buffers.W2, rows, hidden, intermediate, stream);
    }

    /// <summary>Releases the module. Work already launched on a stream must finish first.</summary>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _module.Dispose();
    }

    // y[r, i] = sum_k x[r, k] * w[i, k], with w row-major [n, k].
    private unsafe void Gemm(ulong y, ulong x, ulong w, int rows, int n, int k, nint stream)
    {
        ulong yA = y, xA = x, wA = w;
        int rowsA = rows, nA = n, kA = k;
        void** a = stackalloc void*[6];
        a[0] = &yA; a[1] = &xA; a[2] = &wA; a[3] = &rowsA; a[4] = &nA; a[5] = &kA;
        uint gridX = (uint)((n + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_gemm, gridX, (uint)rows, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }

    // h[i] = act(min(gate[i], GateMax)) * clamp(up[i], UpMin, UpMax) over count elements.
    // The bounds come from the program; infinite bounds disable a clamp.
    private unsafe void Activate(ulong hidden, ulong gate, ulong up, long count, ExpertProgram program, nint stream)
    {
        ulong hA = hidden, gA = gate, uA = up;
        long cA = count;
        int actA = ActivationCode(program.Activation);
        float maxA = program.GateMax, minA = program.UpMin, upMaxA = program.UpMax;
        void** a = stackalloc void*[8];
        a[0] = &hA; a[1] = &gA; a[2] = &uA; a[3] = &cA; a[4] = &actA; a[5] = &maxA; a[6] = &minA; a[7] = &upMaxA;
        uint gridX = (uint)((count + BlockSize - 1) / BlockSize);
        CudaDriverApi.cuLaunchKernel(_activation, gridX, 1, 1, BlockSize, 1, 1, 0, stream, (nint)a, 0).ThrowOnError();
    }
}
