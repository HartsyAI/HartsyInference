using HartsyInference.Core.Tensors;

namespace HartsyInference.ModelHandler.Onnx;

/// <summary>Loads the weight initializers from an ONNX (<c>.onnx</c>) file into a
/// <c>Dictionary&lt;string, Tensor&gt;</c>, mirroring <see cref="SafeTensors.SafeTensorsLoader"/> /
/// <see cref="PyTorch.PytorchPickleLoader"/> so any model loader can consume ONNX-only checkpoints (CosyVoice2's
/// <c>campplus.onnx</c> / <c>speech_tokenizer_v2.onnx</c>, g2pW, …). This reads <b>weights only</b> — it does not
/// execute the graph — but also exposes the parsed node list (<see cref="Model"/>) in topological order so a
/// model whose ONNX export anonymized its weight names (<c>onnx::MatMul_NNNN</c>) can bind them by graph order.
/// <para>Handles FLOAT (F32) and FLOAT16 raw-data initializers, which covers the trained weights; integer
/// constants (shapes/axes) and typed (non-raw) data are not materialized — the engine reimplements each model's
/// forward, so only the float weight tensors are needed.</para></summary>
public sealed unsafe class OnnxWeightLoader : IDisposable
{
    // ONNX TensorProto.DataType values we materialize.
    private const int DtFloat = 1;
    private const int DtFloat16 = 10;

    private readonly Dictionary<string, Tensor> _tensors = [];
    private int _disposed;

    /// <summary>Parsed graph (initializer metadata + nodes in topological order). Valid after <see cref="Load"/>.</summary>
    public OnnxModel? Model { get; private set; }

    public string FilePath { get; private set; } = string.Empty;

    /// <summary>Reads the <c>.onnx</c> file, parses the graph, and materializes every FLOAT/FLOAT16 initializer
    /// into owned native memory.</summary>
    public void Load(string filePath)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        FilePath = filePath;

        byte[] buffer = File.ReadAllBytes(filePath);
        OnnxModel model = OnnxModel.Parse(buffer);
        Model = model;

        fixed (byte* basePtr = buffer)
        {
            foreach (OnnxTensor t in model.Initializers)
            {
                if (t.DataType != DtFloat && t.DataType != DtFloat16) continue;     // skip int constants
                if (t.RawLength == 0) continue;                                     // typed (non-raw) data: skipped
                DType dtype = t.DataType == DtFloat ? DType.F32 : DType.F16;
                TensorShape shape = ToShape(t.Dims);
                long expected = shape.ElementCount * dtype.SizeInBytes;
                if (expected != t.RawLength)
                {
                    throw new InvalidDataException(
                        $"ONNX initializer '{t.Name}': raw_data is {t.RawLength} bytes but shape {shape} " +
                        $"({dtype.Name}) needs {expected}.");
                }
                Tensor tensor = new(shape, dtype);
                Buffer.MemoryCopy(basePtr + t.RawOffset, (void*)tensor.DataPointer, t.RawLength, t.RawLength);
                _tensors[t.Name] = tensor;
            }
        }
    }

    /// <summary>All materialized weight tensors as name → owned Tensor. Valid until <see cref="Dispose"/>.</summary>
    public Dictionary<string, Tensor> GetAllTensors()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        return new Dictionary<string, Tensor>(_tensors);
    }

    /// <summary>Like <see cref="GetAllTensors"/> but with anonymized weight-norm-fused weights (<c>onnx::Conv_NNNN</c>)
    /// renamed to their recovered module names via <see cref="OnnxWeightNameResolver"/>, so they match the PyTorch
    /// weight keys a model loader expects.</summary>
    public Dictionary<string, Tensor> GetResolvedTensors()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (Model is null) throw new InvalidOperationException("Call Load before GetResolvedTensors.");
        return OnnxWeightNameResolver.Resolve(Model, _tensors);
    }

    private static TensorShape ToShape(long[] dims)
        => dims.Length == 0 ? new TensorShape(1) : new TensorShape(dims);     // 0-d scalar → [1]

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        foreach (Tensor t in _tensors.Values) t.Dispose();
        _tensors.Clear();
    }
}
