using System.Globalization;
using System.IO.Compression;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.Tests.Duo;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.PyTorch;

/// <summary>The envelope reader on a synthetic torch checkpoint that holds several state dicts, the shape of a training
/// checkpoint: the non-EMA generator comes first in the pickle, so a reader that takes the first dict would pick it.</summary>
public sealed class PytorchEnvelopeReaderTests : IDisposable
{
    private const string Wrapper = "model._fsdp_wrapped_module.";
    private readonly string _dir = Directory.CreateTempSubdirectory("envelope-reader-").FullName;

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void ItReadsOnlyTheNamedEnvelopeAndStripsItsPrefix()
    {
        string path = Checkpoint(
            ("generator", Dict(("model._fsdp_wrapped_module.w", TensorValue([1f, 2f])))),
            ("generator_ema", Dict(("model._fsdp_wrapped_module.w", TensorValue([10f, 20f])), ("model._fsdp_wrapped_module.b", TensorValue([30f])))),
            ("critic", Dict(("model._fsdp_wrapped_module.w", TensorValue([100f, 200f])))));

        (Dictionary<string, Tensor> weights, IDisposable owner) = PytorchEnvelopeReader.Read(path, "generator_ema", Wrapper, DType.F32);
        try
        {
            Assert.Equal(new[] { "b", "w" }, weights.Keys.OrderBy(k => k, StringComparer.Ordinal).ToArray());
            Assert.Equal(new[] { 10f, 20f }, weights["w"].AsSpan<float>().ToArray());
            Assert.Equal(new[] { 30f }, weights["b"].AsSpan<float>().ToArray());
        }
        finally
        {
            owner.Dispose();
        }
    }

    [Fact]
    public void ItRefusesAFileWithoutTheEnvelopeAndNamesWhatItHas()
    {
        string path = Checkpoint(("generator", Dict(("w", TensorValue([1f])))));

        InvalidOperationException error = Assert.Throws<InvalidOperationException>(() =>
            PytorchEnvelopeReader.Read(path, "generator_ema", Wrapper, DType.F32));
        Assert.Contains("generator_ema", error.Message);
        Assert.Contains("generator", error.Message);
    }

    [Fact]
    public void ItRefusesATensorUnderTheEnvelopeWithoutTheWrapperPrefix()
    {
        string path = Checkpoint(("generator_ema", Dict(("bare.weight", TensorValue([1f])))));

        Assert.Throws<InvalidOperationException>(() =>
            PytorchEnvelopeReader.Read(path, "generator_ema", Wrapper, DType.F32));
    }

    [Fact]
    public void TheOwnerFreesTheTensorsAndDisposingItTwiceIsHarmless()
    {
        string path = Checkpoint(("generator_ema", Dict(("model._fsdp_wrapped_module.w", TensorValue([5f])))));

        (Dictionary<string, Tensor> weights, IDisposable owner) = PytorchEnvelopeReader.Read(path, "generator_ema", Wrapper, DType.F32);
        Assert.Equal(new[] { 5f }, weights["w"].AsSpan<float>().ToArray());
        owner.Dispose();
        owner.Dispose();
    }

    /// <summary>One tensor's pickle value and its storage bytes, keyed by the storage's position in the file.</summary>
    private readonly List<(string Key, byte[] Bytes)> _storages = [];

    private Action<PickleBuilder> TensorValue(float[] values)
    {
        string key = _storages.Count.ToString(CultureInfo.InvariantCulture);
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        _storages.Add((key, bytes));
        long n = values.Length;
        return b => b.Global("torch._utils", "_rebuild_tensor_v2").Tuple(
            x => x.Tuple(y => y.Str("storage"), y => y.Global("torch", "FloatStorage"), y => y.Str(key), y => y.Str("cpu"), y => y.Int(n)).Op((byte)'Q'),
            x => x.Int(0),
            x => x.Tuple(y => y.Int(n)),
            x => x.Tuple(y => y.Int(1)),
            x => x.Bool(false),
            x => x.Global("collections", "OrderedDict").Op((byte)')').Op((byte)'R')).Op((byte)'R');
    }

    private static Action<PickleBuilder> Dict(params (string Name, Action<PickleBuilder> Value)[] entries) =>
        b => b.Dict(entries.Select(e => (new Action<PickleBuilder>(x => x.Str(e.Name)), e.Value)));

    /// <summary>Writes a torch-style ZIP whose root dict holds the given named dicts, in order.</summary>
    private string Checkpoint(params (string Name, Action<PickleBuilder> Value)[] roots)
    {
        // The tensors of this checkpoint are created by the Tensor calls the caller made to build the dicts, so the root is
        // built here and the storages it refers to are written after it.
        PickleBuilder pickle = new(proto4: false);
        Dict(roots)(pickle);
        string path = Path.Combine(_dir, $"checkpoint-{Guid.NewGuid():N}.pt");
        using (MemoryStream stream = new())
        {
            using (ZipArchive zip = new(stream, ZipArchiveMode.Create, leaveOpen: true))
            {
                Add(zip, "archive/data.pkl", pickle.Finish());
                foreach ((string key, byte[] bytes) in _storages)
                {
                    Add(zip, $"archive/data/{key}", bytes);
                }
                Add(zip, "archive/version", "3\n"u8.ToArray());
            }
            File.WriteAllBytes(path, stream.ToArray());
        }
        _storages.Clear();
        return path;
    }

    private static void Add(ZipArchive zip, string name, byte[] bytes)
    {
        using Stream entry = zip.CreateEntry(name, CompressionLevel.NoCompression).Open();
        entry.Write(bytes);
    }
}
