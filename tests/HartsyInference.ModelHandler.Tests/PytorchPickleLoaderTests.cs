using System.IO.Compression;
using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelHandler.PyTorch;
using Xunit;

namespace HartsyInference.ModelHandler.Tests;

/// <summary>Round-trips a hand-built torch-format <c>.pt</c> (ZIP container + a pickle stream emitted with the
/// exact opcode sequence <c>torch.save</c> uses for a flat <c>state_dict</c>) through
/// <see cref="PytorchPickleLoader"/>. Validates the safe-subset pickle VM, the storage→dtype mapping, and the
/// blob read — without requiring a real torch dependency.</summary>
public sealed unsafe class PytorchPickleLoaderTests
{
    [Fact]
    public void Loads_FloatStateDict_FromTorchStyleZip()
    {
        string path = Path.Combine(Path.GetTempPath(), $"hi_pttest_{Guid.NewGuid():N}.pt");
        try
        {
            float[] w = [1f, 2f, 3f, 4f]; // shape [2,2]
            byte[] pickle = BuildStateDictPickle("w", storageKey: "0", numel: 4, size: [2, 2], stride: [2, 1]);
            WriteTorchZip(path, pickle, storageKey: "0", FloatsToBytes(w));

            using PytorchPickleLoader loader = new();
            loader.Load(path);
            Dictionary<string, Tensor> tensors = loader.GetAllTensors();

            Assert.True(tensors.ContainsKey("w"));
            Tensor t = tensors["w"];
            Assert.Equal(DType.F32, t.DType);
            Assert.Equal(2, t.Shape.Rank);
            Assert.Equal(2, (int)t.Shape[0]);
            Assert.Equal(2, (int)t.Shape[1]);
            float* p = (float*)t.DataPointer;
            for (int i = 0; i < 4; i++) Assert.Equal(w[i], p[i], 6);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    [Fact]
    public void Loads_FloatStateDict_FromLegacyTorchSave()
    {
        // Legacy _legacy_save layout: 5 back-to-back pickle streams then each storage's (int64 numel + raw bytes).
        float[] w = [1f, 2f, 3f, 4f]; // shape [2,2]
        List<byte> file = [];
        file.AddRange(MagicStream());                                            // 1: MAGIC_NUMBER
        file.AddRange(new byte[] { 0x80, 0x02, 0x4d, 0xe9, 0x03, (byte)'.' });   // 2: PROTOCOL_VERSION=1001
        file.AddRange(new byte[] { 0x80, 0x02, 0x7d, (byte)'.' });               // 3: sys_info (empty dict)
        file.AddRange(BuildStateDictPickle("w", storageKey: "0", numel: 4, size: [2, 2], stride: [2, 1])); // 4: state_dict
        file.AddRange(KeyListStream("0"));                                       // 5: ["0"]
        // storage "0": int64 numel then raw float bytes.
        file.AddRange(BitConverter.GetBytes((long)4));
        file.AddRange(FloatsToBytes(w));

        string path = Path.Combine(Path.GetTempPath(), $"hi_pttest_legacy_{Guid.NewGuid():N}.bin");
        try
        {
            File.WriteAllBytes(path, [.. file]);
            using PytorchPickleLoader loader = new();
            loader.Load(path);
            Dictionary<string, Tensor> tensors = loader.GetAllTensors();

            Assert.True(tensors.ContainsKey("w"));
            Tensor t = tensors["w"];
            Assert.Equal(DType.F32, t.DType);
            Assert.Equal(2, t.Shape.Rank);
            Assert.Equal(2, (int)t.Shape[0]);
            Assert.Equal(2, (int)t.Shape[1]);
            float* p = (float*)t.DataPointer;
            for (int i = 0; i < 4; i++) Assert.Equal(w[i], p[i], 6);
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    /// <summary>Stream 1: PROTO2 + LONG1(0x1950a86a20f9469cfc6c, 10 bytes LE) + STOP.</summary>
    private static byte[] MagicStream() =>
    [
        0x80, 0x02, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19, (byte)'.',
    ];

    /// <summary>Stream 5: PROTO2 + EMPTY_LIST + BINUNICODE(key) + APPEND + STOP.</summary>
    private static byte[] KeyListStream(string key)
    {
        List<byte> b = [0x80, 0x02, (byte)']'];
        Str(b, key);
        b.Add((byte)'a');
        b.Add((byte)'.');
        return [.. b];
    }

    // --- helpers: emit a torch-style pickle + zip ------------------------------------------------

    private static byte[] BuildStateDictPickle(string key, string storageKey, long numel, int[] size, int[] stride)
    {
        List<byte> b = [];
        b.Add(0x80); b.Add(0x02);                       // PROTO 2
        Global(b, "collections", "OrderedDict");
        b.Add(0x29); b.Add((byte)'R');                  // EMPTY_TUPLE, REDUCE -> OrderedDict
        b.Add((byte)'(');                               // MARK (dict items)
        Str(b, key);                                    // key
        // value = _rebuild_tensor_v2(args)
        Global(b, "torch._utils", "_rebuild_tensor_v2");
        b.Add((byte)'(');                               // MARK (args)
        // storage persistent_id tuple
        b.Add((byte)'(');                               // MARK (pid)
        Str(b, "storage");
        Global(b, "torch", "FloatStorage");
        Str(b, storageKey);
        Str(b, "cpu");
        Int1(b, (int)numel);
        b.Add((byte)'t');                               // TUPLE (pid)
        b.Add((byte)'Q');                               // BINPERSID -> storage ref
        Int1(b, 0);                                     // storage_offset
        Int1(b, size[0]); Int1(b, size[1]); b.Add(0x86); // size TUPLE2
        Int1(b, stride[0]); Int1(b, stride[1]); b.Add(0x86); // stride TUPLE2
        b.Add(0x89);                                    // NEWFALSE requires_grad
        Global(b, "collections", "OrderedDict"); b.Add(0x29); b.Add((byte)'R'); // backward_hooks = OrderedDict()
        b.Add((byte)'t');                               // TUPLE (args)
        b.Add((byte)'R');                               // REDUCE -> tensor
        b.Add((byte)'u');                               // SETITEMS -> dict[key]=tensor
        b.Add((byte)'.');                               // STOP
        return [.. b];
    }

    private static void Global(List<byte> b, string module, string name)
    {
        b.Add((byte)'c');
        b.AddRange(Encoding.ASCII.GetBytes(module)); b.Add((byte)'\n');
        b.AddRange(Encoding.ASCII.GetBytes(name)); b.Add((byte)'\n');
    }

    private static void Str(List<byte> b, string s)
    {
        byte[] u = Encoding.UTF8.GetBytes(s);
        b.Add((byte)'X');
        b.Add((byte)(u.Length & 0xff)); b.Add((byte)((u.Length >> 8) & 0xff));
        b.Add((byte)((u.Length >> 16) & 0xff)); b.Add((byte)((u.Length >> 24) & 0xff));
        b.AddRange(u);
    }

    private static void Int1(List<byte> b, int v) { b.Add((byte)'K'); b.Add((byte)v); }

    private static byte[] FloatsToBytes(float[] f)
    {
        byte[] r = new byte[f.Length * 4];
        Buffer.BlockCopy(f, 0, r, 0, r.Length);
        return r;
    }

    private static void WriteTorchZip(string path, byte[] pickle, string storageKey, byte[] storageBlob)
    {
        using FileStream fs = File.Create(path);
        using ZipArchive zip = new(fs, ZipArchiveMode.Create);
        WriteEntry(zip, "archive/data.pkl", pickle);
        WriteEntry(zip, $"archive/data/{storageKey}", storageBlob);
    }

    private static void WriteEntry(ZipArchive zip, string name, byte[] data)
    {
        ZipArchiveEntry e = zip.CreateEntry(name, CompressionLevel.NoCompression);
        using Stream s = e.Open();
        s.Write(data);
    }
}
