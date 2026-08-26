using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Pins <see cref="PickleCheckpointRepacker"/> to the hand-rolled load→filter→save sequences it
/// replaced across YuE, RVC and Kokoro. Each test writes both forms and compares file hashes, so a future
/// change to the shared repacker that alters any existing model's converted weights fails here rather than
/// silently producing a different checkpoint on a user's disk.</summary>
public sealed class PickleCheckpointRepackerTests
{
    [Fact]
    public void Repack_Passthrough_MatchesHandRolledSave()
    {
        RunComparison(keyMap: null, legacyTransform: tensors => tensors);
    }

    [Fact]
    public void Repack_KeyRename_MatchesHandRolledSave()
    {
        // Kokoro strips the nn.DataParallel wrapper.
        RunComparison(
            keyMap: k => k.Replace(".module.", "."),
            legacyTransform: tensors =>
            {
                Dictionary<string, Tensor> flat = new(StringComparer.Ordinal);
                foreach ((string k, Tensor v) in tensors)
                {
                    flat[k.Replace(".module.", ".")] = v;
                }
                return flat;
            });
    }

    [Fact]
    public void Repack_KeyFilter_MatchesHandRolledSave()
    {
        // YuE's X-Codec repack keeps only the tensors its loader maps.
        RunComparison(
            keyMap: k => k.StartsWith("keep", StringComparison.Ordinal) ? k : null,
            legacyTransform: tensors =>
            {
                Dictionary<string, Tensor> keep = new(StringComparer.Ordinal);
                foreach ((string k, Tensor v) in tensors)
                {
                    if (k.StartsWith("keep", StringComparison.Ordinal))
                    {
                        keep[k] = v;
                    }
                }
                return keep;
            });
    }

    [Fact]
    public void Repack_NoSurvivingTensors_Throws()
    {
        string source = WriteSamplePickle();
        string output = Path.Combine(Path.GetTempPath(), $"hi_repack_{Guid.NewGuid():N}.safetensors");
        try
        {
            Assert.Throws<InvalidDataException>(() => PickleCheckpointRepacker.Repack(source, output, _ => null));
            Assert.False(File.Exists(output));
            Assert.False(File.Exists(output + ".tmp"));
        }
        finally
        {
            Delete(source, output, output + ".tmp");
        }
    }

    [Fact]
    public void Repack_EmbedsMetadataAndFillsHashSlot()
    {
        string source = WriteSamplePickle();
        string output = Path.Combine(Path.GetTempPath(), $"hi_repack_{Guid.NewGuid():N}.safetensors");
        string bare = Path.Combine(Path.GetTempPath(), $"hi_repack_{Guid.NewGuid():N}.safetensors");
        try
        {
            Dictionary<string, string> metadata = new(StringComparer.Ordinal)
            {
                ["modelspec.architecture"] = "yue_music",
                ["hartsy.component"] = "codec",
                ["modelspec.hash_sha256"] = "",
            };
            PickleCheckpointRepacker.Repack(source, output, keyMap: null, recursiveFlatten: false, metadata: metadata);
            PickleCheckpointRepacker.Repack(source, bare, keyMap: null);

            using SafeTensorsLoader loader = new();
            loader.Load(output);
            Assert.NotNull(loader.Metadata);
            Assert.Equal("yue_music", loader.Metadata!["modelspec.architecture"]);
            Assert.Equal("codec", loader.Metadata["hartsy.component"]);

            string embedded = loader.Metadata["modelspec.hash_sha256"];
            Assert.StartsWith("0x", embedded, StringComparison.Ordinal);
            // The embedded digest must describe the payload, so the unstamped file of the same tensors matches it.
            Assert.Equal(embedded[2..], PayloadSha256(bare));
            Assert.Equal(embedded[2..], PayloadSha256(output));
        }
        finally
        {
            Delete(source, output, bare, output + ".tmp", bare + ".tmp");
        }
    }

    private static string PayloadSha256(string path)
    {
        using FileStream fs = File.OpenRead(path);
        Span<byte> lengthBytes = stackalloc byte[8];
        fs.ReadExactly(lengthBytes);
        fs.Seek(8 + BitConverter.ToInt64(lengthBytes), SeekOrigin.Begin);
        return Convert.ToHexString(SHA256.HashData(fs)).ToLowerInvariant();
    }

    private static void RunComparison(Func<string, string?>? keyMap, Func<Dictionary<string, Tensor>, Dictionary<string, Tensor>> legacyTransform)
    {
        string source = WriteSamplePickle();
        string legacyPath = Path.Combine(Path.GetTempPath(), $"hi_legacy_{Guid.NewGuid():N}.safetensors");
        string repackPath = Path.Combine(Path.GetTempPath(), $"hi_repack_{Guid.NewGuid():N}.safetensors");
        try
        {
            using (PytorchPickleLoader loader = new())
            {
                loader.Load(source);
                SafeTensorsWriter.Save(legacyPath, legacyTransform(loader.GetAllTensors()));
            }
            PickleCheckpointRepacker.Repack(source, repackPath, keyMap);
            Assert.Equal(Sha256(legacyPath), Sha256(repackPath));
        }
        finally
        {
            Delete(source, legacyPath, repackPath);
        }
    }

    private static string Sha256(string path)
    {
        using FileStream fs = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(fs));
    }

    private static void Delete(params string[] paths)
    {
        foreach (string path in paths)
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    /// <summary>Writes a torch-style zip whose state dict has one key to rename, one to keep, and one to drop.</summary>
    private static string WriteSamplePickle()
    {
        string path = Path.Combine(Path.GetTempPath(), $"hi_repacksrc_{Guid.NewGuid():N}.pt");
        (string Key, float[] Data)[] entries =
        [
            ("keep.module.w", [1f, 2f, 3f, 4f]),
            ("keep.b", [5f, 6f, 7f, 8f]),
            ("drop.aux", [9f, 10f, 11f, 12f]),
        ];
        byte[] pickle = BuildStateDictPickle(entries.Select(e => e.Key).ToArray());
        using FileStream fs = File.Create(path);
        using ZipArchive zip = new(fs, ZipArchiveMode.Create);
        WriteEntry(zip, "archive/data.pkl", pickle);
        for (int i = 0; i < entries.Length; i++)
        {
            WriteEntry(zip, $"archive/data/{i}", FloatsToBytes(entries[i].Data));
        }
        return path;
    }

    /// <summary>Emits the opcode sequence <c>torch.save</c> produces for a flat multi-key <c>state_dict</c>
    /// of [2,2] float tensors, storage <c>i</c> backing key <c>i</c>.</summary>
    private static byte[] BuildStateDictPickle(string[] keys)
    {
        List<byte> b = [];
        b.Add(0x80); b.Add(0x02);
        Global(b, "collections", "OrderedDict");
        b.Add(0x29); b.Add((byte)'R');
        b.Add((byte)'(');
        for (int i = 0; i < keys.Length; i++)
        {
            Str(b, keys[i]);
            Global(b, "torch._utils", "_rebuild_tensor_v2");
            b.Add((byte)'(');
            b.Add((byte)'(');
            Str(b, "storage");
            Global(b, "torch", "FloatStorage");
            Str(b, i.ToString());
            Str(b, "cpu");
            Int1(b, 4);
            b.Add((byte)'t');
            b.Add((byte)'Q');
            Int1(b, 0);
            Int1(b, 2); Int1(b, 2); b.Add(0x86);
            Int1(b, 2); Int1(b, 1); b.Add(0x86);
            b.Add(0x89);
            Global(b, "collections", "OrderedDict"); b.Add(0x29); b.Add((byte)'R');
            b.Add((byte)'t');
            b.Add((byte)'R');
        }
        b.Add((byte)'u');
        b.Add((byte)'.');
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

    private static void WriteEntry(ZipArchive zip, string name, byte[] data)
    {
        ZipArchiveEntry e = zip.CreateEntry(name, CompressionLevel.NoCompression);
        using Stream s = e.Open();
        s.Write(data);
    }
}
