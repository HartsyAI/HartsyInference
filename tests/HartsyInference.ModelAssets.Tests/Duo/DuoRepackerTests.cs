using System.Security.Cryptography;
using System.Text.Json;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.Duo;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;

namespace HartsyInference.ModelAssets.Tests.Duo;

/// <summary>End-to-end repack on a tiny synthetic backbone (2 heads x head dim 2, hidden 4) and a rank-2 LoRA in a synthetic distributed checkpoint. Values are small integers so every BF16 and F32 step is exact.</summary>
public sealed class DuoRepackerTests : IDisposable
{
    private const int Heads = 2, HeadDim = 2, Inner = 4, Hidden = 4, Rank = 2;
    private readonly string _root = Path.Combine(Path.GetTempPath(), $"duo-repack-{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    private static float[] Ints(int count, int seed, int span = 5)
    {
        Random rng = new(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(rng.Next(span * 2 + 1) - span)).ToArray();
    }

    private static float[] Small(int count, int seed) => Ints(count, seed).Select(v => v / 16f).ToArray();

    private static unsafe void WriteShard(string path, params (string Name, DcpFixture.Tensor1 Tensor)[] tensors)
    {
        List<SafeTensorsStreamWriter.Entry> entries = [];
        foreach ((string name, DcpFixture.Tensor1 source) in tensors)
        {
            DcpFixture.Tensor1 t = source;
            entries.Add(new SafeTensorsStreamWriter.Entry(name, t.DType, new TensorShape(t.Shape), () =>
            {
                Tensor tensor = new(new TensorShape(t.Shape), t.DType);
                t.Bytes.CopyTo(new Span<byte>(tensor.DataPointer, t.Bytes.Length));
                return (tensor, true);
            }));
        }
        SafeTensorsStreamWriter.Write(path, entries);
    }

    private sealed record Fixture(string Backbone, string Dcp, float[] Qkv, float[] OutProj, float[] Fc1, float[] Fc2, float[] PatchProj,
        Dictionary<string, (float[] A, float[] B)> Lora);

    private Fixture Build(bool extraModule = false, string? strayModule = null)
    {
        string backbone = Path.Combine(_root, "backbone"), dcp = Path.Combine(_root, "dcp");
        Directory.CreateDirectory(backbone);
        float[] qkv = Ints(3 * Inner * Hidden, 1), outProj = Ints(Inner * Hidden, 2), fc1 = Ints(8 * Hidden, 3), fc2 = Ints(Hidden * 8, 4);
        float[] patch = Ints(Hidden * 3, 5), adaln = Small(12 * 5, 6), w1 = Small(6 * 8, 7), b1 = Small(6, 8), w2 = Small(5 * 6, 9), b2 = Small(5, 10);

        WriteShard(Path.Combine(backbone, "model-00001-of-00002.safetensors"),
            ("blocks.0.attn.qkv_proj.weight", DcpFixture.Bf16([3 * Inner, Hidden], qkv)),
            ("blocks.0.attn.out_proj.weight", DcpFixture.Bf16([Inner, Hidden], outProj)),
            ("blocks.0.mlp.fc1.weight", DcpFixture.Bf16([8, Hidden], fc1)));
        WriteShard(Path.Combine(backbone, "model-00002-of-00002.safetensors"),
            ("blocks.0.mlp.fc2.weight", DcpFixture.Bf16([Hidden, 8], fc2)),
            ("video_patch_proj.weight", DcpFixture.F32([Hidden, 3], patch)),
            ("blocks.0.adaln_proj.linear.weight", DcpFixture.Bf16([12, 5], adaln)),
            ("final_layer.adaln_proj.linear.weight", DcpFixture.Bf16([4, 5], Ints(20, 11))),
            ("time_embedder.proj_in.weight", DcpFixture.F32([6, 8], w1)),
            ("time_embedder.proj_in.bias", DcpFixture.F32([6], b1)),
            ("time_embedder.proj_out.weight", DcpFixture.F32([5, 6], w2)),
            ("time_embedder.proj_out.bias", DcpFixture.F32([5], b2)));

        Dictionary<string, (float[] A, float[] B)> lora = new()
        {
            ["transformer_blocks.0.attn.to_q"] = (Ints(Rank * Hidden, 20, 2), Ints(Inner * Rank, 21, 2)),
            ["transformer_blocks.0.attn.to_k"] = (Ints(Rank * Hidden, 22, 2), Ints(Inner * Rank, 23, 2)),
            ["transformer_blocks.0.attn.to_v"] = (Ints(Rank * Hidden, 24, 2), Ints(Inner * Rank, 25, 2)),
            ["transformer_blocks.0.attn.to_out.0"] = (Ints(Rank * Inner, 26, 2), Ints(Hidden * Rank, 27, 2)),
            ["transformer_blocks.0.ff.net.0.proj"] = (Ints(Rank * Hidden, 28, 2), Ints(8 * Rank, 29, 2)),
            ["proj_in"] = (Ints(Rank * 3, 30, 2), Ints(Hidden * Rank, 31, 2)),
        };
        if (extraModule) lora["transformer_blocks.5.attn.to_q"] = (Ints(Rank * Hidden, 32, 2), Ints(Inner * Rank, 33, 2));
        if (strayModule is not null) lora[strayModule] = (Ints(Rank * Hidden, 34, 2), Ints(Inner * Rank, 35, 2));

        Dictionary<string, DcpFixture.Tensor1> tensors = [];
        foreach ((string module, (float[] a, float[] b)) in lora)
        {
            int inDim = a.Length / Rank, outDim = b.Length / Rank;
            tensors[$"models.generator.{module}.lora_A.default.weight"] = DcpFixture.F32([Rank, inDim], a);
            tensors[$"models.generator.{module}.lora_B.default.weight"] = DcpFixture.F32([outDim, Rank], b);
        }
        tensors["models.generator.role_embeddings.weight"] = DcpFixture.F32([2, Hidden], new float[2 * Hidden]);
        DcpFixture.Write(dcp, tensors);
        return new Fixture(backbone, dcp, qkv, outProj, fc1, fc2, patch, lora);
    }

    private static float[] Delta(float[] a, float[] b, int rank)
    {
        int inDim = a.Length / rank, outDim = b.Length / rank;
        float[] delta = new float[outDim * inDim];
        for (int r = 0; r < outDim; r++)
            for (int c = 0; c < inDim; c++)
                for (int k = 0; k < rank; k++) delta[r * inDim + c] += b[r * rank + k] * a[k * inDim + c];
        return delta;
    }

    private static float[] ReadAsF32(SafeTensorsLoader loader, string name)
    {
        using Tensor t = loader.GetTensor(name).CastTo(DType.F32);
        return t.AsSpan<float>().ToArray();
    }

    [Fact]
    public void WriteLoraSafetensors_RoundTripsEveryModuleAndLeavesOutTheRoleEmbeddings()
    {
        Fixture fx = Build();
        string path = Path.Combine(_root, "lora.safetensors");
        using (DuoLoraSource dcp = DuoLoraSource.FromDcp(fx.Dcp))
        {
            Assert.Equal(fx.Lora.Keys.Order(StringComparer.Ordinal), dcp.Modules);
            DuoRepacker.WriteLoraSafetensors(dcp, path);
        }

        using DuoLoraSource written = DuoLoraSource.FromSafetensors(path);
        Assert.Equal(fx.Lora.Keys.Order(StringComparer.Ordinal), written.Modules);
        foreach ((string module, (float[] a, float[] b)) in fx.Lora)
        {
            (Tensor ta, Tensor tb) = written.Read(module);
            Assert.Equal(a, ta.AsSpan<float>().ToArray());
            Assert.Equal(b, tb.AsSpan<float>().ToArray());
            ta.Dispose();
            tb.Dispose();
        }
        using SafeTensorsLoader file = new();
        file.Load(path);
        Assert.DoesNotContain(file.Descriptors.Keys, k => k.Contains("role_embeddings", StringComparison.Ordinal));
    }

    [Fact]
    public void MergeBackbone_ReordersQkvMergesEachSliceAndSwapsFc1HalvesLikeUpstream()
    {
        Fixture fx = Build();
        string output = Path.Combine(_root, "merged");
        using (DuoLoraSource lora = DuoLoraSource.FromDcp(fx.Dcp))
        {
            IReadOnlyList<string> files = DuoRepacker.MergeBackbone(fx.Backbone, lora, output,
                new DuoMergeOptions { Heads = Heads, HeadDim = HeadDim });
            Assert.Contains("model-00001-of-00002.safetensors", files);
            Assert.Contains("model.safetensors.index.json", files);
        }

        using SafeTensorsLoader first = new(), second = new();
        first.Load(Path.Combine(output, "model-00001-of-00002.safetensors"));
        second.Load(Path.Combine(output, "model-00002-of-00002.safetensors"));

        // qkv: head-interleaved rows ((h*3+p)*2+d) land at block row (p*2+h)*2+d, then slice p takes the delta of its own module.
        string[] parts = ["to_q", "to_k", "to_v"];
        float[] expectedQkv = new float[3 * Inner * Hidden];
        for (int p = 0; p < 3; p++)
        {
            (float[] a, float[] b) = fx.Lora[$"transformer_blocks.0.attn.{parts[p]}"];
            float[] delta = Delta(a, b, Rank);
            for (int h = 0; h < Heads; h++)
                for (int d = 0; d < HeadDim; d++)
                    for (int c = 0; c < Hidden; c++)
                    {
                        int blockRow = (p * Heads + h) * HeadDim + d, sourceRow = (h * 3 + p) * HeadDim + d;
                        expectedQkv[blockRow * Hidden + c] = fx.Qkv[sourceRow * Hidden + c] + delta[(h * HeadDim + d) * Hidden + c];
                    }
        }
        Assert.Equal(expectedQkv, ReadAsF32(first, "blocks.0.attn.qkv_proj.weight"));

        (float[] oa, float[] ob) = fx.Lora["transformer_blocks.0.attn.to_out.0"];
        float[] outDelta = Delta(oa, ob, Rank);
        Assert.Equal(fx.OutProj.Select((v, i) => v + outDelta[i]).ToArray(), ReadAsF32(first, "blocks.0.attn.out_proj.weight"));

        (float[] fa, float[] fb) = fx.Lora["transformer_blocks.0.ff.net.0.proj"];
        float[] fcDelta = Delta(fa, fb, Rank);
        float[] expectedFc1 = new float[8 * Hidden];
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < Hidden; c++) expectedFc1[r * Hidden + c] = fx.Fc1[r * Hidden + c] + fcDelta[((r + 4) % 8) * Hidden + c];
        Assert.Equal(expectedFc1, ReadAsF32(first, "blocks.0.mlp.fc1.weight"));

        (float[] pa, float[] pb) = fx.Lora["proj_in"];
        float[] patchDelta = Delta(pa, pb, Rank);
        Assert.Equal(fx.PatchProj.Select((v, i) => v + patchDelta[i]).ToArray(), ReadAsF32(second, "video_patch_proj.weight"));
        Assert.Equal(fx.Fc2, ReadAsF32(second, "blocks.0.mlp.fc2.weight"));
        Assert.Contains("blocks.0.adaln_proj.linear.weight", second.Descriptors.Keys);

        using JsonDocument index = JsonDocument.Parse(File.ReadAllText(Path.Combine(output, "model.safetensors.index.json")));
        Assert.Equal("model-00002-of-00002.safetensors", index.RootElement.GetProperty("weight_map").GetProperty("video_patch_proj.weight").GetString());
    }

    [Fact]
    public void MergeBackbone_LeavesAnAlreadyBlockOrderedQkvInPlaceAndCanDropTheBlockAdalnLayers()
    {
        Fixture fx = Build();
        string output = Path.Combine(_root, "merged");
        using (DuoLoraSource lora = DuoLoraSource.FromDcp(fx.Dcp))
        {
            DuoRepacker.MergeBackbone(fx.Backbone, lora, output,
                new DuoMergeOptions { Heads = Heads, HeadDim = HeadDim, InputQkv = DuoQkvInput.Blocks, DropBlockAdalnProj = true });
        }

        using SafeTensorsLoader first = new(), second = new();
        first.Load(Path.Combine(output, "model-00001-of-00002.safetensors"));
        second.Load(Path.Combine(output, "model-00002-of-00002.safetensors"));
        (float[] a, float[] b) = fx.Lora["transformer_blocks.0.attn.to_k"];
        float[] delta = Delta(a, b, Rank);
        float[] merged = ReadAsF32(first, "blocks.0.attn.qkv_proj.weight");
        for (int r = 0; r < Inner; r++)
            for (int c = 0; c < Hidden; c++)
                Assert.Equal(fx.Qkv[(Inner + r) * Hidden + c] + delta[r * Hidden + c], merged[(Inner + r) * Hidden + c]);
        Assert.DoesNotContain("blocks.0.adaln_proj.linear.weight", second.Descriptors.Keys);
        Assert.Contains("final_layer.adaln_proj.linear.weight", second.Descriptors.Keys);
    }

    [Fact]
    public void MergeBackbone_ScalesByAlphaOverRank()
    {
        Fixture fx = Build();
        string output = Path.Combine(_root, "merged");
        using (DuoLoraSource lora = DuoLoraSource.FromDcp(fx.Dcp))
        {
            DuoRepacker.MergeBackbone(fx.Backbone, lora, output, new DuoMergeOptions { Heads = Heads, HeadDim = HeadDim, Alpha = 1.0 });
        }
        using SafeTensorsLoader second = new();
        second.Load(Path.Combine(output, "model-00002-of-00002.safetensors"));
        (float[] pa, float[] pb) = fx.Lora["proj_in"];
        float[] delta = Delta(pa, pb, Rank);
        Assert.Equal(fx.PatchProj.Select((v, i) => v + delta[i] * 0.5f).ToArray(), ReadAsF32(second, "video_patch_proj.weight"));
    }

    [Fact]
    public void MergeBackbone_FailsWhenAModuleMatchesNoBackboneTensorOrHasNoTarget()
    {
        Fixture missing = Build(extraModule: true);
        using (DuoLoraSource lora = DuoLoraSource.FromDcp(missing.Dcp))
        {
            InvalidDataException error = Assert.Throws<InvalidDataException>(() =>
                DuoRepacker.MergeBackbone(missing.Backbone, lora, Path.Combine(_root, "out1"), new DuoMergeOptions { Heads = Heads, HeadDim = HeadDim }));
            Assert.Contains("transformer_blocks.5.attn.to_q", error.Message);
        }

        Directory.Delete(_root, recursive: true);
        Fixture stray = Build(strayModule: "transformer_blocks.0.attn.add_q_proj");
        using DuoLoraSource strayLora = DuoLoraSource.FromDcp(stray.Dcp);
        InvalidDataException unmapped = Assert.Throws<InvalidDataException>(() =>
            DuoRepacker.MergeBackbone(stray.Backbone, strayLora, Path.Combine(_root, "out2"), new DuoMergeOptions { Heads = Heads, HeadDim = HeadDim }));
        Assert.Contains("add_q_proj", unmapped.Message);
    }

    [Fact]
    public void WriteAdalnTables_ComputesSiluOfTheTimeEmbeddingThroughEachBlocksAdalnProj()
    {
        Fixture fx = Build();
        float[] times = [0f, 0.0625f, 1f];
        string path = Path.Combine(_root, "tables.safetensors");
        DuoRepacker.WriteAdalnTables(fx.Backbone, times, path, DuoAdalnPrecision.F32);

        float[] w1 = Small(6 * 8, 7), b1 = Small(6, 8), w2 = Small(5 * 6, 9), b2 = Small(5, 10), adaln = Small(12 * 5, 6);
        using SafeTensorsLoader table = new();
        table.Load(path);
        float[] got = ReadAsF32(table, "blocks.0.adaln_proj.table");
        Assert.Equal(new TensorShape(times.Length * 3, 4), table.Descriptors["blocks.0.adaln_proj.table"].Shape);
        Assert.Equal(times, ReadAsF32(table, DuoRepacker.AdalnTimesKey));

        for (int t = 0; t < times.Length; t++)
        {
            double[] sin = new double[8];
            for (int i = 0; i < 4; i++)
            {
                double arg = times[t] * Math.Exp(-Math.Log(10000.0) * i / 4);
                sin[i] = Math.Cos(arg);
                sin[4 + i] = Math.Sin(arg);
            }
            double[] act = new double[6];
            for (int h = 0; h < 6; h++)
            {
                double s = b1[h];
                for (int k = 0; k < 8; k++) s += sin[k] * w1[h * 8 + k];
                act[h] = s / (1 + Math.Exp(-s));
            }
            double[] emb = new double[5];
            for (int o = 0; o < 5; o++)
            {
                double s = b2[o];
                for (int h = 0; h < 6; h++) s += act[h] * w2[o * 6 + h];
                emb[o] = s;
            }
            for (int n = 0; n < 12; n++)
            {
                double s = 0;
                for (int k = 0; k < 5; k++) s += emb[k] / (1 + Math.Exp(-emb[k])) * adaln[n * 5 + k];
                Assert.Equal(s, got[t * 12 + n], 3);
            }
        }
    }

    [Fact]
    public void Project_Bf16ReferenceRoundsTheOutputAndTheF32ModeDoesNot()
    {
        using Tensor weight = new(new TensorShape(3, 2), DType.F32);
        new float[] { 0.3f, 0.7f, -0.2f, 0.9f, 0.11f, 0.13f }.CopyTo(weight.AsSpan<float>());
        float[] embeddings = [0.37f, -1.21f];

        using Tensor exact = DuoAdalnTables.Project(embeddings, 2, weight, null, DuoAdalnPrecision.F32);
        using Tensor rounded = DuoAdalnTables.Project(embeddings, 2, weight, null, DuoAdalnPrecision.Bf16Reference);

        foreach (float v in rounded.AsSpan<float>()) Assert.Equal(0u, BitConverter.SingleToUInt32Bits(v) & 0xffffu);
        Assert.Contains(exact.AsSpan<float>().ToArray(), v => (BitConverter.SingleToUInt32Bits(v) & 0xffffu) != 0);
        Assert.Throws<ArgumentException>(() => DuoAdalnTables.Project([1f, 2f, 3f], 2, weight, null, DuoAdalnPrecision.F32));
    }

    [Fact]
    public void Manifest_ListsEachFileWithItsSizeAndSha256()
    {
        Directory.CreateDirectory(_root);
        File.WriteAllBytes(Path.Combine(_root, "a.bin"), [1, 2, 3]);
        File.WriteAllBytes(Path.Combine(_root, "b.bin"), []);

        IReadOnlyList<DuoRepackFile> files = DuoRepackManifest.Write(_root, ["b.bin", "a.bin"],
            new Dictionary<string, string> { ["lora"] = "abc" }, new Dictionary<string, string> { ["licence"] = "see LICENSE" });

        Assert.Equal(["a.bin", "b.bin"], files.Select(f => f.RelativePath));
        Assert.Equal(Convert.ToHexString(SHA256.HashData([1, 2, 3])).ToLowerInvariant(), files[0].Sha256);
        Assert.Equal(3, files[0].Bytes);
        using JsonDocument json = JsonDocument.Parse(File.ReadAllText(Path.Combine(_root, DuoRepackManifest.FileName)));
        Assert.Equal(2, json.RootElement.GetProperty("files").GetArrayLength());
        Assert.Equal("see LICENSE", json.RootElement.GetProperty("notes").GetProperty("licence").GetString());
        Assert.Equal("abc", json.RootElement.GetProperty("sources").GetProperty("lora").GetString());
    }
}
