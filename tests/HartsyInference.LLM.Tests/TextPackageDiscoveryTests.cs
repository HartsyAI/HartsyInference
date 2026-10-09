using System.Text;
using System.Text.Json;
using HartsyInference.LLM.Tests.DeepSeekV41;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.Quant;
using Xunit;

namespace HartsyInference.LLM.Tests;

/// <summary>Discovery over a fixture tree, reading no weight data: a checkpoint directory is one package whose shards are its parts; a GGUF file is one package; a complete
/// split is one package, and an incomplete one names its missing parts; an <c>mmproj</c> beside a lone model is that model's vision sidecar, and never a model of its own.</summary>
public sealed class TextPackageDiscoveryTests : IDisposable
{
    private const string SingleFileConfig = "{\"model_type\":\"deepseek_v4\"}";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "hartsy-discovery-" + Guid.NewGuid().ToString("N"));

    /// <summary>A directory a test made unreadable; its mode is restored before the tree is deleted.</summary>
    private string? _locked;

    public TextPackageDiscoveryTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        if (_locked is not null && OperatingSystem.IsLinux())
            File.SetUnixFileMode(_locked, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true);
    }

    [Fact]
    public void Fixture_Tree_Yields_One_Package_Per_Model_With_Its_Parts_And_Sidecars()
    {
        TinyDeepSeekV41Checkpoint.Write(Directory.CreateDirectory(Path.Combine(_root, "hf-official")).FullName);
        TinyDeepSeekV41Checkpoint.Write(Directory.CreateDirectory(Path.Combine(_root, "hf-mlx")).FullName, QuantFlavor.Mlx);
        string llm = Directory.CreateDirectory(Path.Combine(_root, "llm")).FullName;
        for (int part = 1; part <= 3; part++)
            File.WriteAllBytes(Path.Combine(llm, $"qwen-{part:D5}-of-00003.gguf"), [1, 2, 3]);
        File.WriteAllBytes(Path.Combine(llm, "partial-00001-of-00002.gguf"), [1]);
        string vision = Directory.CreateDirectory(Path.Combine(llm, "vision")).FullName;
        File.WriteAllBytes(Path.Combine(vision, "single.gguf"), [1, 2, 3, 4]);
        File.WriteAllBytes(Path.Combine(vision, "mmproj-single.gguf"), [9]);
        string shared = Directory.CreateDirectory(Path.Combine(llm, "shared")).FullName;
        File.WriteAllBytes(Path.Combine(shared, "a.gguf"), [1]);
        File.WriteAllBytes(Path.Combine(shared, "b.gguf"), [1]);
        File.WriteAllBytes(Path.Combine(shared, "mmproj-shared.gguf"), [9]);
        string nested = Directory.CreateDirectory(Path.Combine(llm, "nested")).FullName;
        File.WriteAllBytes(Path.Combine(nested, "deep.gguf"), [1]);

        IReadOnlyList<TextModelPackage> packages = TextPackageDiscovery.Discover(_root).Packages;

        string[] expected = ["hf-mlx", "hf-official", "llm/nested/deep.gguf", "llm/partial", "llm/qwen", "llm/shared/a.gguf", "llm/shared/b.gguf", "llm/vision/single.gguf"];
        Assert.Equal(expected, packages.Select(p => p.Id).ToArray());
        Assert.DoesNotContain(packages, p => p.Id.Contains("mmproj", StringComparison.Ordinal));
    }

    [Fact]
    public void Checkpoint_Directories_Are_Packages_Whose_Shards_Are_Their_Parts()
    {
        TinyDeepSeekV41Checkpoint.Write(Directory.CreateDirectory(Path.Combine(_root, "hf-official")).FullName);
        TinyDeepSeekV41Checkpoint.Write(Directory.CreateDirectory(Path.Combine(_root, "hf-mlx")).FullName, QuantFlavor.Mlx);

        TextModelPackage official = Find(_root, "hf-official");
        Assert.Equal(TextPackageFormat.SafetensorsShards, official.Format);
        Assert.Equal("Official", official.Quant);
        Assert.Equal(["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"], official.Files);
        Assert.Empty(official.Problems);
        // The fixture writes its DSpark draft (mtp) tensors, and serving does not run them yet.
        Assert.Equal("DSpark speculation is not wired into serving yet", official.Speculation.Reason);
        Assert.False(official.Speculation.Supported);

        TextModelPackage mlx = Find(_root, "hf-mlx");
        Assert.Equal("Mlx", mlx.Quant);
        Assert.Equal("MLX checkpoints are not served with DSpark speculation", mlx.Speculation.Reason);
    }

    [Fact]
    public void A_Shard_The_Index_Names_But_The_Directory_Lacks_Is_A_Problem()
    {
        string official = Directory.CreateDirectory(Path.Combine(_root, "hf-official")).FullName;
        TinyDeepSeekV41Checkpoint.Write(official);
        File.Delete(Path.Combine(official, "model-00002-of-00003.safetensors"));

        TextModelPackage package = Find(_root, "hf-official");

        Assert.Equal(["model-00001-of-00003.safetensors", "model-00003-of-00003.safetensors"], package.Files);
        Assert.Equal(["1 shard(s) named by the index are missing: model-00002-of-00003.safetensors"], package.Problems);
    }

    [Fact]
    public void Split_Parts_That_Share_A_Name_But_Not_A_Part_Count_Are_Two_Packages_With_Distinct_Ids()
    {
        string llm = Directory.CreateDirectory(Path.Combine(_root, "llm")).FullName;
        File.WriteAllBytes(Path.Combine(llm, "qwen-00001-of-00002.gguf"), [1]);
        File.WriteAllBytes(Path.Combine(llm, "qwen-00002-of-00002.gguf"), [1]);
        File.WriteAllBytes(Path.Combine(llm, "qwen-00001-of-00003.gguf"), [1, 2]);

        IReadOnlyList<TextModelPackage> packages = TextPackageDiscovery.Discover(_root).Packages;

        Assert.Equal(["llm/qwen-of-00002", "llm/qwen-of-00003"], packages.Select(p => p.Id));
        TextModelPackage two = packages[0];
        Assert.Equal(["qwen-00001-of-00002.gguf", "qwen-00002-of-00002.gguf"], two.Files);
        Assert.DoesNotContain(two.Problems, p => p.StartsWith("missing part", StringComparison.Ordinal));
        TextModelPackage three = packages[1];
        Assert.Equal(["qwen-00001-of-00003.gguf"], three.Files);
        Assert.Contains("missing part(s) 2, 3 of 3", three.Problems);
        Assert.All(packages, p => Assert.Contains("parts named 'qwen' disagree on the part count (2, 3); each count is listed as its own package", p.Problems));
    }

    [Fact]
    public void A_Checkpoint_Without_An_Index_Is_Read_For_Its_Draft_Head_From_The_Safetensors_Header()
    {
        WriteSingleFile(Path.Combine(_root, "with-draft"), "embed.weight", "mtp.0.norm.weight");
        WriteSingleFile(Path.Combine(_root, "without-draft"), "embed.weight", "layers.0.norm.weight");

        Assert.Equal("DSpark speculation is not wired into serving yet", Find(_root, "with-draft").Speculation.Reason);
        TextModelPackage plain = Find(_root, "without-draft");
        Assert.Equal("no DSpark draft head (mtp) in this checkpoint", plain.Speculation.Reason);
        Assert.Equal(["model.safetensors"], plain.Files);
        Assert.Empty(plain.Problems);
    }

    [Fact]
    public void A_Safetensors_Header_That_Cannot_Be_Read_Is_A_Problem_Of_Its_Package()
    {
        string corrupt = Directory.CreateDirectory(Path.Combine(_root, "corrupt")).FullName;
        File.WriteAllText(Path.Combine(corrupt, "config.json"), SingleFileConfig);
        File.WriteAllBytes(Path.Combine(corrupt, "model.safetensors"), [255, 255, 255, 255, 255, 255, 255, 255, 0]);

        TextModelPackage package = Find(_root, "corrupt");

        Assert.Contains(package.Problems, p => p.StartsWith("model.safetensors: safetensors header unreadable", StringComparison.Ordinal));
        Assert.Equal("no DSpark draft head (mtp) in this checkpoint", package.Speculation.Reason);
    }

    [Fact]
    public void Split_GGUF_Is_One_Package_And_Names_Its_Missing_Parts()
    {
        string llm = Directory.CreateDirectory(Path.Combine(_root, "llm")).FullName;
        for (int part = 1; part <= 3; part++)
            File.WriteAllBytes(Path.Combine(llm, $"qwen-{part:D5}-of-00003.gguf"), [1, 2, 3]);
        File.WriteAllBytes(Path.Combine(llm, "partial-00001-of-00002.gguf"), [1]);

        TextModelPackage split = Find(_root, "llm/qwen");
        Assert.Equal(TextPackageFormat.SplitGguf, split.Format);
        Assert.Equal(3, split.Files.Count);
        Assert.Equal("qwen-00001-of-00003.gguf", Path.GetFileName(split.EntryPath));
        Assert.Equal(9, split.TotalBytes);
        Assert.DoesNotContain(split.Problems, p => p.StartsWith("missing part", StringComparison.Ordinal));
        // The loader reads single files only, so a split is not loadable as it stands.
        Assert.Contains(split.Problems, p => p.Contains("merge the parts", StringComparison.Ordinal));

        TextModelPackage partial = Find(_root, "llm/partial");
        Assert.Equal(TextPackageFormat.SplitGguf, partial.Format);
        Assert.Contains("missing part(s) 2 of 2", partial.Problems);
    }

    [Fact]
    public void Lone_Model_Takes_The_Mmproj_Beside_It_As_Its_Vision_Sidecar()
    {
        string vision = Directory.CreateDirectory(Path.Combine(_root, "vision")).FullName;
        File.WriteAllBytes(Path.Combine(vision, "single.gguf"), [1, 2, 3, 4]);
        File.WriteAllBytes(Path.Combine(vision, "mmproj-single.gguf"), [9]);

        TextModelPackage single = Find(_root, "vision/single.gguf");
        Assert.Equal(TextPackageFormat.Gguf, single.Format);
        TextPackageSidecar sidecar = Assert.Single(single.Sidecars);
        Assert.Equal("vision", sidecar.Role);
        Assert.Equal("mmproj-single.gguf", Path.GetFileName(sidecar.Path));
        Assert.Equal("GGUF packages carry no DSpark draft head", single.Speculation.Reason);
        Assert.Empty(single.Problems);
    }

    [Fact]
    public void Several_Models_Sharing_A_Directory_Do_Not_Take_One_Mmproj()
    {
        string shared = Directory.CreateDirectory(Path.Combine(_root, "shared")).FullName;
        File.WriteAllBytes(Path.Combine(shared, "a.gguf"), [1]);
        File.WriteAllBytes(Path.Combine(shared, "b.gguf"), [1]);
        File.WriteAllBytes(Path.Combine(shared, "mmproj-shared.gguf"), [9]);

        TextModelPackage a = Find(_root, "shared/a.gguf");
        Assert.Empty(a.Sidecars);
        Assert.Contains(a.Problems, p => p.Contains("is not attached: 2 models share this directory", StringComparison.Ordinal));
        Assert.Empty(Find(_root, "shared/b.gguf").Sidecars);
    }

    [Fact]
    public void A_Missing_Root_Yields_No_Packages()
    {
        TextPackageScan scan = TextPackageDiscovery.Discover(Path.Combine(_root, "absent"));

        Assert.Empty(scan.Packages);
        Assert.Empty(scan.Problems);
    }

    [Fact]
    public void An_Unreadable_Directory_Is_Skipped_And_Named_While_The_Rest_Is_Listed()
    {
        if (!OperatingSystem.IsLinux()) return;
        File.WriteAllBytes(Path.Combine(_root, "model.gguf"), [1]);
        string locked = Directory.CreateDirectory(Path.Combine(_root, "locked")).FullName;
        File.WriteAllBytes(Path.Combine(locked, "inside.gguf"), [1]);
        Directory.CreateDirectory(Path.Combine(_root, "open"));
        File.WriteAllBytes(Path.Combine(_root, "open", "after.gguf"), [1]);
        _locked = locked;
        File.SetUnixFileMode(locked, UnixFileMode.None);
        // Root lists a mode-000 directory anyway, so under root there is nothing to skip.
        if (CanList(locked)) return;

        TextPackageScan scan = TextPackageDiscovery.Discover(_root);

        Assert.Equal(["model.gguf", "open/after.gguf"], scan.Packages.Select(p => p.Id));
        string problem = Assert.Single(scan.Problems);
        Assert.StartsWith("locked: skipped (", problem, StringComparison.Ordinal);
        Assert.DoesNotContain(_root, problem, StringComparison.Ordinal); // the problem names the exception type, never the server path
    }

    private static TextModelPackage Find(string root, string id) =>
        TextPackageDiscovery.Discover(root).Packages.Single(p => p.Id == id);

    /// <summary>Writes a checkpoint with no shard index: a config and one <c>model.safetensors</c> whose header names <paramref name="tensors"/> (two BF16 values each).</summary>
    private static void WriteSingleFile(string directory, params string[] tensors)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), SingleFileConfig);
        Dictionary<string, object> header = [];
        for (int i = 0; i < tensors.Length; i++)
            header[tensors[i]] = new Dictionary<string, object> { ["dtype"] = "BF16", ["shape"] = new long[] { 2 }, ["data_offsets"] = new long[] { 4L * i, 4L * i + 4 } };
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using FileStream stream = new(Path.Combine(directory, "model.safetensors"), FileMode.Create, FileAccess.Write);
        stream.Write(BitConverter.GetBytes((ulong)json.Length));
        stream.Write(json);
        stream.Write(new byte[4 * tensors.Length]);
    }

    private static bool CanList(string directory)
    {
        try
        {
            _ = Directory.GetFileSystemEntries(directory);
            return true;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }
}
