using System.Text.Json;
using HartsyInference.API;
using HartsyInference.API.Endpoints;
using HartsyInference.Core.Configuration;
using HartsyInference.ModelAssets.Quant;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.HttpResults;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>The speculation capability the model list and the <c>/admin/packages</c> route report: off for every text format, with the reason that format gives. The
/// route lists only the models root and the directories under it, so these tests point the models root at a fixture through the <c>paths.modelsRoot</c> setting.</summary>
[Collection("ModelsRootKnob")]
public sealed class PackageListingTests : IDisposable
{
    private readonly string _modelsRoot = Path.Combine(Path.GetTempPath(), "hartsy-packages-" + Guid.NewGuid().ToString("N"));

    public PackageListingTests()
    {
        // Load the settings file first: its first load would overwrite the override set below.
        _ = EngineKnobs.ModelsRoot.Value;
        Directory.CreateDirectory(_modelsRoot);
        KnobStore.Set(EngineKnobs.ModelsRoot, _modelsRoot);
    }

    public void Dispose()
    {
        KnobStore.Clear(EngineKnobs.ModelsRoot);
        if (Directory.Exists(_modelsRoot)) Directory.Delete(_modelsRoot, recursive: true);
    }

    [Fact]
    public void Gguf_Only_Model_Reports_Speculation_Off_With_The_GGUF_Reason()
    {
        CapabilitiesDto caps = CompatEndpoints.TextCapabilities([null]);

        Assert.False(caps.Speculation);
        Assert.Equal("GGUF packages carry no DSpark draft head", caps.SpeculationReason);
    }

    [Fact]
    public void The_Default_Scan_Is_The_LLM_Folder_Whatever_Its_Case()
    {
        // The engine's text folder is LLM. A lowercase default found nothing on Linux, so the route listed no packages there.
        string textFolder = Path.Combine(_modelsRoot, "LLM");
        Directory.CreateDirectory(textFolder);

        Assert.Equal(textFolder, PackageListing.DefaultRoot(_modelsRoot));
    }

    [Fact]
    public void Mlx_Only_Model_Reports_The_MLX_Reason()
    {
        CapabilitiesDto caps = CompatEndpoints.TextCapabilities([QuantFlavor.Mlx]);

        Assert.False(caps.Speculation);
        Assert.Equal("MLX checkpoints are not served with DSpark speculation", caps.SpeculationReason);
    }

    [Fact]
    public void Safetensors_Variant_Reports_The_Wiring_Reason_Even_Beside_An_MLX_Variant()
    {
        Assert.Equal("DSpark speculation is not wired into serving yet", CompatEndpoints.TextCapabilities([QuantFlavor.Official]).SpeculationReason);
        Assert.Equal("DSpark speculation is not wired into serving yet", CompatEndpoints.TextCapabilities([QuantFlavor.Mlx, QuantFlavor.Official]).SpeculationReason);
    }

    [Fact]
    public void Model_With_No_Variant_Says_So()
    {
        CapabilitiesDto caps = CompatEndpoints.TextCapabilities([]);

        Assert.False(caps.Speculation);
        Assert.Equal("no variant ships a DSpark draft head", caps.SpeculationReason);
    }

    [Fact]
    public void Packages_Route_Refuses_A_Missing_Root_With_404_Naming_Only_The_Value_Given()
    {
        IResult result = PackageListing.Respond("absent");

        Assert.Equal(StatusCodes.Status404NotFound, Assert.IsAssignableFrom<IStatusCodeHttpResult>(result).StatusCode);
        Assert.Equal("'absent' is not a directory.", ErrorMessage(result));
    }

    [Fact]
    public void Packages_Route_Refuses_A_Root_Outside_The_Models_Root_With_400()
    {
        // A sibling whose name only starts with the models root's name is outside it too.
        string sibling = Directory.CreateDirectory(_modelsRoot + "-sibling").FullName;
        try
        {
            foreach (string root in new[] { "..", "../elsewhere", sibling, Path.GetTempPath() })
            {
                IResult result = PackageListing.Respond(root);

                Assert.Equal(StatusCodes.Status400BadRequest, Assert.IsAssignableFrom<IStatusCodeHttpResult>(result).StatusCode);
                Assert.Equal($"'{root}' is outside the models root; only the models root and the directories under it can be listed.", ErrorMessage(result));
            }
        }
        finally
        {
            Directory.Delete(sibling);
        }
    }

    [Fact]
    public void Packages_Route_Lists_The_Packages_Under_The_Given_Root()
    {
        string llm = Directory.CreateDirectory(Path.Combine(_modelsRoot, "LLM")).FullName;
        File.WriteAllBytes(Path.Combine(llm, "model.gguf"), [1, 2, 3]);

        // An absolute root under the models root, the models root itself, and a relative root taken under it.
        foreach ((string root, string id) in new[] { (llm, "model.gguf"), (_modelsRoot, "LLM/model.gguf"), ("LLM", "model.gguf") })
        {
            IResult result = PackageListing.Respond(root);

            PackageListResponse body = Assert.IsType<PackageListResponse>(((IValueHttpResult)result).Value);
            PackageDto package = Assert.Single(body.Packages);
            Assert.Equal(id, package.Id);
            Assert.Equal("gguf", package.Format);
            Assert.Equal(3, package.TotalBytes);
            Assert.False(package.Capabilities.Speculation);
            Assert.Empty(body.Problems);
        }
        // Root is reported relative to the models root, so no server path leaves through the admin route.
        Assert.Equal("LLM", Assert.IsType<PackageListResponse>(((IValueHttpResult)PackageListing.Respond("LLM")).Value).Root);
    }

    [Fact]
    public void No_Path_In_The_Listing_Is_A_Server_Path()
    {
        // A GGUF model with its mmproj sidecar, under the default folder: every path the listing reports is relative to the models root.
        string llm = Directory.CreateDirectory(Path.Combine(_modelsRoot, "LLM")).FullName;
        File.WriteAllBytes(Path.Combine(llm, "model.gguf"), [1, 2, 3]);
        File.WriteAllBytes(Path.Combine(llm, "mmproj-model.gguf"), [4]);

        IResult result = PackageListing.Respond(null);

        PackageListResponse body = Assert.IsType<PackageListResponse>(((IValueHttpResult)result).Value);
        PackageDto package = Assert.Single(body.Packages);
        Assert.Equal(Path.Combine("LLM", "model.gguf"), package.Path);
        Assert.Equal(Path.Combine("LLM", "mmproj-model.gguf"), Assert.Single(package.Sidecars).Path);
        string escapedRoot = JsonSerializer.Serialize(_modelsRoot).Trim('"');
        Assert.DoesNotContain(escapedRoot, JsonSerializer.Serialize(body), StringComparison.Ordinal);
    }

    [Fact]
    public void The_Same_Checkpoint_Gives_The_Same_Reason_On_The_Package_Listing_And_The_Model_List()
    {
        // What discovery reads from disk against what the catalog knows of the same package: its flavor, null for GGUF.
        WriteCheckpoint(Path.Combine(_modelsRoot, "official"), "{\"model_type\":\"deepseek_v4\",\"quantization_config\":{\"quant_method\":\"fp8\"}}");
        WriteCheckpoint(Path.Combine(_modelsRoot, "mlx"), "{\"model_type\":\"deepseek_v4\",\"quantization\":{\"mode\":\"affine\",\"bits\":4,\"group_size\":64}}");
        File.WriteAllBytes(Path.Combine(_modelsRoot, "model.gguf"), [1]);

        PackageListResponse body = Assert.IsType<PackageListResponse>(((IValueHttpResult)PackageListing.Respond(_modelsRoot)).Value);

        foreach ((string id, QuantFlavor? flavor) in new (string, QuantFlavor?)[] { ("official", QuantFlavor.Official), ("mlx", QuantFlavor.Mlx), ("model.gguf", null) })
        {
            string listed = body.Packages.Single(p => p.Id == id).Capabilities.SpeculationReason;
            Assert.Equal(CompatEndpoints.TextCapabilities([flavor]).SpeculationReason, listed);
        }
    }

    /// <summary>A checkpoint directory with <paramref name="config"/>, an index that names a draft (<c>mtp</c>) tensor, and the one shard it names.</summary>
    private static void WriteCheckpoint(string directory, string config)
    {
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), config);
        File.WriteAllText(Path.Combine(directory, "model.safetensors.index.json"),
            "{\"weight_map\":{\"embed.weight\":\"model-00001-of-00001.safetensors\",\"mtp.0.norm.weight\":\"model-00001-of-00001.safetensors\"}}");
        File.WriteAllBytes(Path.Combine(directory, "model-00001-of-00001.safetensors"), [0]);
    }

    private static string ErrorMessage(IResult result) => Assert.IsType<OpenAiError>(((IValueHttpResult)result).Value).Error.Message;
}
