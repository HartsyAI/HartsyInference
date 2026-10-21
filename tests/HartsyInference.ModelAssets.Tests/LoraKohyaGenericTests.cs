using System.Text;
using System.Text.Json;
using HartsyInference.Core.Exceptions;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>The generic kohya arm: a <c>lora_unet_</c> file no specific layout claims (Krea 2) is detected as <see cref="LoraFormat.KohyaGeneric"/>, and its underscored module names resolve against the target model's own weight names — ComfyUI's rule, no per-family table. Detection of every other format is pinned by the neighbouring detector tests.</summary>
public sealed unsafe class LoraKohyaGenericTests : IDisposable
{
    private readonly string _tempDir;
    private readonly ITestOutputHelper _output;

    public LoraKohyaGenericTests(ITestOutputHelper output)
    {
        _output = output;
        _tempDir = Path.Combine(Path.GetTempPath(), $"lora_kohya_generic_tests_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    /// <summary>Native Krea 2 weight names, as the checkpoint stores them (norm scales and the modulation table are not <c>.weight</c> keys).</summary>
    private static readonly string[] _krea2Names =
    [
        "blocks.0.attn.wq.weight", "blocks.0.attn.wo.weight", "blocks.0.attn.gate.weight", "blocks.0.mlp.up.weight",
        "blocks.0.mod.lin", "blocks.0.prenorm.scale", "blocks.0.attn.qknorm.qnorm.scale",
        "blocks.27.mlp.down.weight",
        "txtfusion.layerwise_blocks.3.mlp.up.weight", "txtfusion.layerwise_blocks.3.attn.wk.weight",
        "txtfusion.refiner_blocks.1.attn.wq.weight", "txtfusion.refiner_blocks.1.mlp.gate.weight",
        "txtfusion.projector.weight",
        "first.weight", "first.bias", "tmlp.0.weight", "tmlp.0.bias", "tproj.1.weight",
        "txtmlp.0.scale", "txtmlp.1.weight", "last.linear.weight", "last.modulation.lin",
    ];

    private static SafeTensorDescriptor Desc(string name) => new()
    {
        Name = name,
        DType = DType.F16,
        Shape = new TensorShape(16, 2048),
        DataOffset = 0,
        ByteLength = 16 * 2048 * 2,
    };

    private static Dictionary<string, SafeTensorDescriptor> Descriptors(params string[] keys)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = [];
        foreach (string key in keys)
        {
            descriptors[key] = Desc(key);
        }
        return descriptors;
    }

    [Theory]
    [InlineData("lora_unet_blocks_0_attn_wq")]
    [InlineData("lora_unet_txtfusion_layerwise_blocks_3_mlp_up")]
    [InlineData("lora_unet_txtfusion_refiner_blocks_1_attn_wq")]
    [InlineData("lora_unet_first")]
    [InlineData("lora_unet_tmlp_0")]
    public void KreaKeyShape_IsKohyaGeneric(string root)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            $"{root}.lora_down.weight",
            $"{root}.lora_up.weight",
            $"{root}.alpha");
        Assert.Equal(LoraFormat.KohyaGeneric, LoraFormatDetector.Detect(descriptors));
    }

    /// <summary>A scalar alone is not evidence of an adapter, the same rule the bare-root arm holds itself to.</summary>
    [Fact]
    public void AlphaOnly_IsStillUnknown()
    {
        Assert.Equal(LoraFormat.Unknown, LoraFormatDetector.Detect(Descriptors("lora_unet_first.alpha")));
    }

    /// <summary>The generic arm is last: any file a specific layout claims keeps its format even when it also carries an unclaimed <c>lora_unet_</c> module.</summary>
    [Theory]
    [InlineData("lora_unet_double_blocks_0_img_attn_qkv", LoraFormat.KohyaFlux)]
    [InlineData("lora_unet_blocks_0_self_attn_q", LoraFormat.KohyaWan)]
    [InlineData("lora_unet_down_blocks_0_attentions_0_proj_in", LoraFormat.KohyaSd15)]
    [InlineData("lora_unet_input_blocks_1_1_proj_in", LoraFormat.KohyaSd15)]
    public void SpecificLayout_WinsOverGeneric(string claimedRoot, LoraFormat expected)
    {
        Dictionary<string, SafeTensorDescriptor> descriptors = Descriptors(
            $"{claimedRoot}.lora_down.weight",
            $"{claimedRoot}.lora_up.weight",
            "lora_unet_first.lora_down.weight",
            "lora_unet_first.lora_up.weight");
        Assert.Equal(expected, LoraFormatDetector.Detect(descriptors));
    }

    [Fact]
    public void Index_ResolvesKreaModules_IncludingTextFusionStacks()
    {
        KohyaUnderscoreIndex index = KohyaUnderscoreIndex.Build(_krea2Names);

        string[] modules =
        [
            "blocks_0_attn_wq", "blocks_0_attn_gate", "blocks_27_mlp_down",
            "txtfusion_layerwise_blocks_3_mlp_up", "txtfusion_layerwise_blocks_3_attn_wk",
            "txtfusion_refiner_blocks_1_attn_wq", "txtfusion_refiner_blocks_1_mlp_gate",
            "txtfusion_projector", "first", "tmlp_0", "tproj_1", "txtmlp_1", "last_linear",
        ];
        foreach (string module in modules)
        {
            Assert.True(index.TryResolve(module, out string key), $"'{module}' did not resolve");
            Assert.Equal(module, key[..^".weight".Length].Replace('.', '_'));
            Assert.Contains(key, _krea2Names);
        }
        Assert.True(index.TryResolve("txtfusion_layerwise_blocks_3_mlp_up", out string layerwise));
        Assert.Equal("txtfusion.layerwise_blocks.3.mlp.up.weight", layerwise);
        Assert.True(index.TryResolve("txtfusion_refiner_blocks_1_attn_wq", out string refiner));
        Assert.Equal("txtfusion.refiner_blocks.1.attn.wq.weight", refiner);
    }

    /// <summary>Only <c>.weight</c> keys name modules: a bias, a norm scale or the modulation table is never a LoRA target.</summary>
    [Theory]
    [InlineData("first_bias")]
    [InlineData("blocks_0_mod_lin")]
    [InlineData("blocks_0_prenorm_scale")]
    [InlineData("blocks_99_attn_wq")]
    public void Index_DoesNotResolveNonWeightsOrMissingModules(string module)
    {
        Assert.False(KohyaUnderscoreIndex.Build(_krea2Names).TryResolve(module, out _));
    }

    [Fact]
    public void Index_AmbiguousModule_RefusesNamingBothWeights()
    {
        KohyaUnderscoreIndex index = KohyaUnderscoreIndex.Build(["a.b_c.weight", "a_b.c.weight", "a.d.weight"]);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(() => index.TryResolve("a_b_c", out _));
        Assert.Contains("'a.b_c.weight'", ex.Message);
        Assert.Contains("'a_b.c.weight'", ex.Message);

        // A collision no LoRA module lands on does not block the ones that do not collide.
        Assert.True(index.TryResolve("a_d", out string key));
        Assert.Equal("a.d.weight", key);
    }

    /// <summary>The loader renames checkpoint keys, but the LoRA spells the checkpoint's names: the index maps native → dictionary key, and a dictionary key's own spelling is not a module name.</summary>
    [Fact]
    public void Index_WithNativeNames_ResolvesToTheRenamedDictionaryKey()
    {
        Dictionary<string, string> nativeToTarget = RenamedKrea2Map();
        KohyaUnderscoreIndex index = KohyaUnderscoreIndex.Build(nativeToTarget);

        Assert.True(index.TryResolve("blocks_0_attn_wq", out string wq));
        Assert.Equal("transformer_blocks.0.attn.to_q.weight", wq);
        Assert.True(index.TryResolve("txtfusion_layerwise_blocks_3_mlp_up", out string up));
        Assert.Equal("text_fusion.layerwise_blocks.3.ff.up.weight", up);
        Assert.False(index.TryResolve("transformer_blocks_0_attn_to_q", out _));
    }

    [Fact]
    public void Load_KreaStyleFile_ParsesUnresolvedLayers()
    {
        string path = WriteLora("krea_style", "blocks_0_attn_wq", "txtfusion_refiner_blocks_1_attn_wq", "first");

        using LoraFile file = LoraFile.Load(path);

        Assert.Equal(LoraFormat.KohyaGeneric, file.Format);
        Assert.Equal(3, file.Layers.Count);
        Assert.All(file.Layers, l => Assert.Equal(LoraTarget.Transformer, l.Target));
        LoraLayer layer = file.Layers.Single(l => l.KohyaModuleName == "blocks_0_attn_wq");
        Assert.Equal("lora_unet_blocks_0_attn_wq", layer.TargetKey);
        Assert.Equal(2, layer.Rank);
        Assert.Equal(2.0f, layer.Alpha);
    }

    /// <summary>End to end through <see cref="LoraStack"/> against the dictionary the Krea 2 recipe builds: renamed keys plus the native-name map. Resolved weights merge, an unmatched module neither throws nor touches anything else.</summary>
    [Fact]
    public void Stack_MergesIntoRenamedDictionary_AndSkipsUnmatchedModules()
    {
        string path = WriteLora("krea_merge",
            "blocks_0_attn_wq", "txtfusion_layerwise_blocks_3_mlp_up", "txtfusion_refiner_blocks_1_attn_wq", "first", "tmlp_0",
            "blocks_99_attn_wq");

        Dictionary<string, string> nativeToTarget = RenamedKrea2Map();
        Dictionary<string, Tensor> weights = [];
        foreach (string target in nativeToTarget.Values)
        {
            weights[target] = Ones(4, 4);
        }
        Tensor untouched = weights["transformer_blocks.0.attn.to_k.weight"] = Ones(4, 4);

        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path);
        int merged = stack.ApplyTo(weights, LoraTarget.Transformer, backend, nativeToTarget);

        Assert.Equal(5, merged);
        // alpha 2 / rank 2 = scale 1; ΔW = up·down = 2 · 0.5 · 0.5 = 0.5 per element.
        string[] mergedKeys =
        [
            "transformer_blocks.0.attn.to_q.weight", "text_fusion.layerwise_blocks.3.ff.up.weight",
            "text_fusion.refiner_blocks.1.attn.to_q.weight", "img_in.weight", "time_embed.linear_1.weight",
        ];
        foreach (string key in mergedKeys)
        {
            AssertAll(weights[key], 1.5f, key);
        }
        Assert.Same(untouched, weights["transformer_blocks.0.attn.to_k.weight"]);
        AssertAll(weights["transformer_blocks.0.attn.to_k.weight"], 1.0f, "to_k");
        Assert.DoesNotContain(weights.Keys, k => k.Contains("99", StringComparison.Ordinal));
    }

    /// <summary>With no native-name map the dictionary keys are taken as the model's own names (a family whose loader does not rename).</summary>
    [Fact]
    public void Stack_WithoutNativeNames_ResolvesAgainstDictionaryKeys()
    {
        string path = WriteLora("native_keys", "blocks_0_attn_wq", "first");
        Dictionary<string, Tensor> weights = new()
        {
            ["blocks.0.attn.wq.weight"] = Ones(4, 4),
            ["first.weight"] = Ones(4, 4),
        };

        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path);

        Assert.Equal(2, stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        AssertAll(weights["blocks.0.attn.wq.weight"], 1.5f, "wq");
    }

    [Fact]
    public void Stack_AmbiguousWeightNames_RefuseNamingBoth()
    {
        string path = WriteLora("ambiguous", "a_b_c");
        Dictionary<string, Tensor> weights = new()
        {
            ["a.b_c.weight"] = Ones(4, 4),
            ["a_b.c.weight"] = Ones(4, 4),
        };

        using CpuBackend backend = new();
        using LoraStack stack = new();
        stack.AddFromPath(path);

        HartsyInferenceException ex = Assert.Throws<HartsyInferenceException>(
            () => stack.ApplyTo(weights, LoraTarget.Transformer, backend));
        Assert.Contains("'a.b_c.weight'", ex.Message);
        Assert.Contains("'a_b.c.weight'", ex.Message);
        AssertAll(weights["a.b_c.weight"], 1.0f, "a.b_c");
        AssertAll(weights["a_b.c.weight"], 1.0f, "a_b.c");
    }

    /// <summary>The real Krea 2 LoRA against the real Krea 2 checkpoint, headers only: the file is detected, and every module it adapts names a checkpoint weight — directly, and through the converter's rename into the dictionary key the recipe merges into.</summary>
    [Fact]
    [Trait("Category", "Integration")]
    [Trait("Category", "RealWeights")]
    public void RealKreaLora_EveryModuleResolvesToACheckpointWeight()
    {
        if (!RealWeightGate.Require(_output.WriteLine, TestPaths.Krea2.GpkLora, TestPaths.Krea2.TurboCheckpoint)) return;

        SafeTensorHeader lora = SafeTensorHeaderReader.Read(TestPaths.Krea2.GpkLora);
        SafeTensorHeader checkpoint = SafeTensorHeaderReader.Read(TestPaths.Krea2.TurboCheckpoint);
        Assert.Equal(LoraFormat.KohyaGeneric, LoraFormatDetector.Detect(lora.Tensors));

        HashSet<string> modules = [];
        foreach (string key in lora.Tensors.Keys)
        {
            Assert.True(LoraRoleSuffix.TryStrip(key, out string root, out LoraRole _), key);
            Assert.StartsWith("lora_unet_", root, StringComparison.Ordinal);
            modules.Add(root["lora_unet_".Length..]);
        }
        Assert.Equal(264, modules.Count);

        List<string> weightKeys = checkpoint.Tensors.Keys.ToList();
        KohyaUnderscoreIndex direct = KohyaUnderscoreIndex.Build(weightKeys);
        Dictionary<string, string> nativeToTarget = weightKeys.ToDictionary(
            k => k, k => Krea2CheckpointConverter.RemapTransformerKey(k));
        KohyaUnderscoreIndex renamed = KohyaUnderscoreIndex.Build(nativeToTarget);

        HashSet<string> targets = [];
        foreach (string module in modules)
        {
            Assert.True(direct.TryResolve(module, out string native), $"'{module}' resolves to no checkpoint weight");
            Assert.Contains(native, checkpoint.Tensors.Keys);
            Assert.True(renamed.TryResolve(module, out string target));
            Assert.Equal(nativeToTarget[native], target);
            Assert.True(targets.Add(target), $"'{target}' was claimed by two modules");
        }
        _output.WriteLine($"{modules.Count} LoRA modules resolved against {weightKeys.Count} checkpoint tensors.");
    }

    /// <summary>The native Krea 2 names with the key each gets in the dictionary the recipe builds.</summary>
    private static Dictionary<string, string> RenamedKrea2Map()
    {
        Dictionary<string, string> map = [];
        foreach (string name in _krea2Names)
        {
            map[name] = Krea2CheckpointConverter.RemapTransformerKey(name);
        }
        map["blocks.0.attn.wk.weight"] = Krea2CheckpointConverter.RemapTransformerKey("blocks.0.attn.wk.weight");
        return map;
    }

    private static Tensor Ones(int rows, int cols)
    {
        Tensor tensor = new(new TensorShape(rows, cols), DType.F32);
        new Span<float>((float*)tensor.DataPointer, rows * cols).Fill(1.0f);
        return tensor;
    }

    private static void AssertAll(Tensor tensor, float expected, string name)
    {
        float* p = (float*)tensor.DataPointer;
        for (int i = 0; i < tensor.ElementCount; i++)
        {
            Assert.True(MathF.Abs(p[i] - expected) < 1e-5f, $"{name}[{i}] = {p[i]}, expected {expected}");
        }
    }

    /// <summary>A rank-2 LoRA (down [2,4] of 0.5, up [4,2] of 0.5, alpha 2) for each module.</summary>
    private string WriteLora(string name, params string[] modules)
    {
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors = [];
        foreach (string module in modules)
        {
            tensors[$"lora_unet_{module}.lora_down.weight"] = (DType.F32, [2, 4], Enumerable.Repeat(0.5f, 8).ToArray());
            tensors[$"lora_unet_{module}.lora_up.weight"] = (DType.F32, [4, 2], Enumerable.Repeat(0.5f, 8).ToArray());
            tensors[$"lora_unet_{module}.alpha"] = (DType.F32, [1], [2.0f]);
        }
        return CreateSafeTensorsFile(_tempDir, name, tensors);
    }

    private static string CreateSafeTensorsFile(string dir, string name,
        Dictionary<string, (DType dtype, long[] shape, float[] data)> tensors)
    {
        using MemoryStream dataStream = new();
        Dictionary<string, object> headerDict = [];
        foreach (KeyValuePair<string, (DType dtype, long[] shape, float[] data)> kvp in tensors)
        {
            long start = dataStream.Position;
            foreach (float val in kvp.Value.data)
            {
                byte[] bytes = BitConverter.GetBytes(val);
                dataStream.Write(bytes, 0, bytes.Length);
            }
            headerDict[kvp.Key] = new Dictionary<string, object>
            {
                ["dtype"] = kvp.Value.dtype.Name,
                ["shape"] = kvp.Value.shape,
                ["data_offsets"] = new long[] { start, dataStream.Position },
            };
        }

        byte[] headerBytes = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(headerDict));
        string filePath = Path.Combine(dir, $"{name}.safetensors");
        using FileStream fs = new(filePath, FileMode.Create, FileAccess.Write);
        using BinaryWriter writer = new(fs);
        writer.Write((long)headerBytes.Length);
        writer.Write(headerBytes);
        writer.Write(dataStream.ToArray());
        return filePath;
    }

    public void Dispose()
    {
        try
        {
            if (Directory.Exists(_tempDir))
            {
                Directory.Delete(_tempDir, true);
            }
        }
        catch
        {
            // best-effort cleanup
        }
    }
}
