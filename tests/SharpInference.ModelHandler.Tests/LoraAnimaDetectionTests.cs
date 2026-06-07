using SharpInference.Core.Tensors;
using SharpInference.ModelHandler.Lora;
using SharpInference.ModelHandler.SafeTensors;
using Xunit;

namespace SharpInference.ModelHandler.Tests;

/// <summary>Verifies that Anima / Cosmos-Predict2 diffusers-PEFT LoRAs are recognized and route through
/// the existing (architecture-agnostic) diffusers passthrough mapper — no Anima-specific mapper needed,
/// because the mapper strips <c>transformer.</c> and the remaining body is already the canonical
/// <c>AnimaTransformer</c> key (<c>blocks.{i}.self_attn.q_proj.weight</c>).</summary>
public sealed class LoraAnimaDetectionTests
{
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
        Dictionary<string, SafeTensorDescriptor> d = new();
        foreach (string k in keys) d[k] = Desc(k);
        return d;
    }

    [Fact]
    public void AnimaCosmosKeys_DetectedAsDiffusersPeft()
    {
        // Cosmos/Anima single-stream DiT names blocks `transformer.blocks.{i}.*` with PEFT lora_A/lora_B.
        Dictionary<string, SafeTensorDescriptor> d = Descriptors(
            "transformer.blocks.0.self_attn.q_proj.lora_A.weight",
            "transformer.blocks.0.self_attn.q_proj.lora_B.weight",
            "transformer.blocks.0.self_attn.k_proj.lora_A.weight",
            "transformer.blocks.0.self_attn.k_proj.lora_B.weight",
            "transformer.blocks.5.mlp.fc1.lora_A.weight",
            "transformer.blocks.5.mlp.fc1.lora_B.weight");
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(d));
    }

    [Fact]
    public void FluxDiffusersKeys_StillDetected_NoRegression()
    {
        Dictionary<string, SafeTensorDescriptor> d = Descriptors(
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
            "transformer.single_transformer_blocks.3.attn.to_v.lora_A.weight",
            "transformer.single_transformer_blocks.3.attn.to_v.lora_B.weight");
        Assert.Equal(LoraFormat.DiffusersFlux, LoraFormatDetector.Detect(d));
    }

    [Fact]
    public void UnrelatedKeys_AreUnknown()
    {
        Dictionary<string, SafeTensorDescriptor> d = Descriptors(
            "some.random.tensor.weight",
            "another.weight");
        Assert.Equal(LoraFormat.Unknown, LoraFormatDetector.Detect(d));
    }
}
