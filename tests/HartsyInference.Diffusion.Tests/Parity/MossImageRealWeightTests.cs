using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Engine.Features;
using HartsyInference.Engine.Recipes.Image;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Checkpoints;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Tests.Common;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.Diffusion.Tests.Parity;

/// <summary>MOSS-Image-Base against OpenMOSS's own ComfyUI module on the real weights, one stage at a time, so a drift
/// names its stage: the Qwen3.5 prompt ids, the encoder's final hidden states, and the transformer's velocity for the
/// conditional branch and the unconditional branch with block 22 skipped, at the noisy, middle and clean ends of the
/// schedule. The reference is <c>dump_moss_image_reference.py real</c> (see <see cref="TestPaths.MossImage"/>); the
/// encoder and transformer run on every GPU backend from the reference's own inputs, so their errors do not compound.</summary>
[Trait("Category", "Integration")]
[Trait("Category", "RealWeights")]
public sealed class MossImageRealWeightTests(ITestOutputHelper output)
{
    [Fact]
    public void Tokenizer_ReproducesTheReferenceIds()
    {
        if (!RealWeightGate.Require(output.WriteLine, TestPaths.MossImage.Base, TestPaths.MossImage.Reference)) return;
        using SafeTensorsLoader reference = OpenReference();
        MossImagePromptTokenizer tokenizer = new(LoadTokenizer());
        Assert.Equal(Ids(reference, "prompt.ids"), tokenizer.Tokenize(Prompt(reference)).Tokens);
        Assert.Equal(Ids(reference, "negative.ids"), tokenizer.Tokenize("").Tokens);
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void TextEncoder_ReproducesTheReferenceHiddenStates(string kind)
    {
        if (!RealWeightGate.Require(output.WriteLine, TestPaths.MossImage.Base, TestPaths.MossImage.Reference)) return;
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        using SafeTensorsLoader reference = OpenReference();
        using Qwen35TextEncoder encoder = Qwen35TextEncoder.Load(Path.Combine(TestPaths.MossImage.Base, MossImageRecipe.TextEncoderFolder),
            MossImageConfig.MaxTextTokens, "MOSS-Image test");
        foreach (string label in new[] { "prompt", "negative" })
        {
            using Tensor hidden = encoder.Encode(backend, Ids(reference, $"{label}.ids"));
            (double relL2, double cosine) = TensorAssert.Agrees(hidden, reference.GetTensor($"{label}.hidden"), encoder.HiddenSize,
                maxRelL2: 2e-2, minRowCosine: 0.995, because: $"{label} hidden states on {kind}");
            output.WriteLine($"{kind} {label}: rel L2 {relL2:E3}, worst token cosine {cosine:F6}");
        }
    }

    [Trait("Category", "GpuIntegration")]
    [Theory]
    [MemberData(nameof(BackendGate.GpuKinds), MemberType = typeof(BackendGate))]
    public void Transformer_ReproducesTheReferenceVelocities(string kind)
    {
        if (!RealWeightGate.Require(output.WriteLine, TestPaths.MossImage.Base, TestPaths.MossImage.Reference)) return;
        if (!BackendGate.TryOpen(kind, output.WriteLine, out IBackend? gpu)) return;
        using IBackend backend = gpu!;
        using SafeTensorsLoader reference = OpenReference();
        IReadOnlyList<string> files = HfCheckpointDirectory.CheckpointFiles(Path.Combine(TestPaths.MossImage.Base, MossImageRecipe.TransformerFolder));
        using CheckpointSource source = CheckpointSource.OpenShards(files);
        Dictionary<string, Tensor> converted = MossImageCheckpointConverter.ConvertTransformer(source.Weights);
        using MossImageTransformer transformer = new(MossImageConfig.FromWeights(converted));
        transformer.LoadWeights(converted);
        backend.PreloadWeights(transformer.EnumerateWeights());

        Tensor latent = reference.GetTensor("inputs.latent");
        int grid = (int)Math.Sqrt(latent.Shape[1]);
        using MossImageConditioning cond = transformer.PrepareConditioning(backend, reference.GetTensor("prompt.hidden"), grid, grid);
        using MossImageConditioning uncond = transformer.PrepareConditioning(backend, reference.GetTensor("negative.hidden"), grid, grid);
        ReadOnlySpan<float> timesteps = reference.GetTensor("inputs.timesteps").AsReadOnlySpan<float>();
        for (int i = 0; i < timesteps.Length; i++)
        {
            using Tensor condVelocity = transformer.Forward(backend, latent, timesteps[i], cond);
            using Tensor uncondVelocity = transformer.Forward(backend, latent, timesteps[i], uncond, [MossImageConfig.ReferenceUncondSkipLayer]);
            Report(kind, $"cond t={timesteps[i]}", TensorAssert.Agrees(condVelocity, reference.GetTensor($"cond.velocity.{i}"),
                transformer.Config.LatentChannels, maxRelL2: 3e-2, minRowCosine: 0.99, because: $"cond velocity {i} on {kind}"));
            Report(kind, $"uncond t={timesteps[i]}", TensorAssert.Agrees(uncondVelocity, reference.GetTensor($"uncond.velocity.{i}"),
                transformer.Config.LatentChannels, maxRelL2: 3e-2, minRowCosine: 0.99, because: $"uncond velocity {i} on {kind}"));
        }
        backend.FreeWeights(transformer.EnumerateWeights());
    }

    private void Report(string kind, string label, (double RelL2, double MinRowCosine) result) =>
        output.WriteLine($"{kind} {label}: rel L2 {result.RelL2:E3}, worst token cosine {result.MinRowCosine:F6}");

    private static SafeTensorsLoader OpenReference()
    {
        SafeTensorsLoader loader = new();
        loader.Load(TestPaths.MossImage.Reference);
        return loader;
    }

    private static string Prompt(SafeTensorsLoader reference) =>
        reference.Metadata?.GetValueOrDefault("prompt") ?? throw new InvalidDataException("The reference carries no prompt metadata.");

    private static int[] Ids(SafeTensorsLoader reference, string key) =>
        [.. reference.GetTensor(key).AsReadOnlySpan<float>().ToArray().Select(id => (int)id)];

    private static ILlmTokenizer LoadTokenizer()
    {
        using FileStream json = File.OpenRead(Path.Combine(TestPaths.MossImage.Base, MossImageRecipe.TokenizerFolder, HfCheckpointDirectory.TokenizerFileName));
        return HfTokenizerJson.LoadByteLevelBpe(json);
    }
}
