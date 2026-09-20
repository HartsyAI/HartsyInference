using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Runs the Qwen-Image 2.1 denoiser on synthetic weights, at a size that fits a CPU unit test. Shapes and
/// the SwiGLU half order are the two things here that would otherwise only surface as a wrong image after a
/// multi-minute GPU load.</summary>
public sealed class QwenImage21ForwardTests
{
    private const int Hidden = 64;
    private const int Heads = 2;
    private const int HeadDim = 32;
    private const int Depth = 2;
    private const int Channels = 8;
    private const int ContextDim = 16;
    private const int MlpRatio = 2;

    private static QwenImage21Config TinyConfig => new()
    {
        HiddenSize = Hidden,
        NumHeads = Heads,
        HeadDim = HeadDim,
        Depth = Depth,
        InChannels = Channels,
        OutChannels = Channels,
        ContextDim = ContextDim,
        MlpRatio = MlpRatio,
        AxesDim = [8, 12, 12],
    };

    /// <summary>The fused <c>gate_up</c>'s FIRST half is the gate. ComfyUI's <c>_swiglu_eager</c> chunks
    /// <c>(gate, up)</c> in that order and its LoRA key map addresses <c>gate_layer</c> at row 0, so a swap here
    /// produces <c>silu(up)·gate</c> — a finite, plausible number, not an error. Driven with gate = 4 and up = 1,
    /// where the two orders differ by more than a factor of 1.3.</summary>
    [Fact]
    public void TheFirstHalfOfGateUpIsTheGate()
    {
        const int mlpDim = Hidden * MlpRatio;
        using CpuBackend backend = new CpuBackend();
        QwenImage21Block block = new QwenImage21Block(Hidden, Heads, HeadDim, mlpDim);

        // Input is all ones, so each output row equals that row's weight sum. Gate rows sum to 4, up rows to 1.
        Tensor gateUp = new Tensor(new TensorShape(2 * mlpDim, Hidden), DType.F32);
        Span<float> gu = gateUp.AsSpan<float>();
        for (int row = 0; row < 2 * mlpDim; row++)
        {
            float perElement = (row < mlpDim ? 4.0f : 1.0f) / Hidden;
            for (int col = 0; col < Hidden; col++) gu[row * Hidden + col] = perElement;
        }
        // Down projection sums its input, so one output element reveals the activation directly.
        Tensor outW = new Tensor(new TensorShape(Hidden, mlpDim), DType.F32);
        outW.AsSpan<float>().Fill(1.0f / mlpDim);

        Dictionary<string, Tensor> weights = new()
        {
            ["b.attn.to_q.weight"] = Zeros(Hidden, Hidden),
            ["b.attn.to_k.weight"] = Zeros(Hidden, Hidden),
            ["b.attn.to_v.weight"] = Zeros(Hidden, Hidden),
            ["b.attn.to_out.0.weight"] = Zeros(Hidden, Hidden),
            ["b.attn.norm_q.weight"] = Ones(HeadDim),
            ["b.attn.norm_k.weight"] = Ones(HeadDim),
            ["b.img_mlp.gate_up.weight"] = gateUp,
            ["b.img_mlp.out.weight"] = outW,
        };
        block.LoadWeights(weights, "b");

        Tensor input = new Tensor(new TensorShape(1, 1, Hidden), DType.F32);
        input.AsSpan<float>().Fill(1.0f);
        using Tensor result = block.ForwardMlp(backend, input, seq: 1, act: DType.F32);
        input.Dispose();

        float silu4 = 4.0f / (1.0f + MathF.Exp(-4.0f));
        float silu1 = 1.0f / (1.0f + MathF.Exp(-1.0f));
        float gateFirst = silu4 * 1.0f;      // ~3.928 — correct
        float upFirst = silu1 * 4.0f;        // ~2.924 — the swap
        Assert.Equal(gateFirst, result.AsReadOnlySpan<float>()[0], 3);
        Assert.NotEqual(upFirst, result.AsReadOnlySpan<float>()[0], 3);
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    /// <summary>A whole forward on synthetic weights: text prefix through every block, then image rows against the
    /// cached K/V. Asserts the contract the pipeline depends on — velocity comes back in the latent's own shape, and
    /// the prefix holds one K and one V per block.</summary>
    [Fact]
    public void AFullForwardReturnsTheLatentShapeAndAFiniteVelocity()
    {
        using CpuBackend backend = new CpuBackend();
        QwenImage21Config config = TinyConfig;
        using QwenImage21Transformer transformer = new QwenImage21Transformer(config, DType.F32);
        Dictionary<string, Tensor> weights = BuildWeights(config, seed: 7);
        transformer.LoadWeights(weights);

        const int textLen = 5, h = 4, w = 3;
        Tensor context = Random(new TensorShape(1, textLen, ContextDim), 11);
        using QwenImage21PrefixCache prefix = transformer.BuildPrefix(backend, context);
        context.Dispose();
        Assert.Equal(textLen, prefix.Length);

        Tensor latent = Random(new TensorShape(1, Channels, h, w), 13);
        using Tensor velocity = transformer.Forward(backend, latent, prefix, timestep: 0.5f);
        latent.Dispose();

        Assert.Equal(4, velocity.Shape.Rank);
        Assert.Equal(1, velocity.Shape[0]);
        Assert.Equal(Channels, velocity.Shape[1]);
        Assert.Equal(h, velocity.Shape[2]);
        Assert.Equal(w, velocity.Shape[3]);
        ReadOnlySpan<float> v = velocity.AsReadOnlySpan<float>();
        bool nonZero = false;
        foreach (float value in v)
        {
            Assert.True(float.IsFinite(value), $"velocity contains {value}");
            if (value != 0f) nonZero = true;
        }
        Assert.True(nonZero, "velocity is identically zero, so nothing in the block actually ran");
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    /// <summary>The prefix is timestep-independent by construction — that is the whole reason it can be cached
    /// across steps. Built twice it must be bit-identical, and the velocity must still change with the timestep, so
    /// the test cannot pass by the model simply ignoring <c>t</c>.</summary>
    [Fact]
    public void ThePrefixIsTimestepIndependentButTheVelocityIsNot()
    {
        using CpuBackend backend = new CpuBackend();
        QwenImage21Config config = TinyConfig;
        using QwenImage21Transformer transformer = new QwenImage21Transformer(config, DType.F32);
        Dictionary<string, Tensor> weights = BuildWeights(config, seed: 3);
        transformer.LoadWeights(weights);

        Tensor context = Random(new TensorShape(1, 4, ContextDim), 5);
        using QwenImage21PrefixCache first = transformer.BuildPrefix(backend, context);
        using QwenImage21PrefixCache second = transformer.BuildPrefix(backend, context);
        context.Dispose();

        Tensor latent = Random(new TensorShape(1, Channels, 2, 2), 17);
        using Tensor early = transformer.Forward(backend, latent, first, timestep: 0.9f);
        using Tensor late = transformer.Forward(backend, latent, second, timestep: 0.1f);
        using Tensor earlyAgain = transformer.Forward(backend, latent, second, timestep: 0.9f);
        latent.Dispose();

        Assert.Equal(early.AsReadOnlySpan<float>().ToArray(), earlyAgain.AsReadOnlySpan<float>().ToArray());
        Assert.NotEqual(early.AsReadOnlySpan<float>().ToArray(), late.AsReadOnlySpan<float>().ToArray());
        foreach (Tensor t in weights.Values) t.Dispose();
    }

    private static Dictionary<string, Tensor> BuildWeights(QwenImage21Config config, int seed)
    {
        int mlpDim = config.HiddenSize * config.MlpRatio;
        Dictionary<string, Tensor> w = new()
        {
            ["img_in.weight"] = Random(new TensorShape(config.HiddenSize, config.InChannels), seed + 1),
            ["txt_in.text_norm.weight"] = Random(new TensorShape(config.ContextDim), seed + 2),
            ["txt_in.in_layer.weight"] = Random(new TensorShape(config.HiddenSize, config.ContextDim), seed + 3),
            ["txt_in.out_layer.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 4),
            ["time_text_embed.timestep_embedder.linear_1.weight"] = Random(new TensorShape(config.HiddenSize, 256), seed + 5),
            ["time_text_embed.timestep_embedder.linear_2.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 6),
            ["modulation.1.weight"] = Random(new TensorShape(4 * config.HiddenSize, config.HiddenSize), seed + 7),
            ["norm_out.linear.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 8),
            ["proj_out.weight"] = Random(new TensorShape(config.OutChannels, config.HiddenSize), seed + 9),
        };
        for (int i = 0; i < config.Depth; i++)
        {
            string p = $"transformer_blocks.{i}";
            w[$"{p}.attn.to_q.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 100 + i * 8);
            w[$"{p}.attn.to_k.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 101 + i * 8);
            w[$"{p}.attn.to_v.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 102 + i * 8);
            w[$"{p}.attn.to_out.0.weight"] = Random(new TensorShape(config.HiddenSize, config.HiddenSize), seed + 103 + i * 8);
            w[$"{p}.attn.norm_q.weight"] = Ones(config.HeadDim);
            w[$"{p}.attn.norm_k.weight"] = Ones(config.HeadDim);
            w[$"{p}.img_mlp.gate_up.weight"] = Random(new TensorShape(2 * mlpDim, config.HiddenSize), seed + 104 + i * 8);
            w[$"{p}.img_mlp.out.weight"] = Random(new TensorShape(config.HiddenSize, mlpDim), seed + 105 + i * 8);
        }
        return w;
    }

    private static Tensor Random(TensorShape shape, int seed)
    {
        Tensor t = new Tensor(shape, DType.F32);
        Random rng = new Random(seed);
        Span<float> s = t.AsSpan<float>();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() - 0.5) * 0.2f;
        return t;
    }

    private static Tensor Zeros(int rows, int cols)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        t.AsSpan<float>().Clear();
        return t;
    }

    private static Tensor Ones(int n)
    {
        Tensor t = new Tensor(new TensorShape(n), DType.F32);
        t.AsSpan<float>().Fill(1.0f);
        return t;
    }
}
