using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.SafeTensors;
using Xunit;
using Xunit.Abstractions;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Loads a REAL LTX-2.5 `int8_tensorwise` + ConvRot transformer and checks every quantized layer arrives
/// packed, scaled and tagged. The synthetic-fixture tests cover the attach logic; what only a real file can catch is
/// a layer the shipped quantizer treated differently from the ones we sampled — an unrotated projection, a per-tensor
/// scale, a `full_precision_matrix_mult` opt-out.</summary>
/// <remarks><para>Cheap despite the 21.5 GB file: <see cref="SafeTensorsLoader"/> mmaps, and this reads only the
/// scale and descriptor companions (a few MB) plus the header. The int8 weight bytes are never touched.</para>
/// <para>Set <c>LTX25_INT8_CHECKPOINT</c> to point at the file, or drop it in <c>Models/diffusion_models/</c>. The
/// official <c>Lightricks/LTX-2.5</c> build is gated (downloads 401 even for byte ranges);
/// <c>DmitryDB/LTX-2.5-ComfyUI-Quants</c> republishes the same layout ungated.</para></remarks>
public sealed class Ltx25Int8ConvRotLoadTests(ITestOutputHelper output)
{
    private const string FileName = "ltx-2.5-22b-dev-transformer-int8_lean_convrot.safetensors";

    private readonly ITestOutputHelper _output = output;

    /// <summary>Smallest plausible size for this build (21.50 GB). An in-progress download passes
    /// <see cref="File.Exists"/> and then fails deep inside the loader, which reads as a real regression.</summary>
    private const long MinimumBytes = 20L << 30;

    private static string? CheckpointPath()
    {
        string? fromEnv = Environment.GetEnvironmentVariable("LTX25_INT8_CHECKPOINT");
        if (!string.IsNullOrWhiteSpace(fromEnv)) return Complete(fromEnv) ? fromEnv : null;

        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "Models", "diffusion_models", FileName);
            if (Complete(candidate)) return candidate;
        }
        return null;
    }

    private static bool Complete(string path) => File.Exists(path) && new FileInfo(path).Length >= MinimumBytes;

    private const string TextEncoderFileName = "gemma4-12b-with-proj-ltx-2.5-int8_lean_convrot.safetensors";

    private static string? TextEncoderPath()
    {
        string? fromEnv = Environment.GetEnvironmentVariable("LTX25_GEMMA4_INT8");
        if (!string.IsNullOrWhiteSpace(fromEnv)) return Sized(fromEnv, 15_000_000_000L) ? fromEnv : null;

        string? dir = AppContext.BaseDirectory;
        for (int up = 0; up < 8 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "Models", "text_encoders", TextEncoderFileName);
            if (Sized(candidate, 15_000_000_000L)) return candidate;
        }
        return null;
    }

    private static bool Sized(string path, long minimum) => File.Exists(path) && new FileInfo(path).Length >= minimum;

    [Fact]
    [Trait("Category", "Integration")]
    public void EveryQuantizedTransformerLayerArrivesPackedAndTagged()
    {
        string? path = CheckpointPath();
        if (path is null) return;   // tier-lint: guarded

        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(path);
        using (loader)
        {
            int packed = 0, rotated = 0, unrotated = 0, perTensorScale = 0, fullPrecision = 0;
            foreach (KeyValuePair<string, Tensor> entry in weights.Transformer.Concat(weights.Connectors))
            {
                if (entry.Value.DType != DType.I8) continue;
                packed++;

                QuantWeightInfo? info = entry.Value.QuantInfo;
                Assert.True(info is not null, $"int8 weight '{entry.Key}' reached the model with no QuantInfo.");
                Assert.Equal("int8_tensorwise", info!.Format);
                Assert.NotNull(info.RowScale);
                Assert.Equal(DType.F32, info.RowScale!.DType);

                long outFeatures = entry.Value.Shape[0];
                long inFeatures = entry.Value.Shape[1];
                Assert.True(info.RowScale.ElementCount == outFeatures || info.RowScale.ElementCount == 1,
                    $"'{entry.Key}' scale holds {info.RowScale.ElementCount}, expected {outFeatures} or 1.");
                if (info.RowScale.ElementCount == 1) perTensorScale++;
                if (info.FullPrecisionMatMul) fullPrecision++;

                if (info.ConvRotGroupSize == 0) { unrotated++; continue; }
                rotated++;
                Assert.True(Int8ConvRotCodec.IsValidGroupSize(info.ConvRotGroupSize),
                    $"'{entry.Key}' ConvRot group {info.ConvRotGroupSize} is not a power of four.");
                Assert.True(inFeatures % info.ConvRotGroupSize == 0,
                    $"'{entry.Key}' in_features {inFeatures} is not a multiple of ConvRot group {info.ConvRotGroupSize}.");
            }

            _output.WriteLine($"packed int8 layers: {packed} (rotated {rotated}, unrotated {unrotated}, "
                + $"per-tensor scale {perTensorScale}, full-precision opt-out {fullPrecision})");

            // The dev build's own count. A converter change that starts dequantizing or dropping layers shows up
            // here as a number, not as a slow, subtly-wrong generation.
            Assert.Equal(1440, packed);

            // No companion may survive into the model: a stray `.weight_scale` is a silent extra weight, and a
            // stray `.comfy_quant` would fail a strict key check downstream.
            foreach (string key in weights.Transformer.Keys.Concat(weights.Connectors.Keys))
            {
                Assert.False(key.EndsWith(".weight_scale", StringComparison.Ordinal), $"companion leaked: {key}");
                Assert.False(key.EndsWith(".comfy_quant", StringComparison.Ordinal), $"companion leaked: {key}");
            }
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void TheStandaloneGemma4TowerRoutesToTheTextEncoderBucketNotTheDit()
    {
        string? path = TextEncoderPath();
        if (path is null) return;   // tier-lint: guarded

        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(path);
        using (loader)
        {
            int packedTowerWeights = weights.TextEncoder.Count(
                e => e.Value.DType == DType.I8 && e.Value.QuantInfo is not null);
            _output.WriteLine($"buckets — text encoder {weights.TextEncoder.Count} ({packedTowerWeights} packed int8), "
                + $"connectors {weights.Connectors.Count}, transformer {weights.Transformer.Count}, vae {weights.Vae.Count}");

            // The tower ships with BARE `model.layers.*` keys. Before these were routed explicitly they fell through
            // to the DiT mapper, so this asserts the negative: not one tower tensor may reach the transformer.
            Assert.Empty(weights.Transformer);
            Assert.Equal(328, packedTowerWeights);
            Assert.True(weights.TextEncoder.ContainsKey("model.layers.0.layer_scalar"),
                "the Gemma 4 discriminator key must survive into the text-encoder bucket — the recipe branches on it.");
            Assert.True(weights.TextEncoder.ContainsKey("tokenizer_json"),
                "Gemma 4's vocabulary ships only inside this file; without it the recipe cannot build a tokenizer.");

            // The caption projection lives in the same file but belongs to the connectors, not the tower.
            Assert.True(weights.Connectors.ContainsKey("text_embedding_projection.video_aggregate_embed.weight"));
            Assert.True(weights.Connectors.ContainsKey("text_embedding_projection.audio_aggregate_embed.weight"));
        }
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void DequantizedWeightsAreCenteredAndFinite()
    {
        string? path = CheckpointPath();
        if (path is null) return;   // tier-lint: guarded

        (LtxVideo2CheckpointConverter.ConvertedWeights weights, SafeTensorsLoader loader) =
            LtxVideo2CheckpointConverter.LoadAndConvert(path);
        using (loader)
        {
            // One representative layer, dequantized through the same codec the CPU/Vulkan fallback uses. A wrong
            // rotation or a mis-broadcast scale does not throw — it produces a weight whose scale is orders out,
            // which is exactly what this catches without needing the BF16 sibling downloaded.
            KeyValuePair<string, Tensor> sample = weights.Transformer
                .Where(e => e.Value.DType == DType.I8 && e.Value.QuantInfo?.ConvRotGroupSize > 0)
                .OrderBy(e => e.Key, StringComparer.Ordinal)
                .First();
            QuantWeightInfo info = sample.Value.QuantInfo!;

            using Tensor dequantized = Int8ConvRotCodec.DequantToBf16(sample.Value, info.RowScale!, info.ConvRotGroupSize);
            ReadOnlySpan<ushort> raw = dequantized.AsReadOnlySpan<ushort>();
            double sumSquares = 0;
            float maxAbs = 0;
            for (int i = 0; i < raw.Length; i++)
            {
                float value = BitConverter.UInt32BitsToSingle((uint)raw[i] << 16);
                Assert.True(float.IsFinite(value), $"'{sample.Key}' element {i} dequantized to {value}.");
                sumSquares += (double)value * value;
                maxAbs = Math.Max(maxAbs, Math.Abs(value));
            }
            double rms = Math.Sqrt(sumSquares / raw.Length);
            _output.WriteLine($"{sample.Key}: shape {sample.Value.Shape}, group {info.ConvRotGroupSize}, "
                + $"rms {rms:E3}, absmax {maxAbs:E3}");

            // Transformer projections sit in the 1e-3..1e-1 RMS band in every LTX-2 build inspected; a dropped
            // rotation or a squared/omitted scale lands orders outside it.
            Assert.InRange(rms, 1e-4, 1.0);
            Assert.InRange(maxAbs, 1e-3, 100.0);
        }
    }
}
