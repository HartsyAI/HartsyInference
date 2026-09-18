using System.Text;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets;
using HartsyInference.ModelAssets.Nf4;
using Xunit;

namespace HartsyInference.ModelAssets.Tests;

/// <summary>Covers the bitsandbytes <c>Linear4bit</c> fold.
/// <para><b>What this proves and what it does not:</b> the fixtures here are built to bitsandbytes' documented packed
/// layout, so these are <i>self-consistency</i> tests — they pin that the fold reads the layout this code believes in,
/// decodes it to the right values, and drops the companions. They cannot prove the belief matches a real bitsandbytes
/// file; that needs a shipped nf4 checkpoint and belongs in an Integration test. What protects a real file in the
/// meantime is that the fold reconciles every declared quantity against the actual byte counts and refuses on any
/// mismatch, so a layout misread surfaces as a named error rather than as decoded noise.</para></summary>
public sealed unsafe class Nf4CompanionFoldTests
{
    private const string StateKeySuffix = ".quant_state.bitsandbytes__nf4";

    private static Tensor Codebook()
    {
        Tensor tensor = new Tensor(new TensorShape(Nf4Codec.Nf4Lut.Length), DType.F32);
        Nf4Codec.Nf4Lut.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    private static Tensor Blob(string json)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(json);
        Tensor tensor = new Tensor(new TensorShape(bytes.Length), DType.U8);
        bytes.CopyTo(tensor.AsSpan<byte>());
        return tensor;
    }

    /// <summary>Packs <paramref name="codes"/> two nibbles per byte, first element in the high nibble.</summary>
    private static Tensor Packed(params int[] codes)
    {
        Tensor tensor = new Tensor(new TensorShape((codes.Length + 1) / 2), DType.U8);
        Span<byte> bytes = tensor.AsSpan<byte>();
        for (int i = 0; i < codes.Length; i++)
        {
            if ((i & 1) == 0) bytes[i >> 1] = (byte)(codes[i] << 4);
            else bytes[i >> 1] |= (byte)codes[i];
        }
        return tensor;
    }

    private static Tensor Absmax(params float[] values)
    {
        Tensor tensor = new Tensor(new TensorShape(values.Length), DType.F32);
        values.CopyTo(tensor.AsSpan<float>());
        return tensor;
    }

    /// <summary>A 2×4 NF4 weight in one block of 8, as bitsandbytes writes it.</summary>
    private static Dictionary<string, Tensor> Layer(string prefix = "mlp.gate_proj.weight",
        string json = "{\"quant_type\": \"nf4\", \"blocksize\": 8, \"shape\": [2, 4], \"dtype\": \"float32\"}") => new()
        {
            [prefix] = Packed(0, 5, 8, 15, 15, 8, 5, 0),
            [$"{prefix}.absmax"] = Absmax(2.0f),
            [$"{prefix}.quant_map"] = Codebook(),
            [$"{prefix}{StateKeySuffix}"] = Blob(json),
        };

    private static void DisposeAll(Dictionary<string, Tensor> weights)
    {
        foreach (Tensor tensor in weights.Values) tensor.Dispose();
    }

    [Fact]
    public void Apply_DequantizesTheWeightAndDropsEveryCompanion()
    {
        Dictionary<string, Tensor> weights = Layer();
        weights["mlp.gate_proj.bias"] = new Tensor(new TensorShape(2), DType.F32);
        try
        {
            Dictionary<string, Tensor> folded = Nf4CompanionFold.Apply(weights);
            try
            {
                Tensor weight = folded["mlp.gate_proj.weight"];
                Assert.Equal(DType.F32, weight.DType);
                Assert.Equal(2L, weight.Shape[0]);
                Assert.Equal(4L, weight.Shape[1]);
                ReadOnlySpan<float> values = weight.AsReadOnlySpan<float>();
                for (int i = 0; i < 8; i++)
                {
                    int code = i < 4 ? new[] { 0, 5, 8, 15 }[i] : new[] { 15, 8, 5, 0 }[i - 4];
                    Assert.Equal(Nf4Codec.Nf4Lut[code] * 2.0f, values[i], 5);
                }
                Assert.DoesNotContain("mlp.gate_proj.weight.absmax", folded.Keys);
                Assert.DoesNotContain("mlp.gate_proj.weight.quant_map", folded.Keys);
                Assert.DoesNotContain($"mlp.gate_proj.weight{StateKeySuffix}", folded.Keys);
                // Everything that is not a companion passes through untouched.
                Assert.Same(weights["mlp.gate_proj.bias"], folded["mlp.gate_proj.bias"]);
            }
            finally
            {
                folded["mlp.gate_proj.weight"].Dispose();
            }
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void OpenedThroughTheContainer_TheDecodedWeightIsFreedWithIt()
    {
        // An NF4 weight is decoded into a NEW tensor, not a view of the file, so nothing frees it unless the source
        // does. A recipe treats what the container hands over as borrowed, so a failed construction or an unload
        // would otherwise leave gigabytes alive until a finalizer ran.
        string dir = Path.Combine(Path.GetTempPath(), $"nf4_container_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            Dictionary<string, Tensor> weights = Layer();
            string path = Path.Combine(dir, "model.safetensors");
            try
            {
                SafeTensors.SafeTensorsWriter.Save(path, weights);
            }
            finally
            {
                DisposeAll(weights);
            }

            Tensor decoded;
            using (Checkpoints.CheckpointSource source = Checkpoints.CheckpointSource.Open(path))
            {
                decoded = source.Weights["mlp.gate_proj.weight"];
                Assert.Equal(DType.F32, decoded.DType);
                _ = decoded.DataPointer;
            }

            Assert.Throws<ObjectDisposedException>(() => _ = decoded.DataPointer);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); }
            catch (IOException) { /* a mmap the OS has not released yet is not a test failure */ }
        }
    }

    [Fact]
    public void Apply_FreesWhatItAlreadyDecodedWhenALaterWeightRefuses()
    {
        // A checkpoint decodes one weight at a time and a malformed companion set can sit anywhere in it, so learning
        // what was allocated only on a successful return leaks everything decoded before the failure — gigabytes on a
        // real file, and again on every retry.
        Dictionary<string, Tensor> weights = Layer("good.weight");
        foreach (KeyValuePair<string, Tensor> entry in Layer("bad.weight",
            json: "{\"quant_type\": \"nf4\", \"blocksize\": 8, \"shape\": [4, 4], \"dtype\": \"float32\"}"))
        {
            weights[entry.Key] = entry.Value;
        }
        List<Tensor> allocated = new List<Tensor>();
        try
        {
            Assert.Throws<NotSupportedException>(() => Nf4CompanionFold.Apply(weights, allocated));

            // The good weight decoded before the bad one refused, and the caller can now free it.
            Assert.NotEmpty(allocated);
            foreach (Tensor tensor in allocated) tensor.Dispose();
            foreach (Tensor tensor in allocated)
                Assert.Throws<ObjectDisposedException>(() => _ = tensor.DataPointer);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_ReturnsTheInputUntouchedWhenNothingIsNf4()
    {
        Dictionary<string, Tensor> weights = new() { ["blocks.0.attn.to_q.weight"] = new Tensor(new TensorShape(4, 8), DType.F32) };
        try
        {
            Assert.Same(weights, Nf4CompanionFold.Apply(weights));
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_DequantizesToTheDtypeTheQuantizerCapturedFrom()
    {
        Dictionary<string, Tensor> weights = Layer(json:
            "{\"quant_type\": \"nf4\", \"blocksize\": 8, \"shape\": [2, 4], \"dtype\": \"torch.bfloat16\"}");
        try
        {
            Dictionary<string, Tensor> folded = Nf4CompanionFold.Apply(weights);
            try
            {
                Assert.Equal(DType.BF16, folded["mlp.gate_proj.weight"].DType);
            }
            finally
            {
                folded["mlp.gate_proj.weight"].Dispose();
            }
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_RefusesWhenTheDeclaredShapeDoesNotMatchThePackedBytes()
    {
        // The guard that stops a misread layout from decoding to plausible noise.
        Dictionary<string, Tensor> weights = Layer(json:
            "{\"quant_type\": \"nf4\", \"blocksize\": 8, \"shape\": [4, 4], \"dtype\": \"float32\"}");
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Nf4CompanionFold.Apply(weights));
            Assert.Contains("does not describe this weight", error.Message);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_RefusesWhenTheBlockScaleCountDoesNotMatchTheBlockSize()
    {
        Dictionary<string, Tensor> weights = Layer(json:
            "{\"quant_type\": \"nf4\", \"blocksize\": 4, \"shape\": [2, 4], \"dtype\": \"float32\"}");
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Nf4CompanionFold.Apply(weights));
            Assert.Contains("block scales", error.Message);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_RefusesACodebookThatIsNotTheNf4Table()
    {
        Dictionary<string, Tensor> weights = Layer();
        weights["mlp.gate_proj.weight.quant_map"].AsSpan<float>()[3] = 0.123f;
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Nf4CompanionFold.Apply(weights));
            Assert.Contains("not bitsandbytes' NF4 table", error.Message);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_RefusesFp4RatherThanDecodingItThroughNf4Quantiles()
    {
        Dictionary<string, Tensor> weights = Layer();
        Tensor state = weights[$"mlp.gate_proj.weight{StateKeySuffix}"];
        weights.Remove($"mlp.gate_proj.weight{StateKeySuffix}");
        weights["mlp.gate_proj.weight.quant_state.bitsandbytes__fp4"] = state;
        try
        {
            NotSupportedException error = Assert.Throws<NotSupportedException>(() => Nf4CompanionFold.Apply(weights));
            Assert.Contains("not nf4", error.Message);
        }
        finally
        {
            DisposeAll(weights);
        }
    }

    [Fact]
    public void Apply_RebuildsADoubleQuantizedAbsmaxBeforeDecoding()
    {
        Dictionary<string, Tensor> weights = Layer();
        weights["mlp.gate_proj.weight.absmax"].Dispose();

        // absmax[0] = nestedCodebook[code] * nestedAbsmax[0] + offset = 0.5 * 3.0 + 0.5 = 2.0, the plain case's scale.
        Tensor quantizedAbsmax = new Tensor(new TensorShape(1), DType.U8);
        quantizedAbsmax.AsSpan<byte>()[0] = 7;
        Tensor nestedCodebook = new Tensor(new TensorShape(256), DType.F32);
        nestedCodebook.AsSpan<float>()[7] = 0.5f;
        weights["mlp.gate_proj.weight.absmax"] = quantizedAbsmax;
        weights["mlp.gate_proj.weight.nested_absmax"] = Absmax(3.0f);
        weights["mlp.gate_proj.weight.nested_quant_map"] = nestedCodebook;
        weights[$"mlp.gate_proj.weight{StateKeySuffix}"].Dispose();
        weights[$"mlp.gate_proj.weight{StateKeySuffix}"] = Blob(
            "{\"quant_type\": \"nf4\", \"blocksize\": 8, \"shape\": [2, 4], \"dtype\": \"float32\", "
            + "\"nested_blocksize\": 256, \"nested_offset\": 0.5}");
        try
        {
            Dictionary<string, Tensor> folded = Nf4CompanionFold.Apply(weights);
            try
            {
                ReadOnlySpan<float> values = folded["mlp.gate_proj.weight"].AsReadOnlySpan<float>();
                Assert.Equal(Nf4Codec.Nf4Lut[5] * 2.0f, values[1], 5);
                Assert.DoesNotContain("mlp.gate_proj.weight.nested_absmax", folded.Keys);
                Assert.DoesNotContain("mlp.gate_proj.weight.nested_quant_map", folded.Keys);
            }
            finally
            {
                folded["mlp.gate_proj.weight"].Dispose();
            }
        }
        finally
        {
            DisposeAll(weights);
        }
    }
}
