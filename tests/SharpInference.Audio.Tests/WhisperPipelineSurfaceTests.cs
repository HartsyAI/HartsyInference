using SharpInference.Audio.Models.Whisper;
using SharpInference.Audio.Pipelines;
using SharpInference.Audio.Preprocessing;
using Xunit;

namespace SharpInference.Audio.Tests;

/// <summary>Pipeline-surface tests that don't actually load weights. The end-to-end
/// "download whisper-tiny and transcribe" test belongs in a network-gated trait —
/// covered separately when run with the appropriate filter.</summary>
public sealed class WhisperPipelineSurfaceTests
{
    [Fact]
    public void WhisperOptions_Defaults_AreSafe()
    {
        WhisperOptions opts = new();
        Assert.Equal("en", opts.Language);
        Assert.False(opts.Translate);
        Assert.False(opts.WithTimestamps);
        Assert.Equal(224, opts.MaxNewTokens);
        Assert.Equal(0f, opts.Temperature);
    }

    [Fact]
    public void WhisperConfig_NumMelBins_DrivesMelExtractorChoice()
    {
        // The pipeline constructs a MelSpectrogramExtractor from the model's mel-bin
        // count. Validate that both branches (80 and 128 mel) produce valid configs.
        MelSpectrogramExtractor.Config mel80 = MelSpectrogramExtractor.WhisperConfig(WhisperConfig.Tiny.NumMelBins);
        Assert.Equal(80, mel80.NMels);

        MelSpectrogramExtractor.Config mel128 = MelSpectrogramExtractor.WhisperConfig(WhisperConfig.LargeV3.NumMelBins);
        Assert.Equal(128, mel128.NMels);
    }

    [Fact]
    public void Encoder_Construction_DoesNotAllocateWeightTensors()
    {
        // Constructing an encoder should not allocate any GPU/CPU weight memory —
        // weights are populated only by LoadWeights. This guarantees that an audio
        // pipeline can be constructed at startup for many models without paying the
        // full RAM cost of every checkpoint.
        WhisperEncoder enc = new(WhisperConfig.Tiny);
        // EnumerateWeights returns no items before LoadWeights — there's nothing to enumerate.
        Assert.Empty(enc.EnumerateWeights());
        enc.Dispose();
    }

    [Fact]
    public void Decoder_Construction_AlsoDoesNotAllocate()
    {
        WhisperDecoder dec = new(WhisperConfig.Tiny);
        Assert.Empty(dec.EnumerateWeights());
        dec.Dispose();
    }

    [Fact]
    public void Encoder_Forward_BeforeLoadWeights_Throws()
    {
        using WhisperEncoder enc = new(WhisperConfig.Tiny);
        SharpInference.Core.Tensors.Tensor mel = new(
            new SharpInference.Core.Tensors.TensorShape(1, 80, 3000),
            SharpInference.Core.Tensors.DType.F32);
        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                enc.Forward(new DummyBackend(), mel));
        }
        finally { mel.Dispose(); }
    }

    /// <summary>A minimal IBackend stub used only to prove that surface tests don't
    /// accidentally hit the math path. Every operation throws — tests using this
    /// backend are expected to fail before invoking it.</summary>
    private sealed class DummyBackend : SharpInference.Core.Backends.IBackend
    {
        public SharpInference.Core.Backends.DeviceKind Device => default;
        public SharpInference.Core.Backends.BackendCapabilities Capabilities => new() { Name = "Dummy" };
        public void Dispose() { }
        public void MatMul(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor a, SharpInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void BatchedMatMul(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor a, SharpInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void Linear(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, SharpInference.Core.Tensors.Tensor? b) => throw new NotImplementedException();
        public void Conv2D(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, SharpInference.Core.Tensors.Tensor? b, int sH, int sW, int pH, int pW) => throw new NotImplementedException();
        public void GroupNorm(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, SharpInference.Core.Tensors.Tensor b, int g, float e) => throw new NotImplementedException();
        public void LayerNorm(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, SharpInference.Core.Tensors.Tensor b, float e) => throw new NotImplementedException();
        public void RmsNorm(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, float e) => throw new NotImplementedException();
        public void ScaledDotProductAttention(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor q, SharpInference.Core.Tensors.Tensor k, SharpInference.Core.Tensors.Tensor v, SharpInference.Core.Tensors.Tensor? m, float s) => throw new NotImplementedException();
        public void Gelu(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Silu(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Sigmoid(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Tanh(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Elu(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, float a) => throw new NotImplementedException();
        public void Snake(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor a, SharpInference.Core.Tensors.Tensor? b) => throw new NotImplementedException();
        public void Conv1d(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, SharpInference.Core.Tensors.Tensor? b, int s, int pl, int pr, int d, int g) => throw new NotImplementedException();
        public void ConvTranspose1d(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor w, SharpInference.Core.Tensors.Tensor? b, int s, int pl, int pr, int d) => throw new NotImplementedException();
        public void Add(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor a, SharpInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void Mul(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor a, SharpInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void Scale(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, float s) => throw new NotImplementedException();
        public void Clamp(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, float a, float b) => throw new NotImplementedException();
        public void Transpose2D(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, int a, int b) => throw new NotImplementedException();
        public void Permute0213(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, int s, int h, int d) => throw new NotImplementedException();
        public void GeGlu(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void BroadcastAdd(SharpInference.Core.Tensors.Tensor h, SharpInference.Core.Tensors.Tensor b, int c, int s) => throw new NotImplementedException();
        public void Concat(SharpInference.Core.Tensors.Tensor o, ReadOnlySpan<SharpInference.Core.Tensors.Tensor> i, int d) => throw new NotImplementedException();
        public void Split(ReadOnlySpan<SharpInference.Core.Tensors.Tensor> o, SharpInference.Core.Tensors.Tensor i, int d) => throw new NotImplementedException();
        public void UpsampleNearest2D(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, int sH, int sW) => throw new NotImplementedException();
        public void UpsampleBilinear2D(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, int sH, int sW) => throw new NotImplementedException();
        public void CopyTo(SharpInference.Core.Tensors.Tensor d, SharpInference.Core.Tensors.Tensor s) => throw new NotImplementedException();
        public void Fill(SharpInference.Core.Tensors.Tensor t, float v) => throw new NotImplementedException();
        public void Fft(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Stft(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, int n, int h, SharpInference.Core.Tensors.Tensor w) => throw new NotImplementedException();
        public void MelFilterbank(SharpInference.Core.Tensors.Tensor o, SharpInference.Core.Tensors.Tensor i, SharpInference.Core.Tensors.Tensor f) => throw new NotImplementedException();
        public SharpInference.Core.Backends.IStreamingWeightCache? StreamingCache => null;
    }
}
