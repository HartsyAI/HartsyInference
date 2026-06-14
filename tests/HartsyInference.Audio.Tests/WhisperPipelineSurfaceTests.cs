using HartsyInference.Audio.Models.Whisper;
using HartsyInference.Audio.Pipelines;
using HartsyInference.Audio.Preprocessing;
using Xunit;

namespace HartsyInference.Audio.Tests;

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
        HartsyInference.Core.Tensors.Tensor mel = new(
            new HartsyInference.Core.Tensors.TensorShape(1, 80, 3000),
            HartsyInference.Core.Tensors.DType.F32);
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
    private sealed class DummyBackend : HartsyInference.Core.Backends.IBackend
    {
        public HartsyInference.Core.Backends.DeviceKind Device => default;
        public HartsyInference.Core.Backends.BackendCapabilities Capabilities => new() { Name = "Dummy" };
        public void Dispose() { }
        public void MatMul(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor a, HartsyInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void BatchedMatMul(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor a, HartsyInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void Linear(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, HartsyInference.Core.Tensors.Tensor? b) => throw new NotImplementedException();
        public void Conv2D(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, HartsyInference.Core.Tensors.Tensor? b, int sH, int sW, int pH, int pW) => throw new NotImplementedException();
        public void GroupNorm(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, HartsyInference.Core.Tensors.Tensor b, int g, float e) => throw new NotImplementedException();
        public void LayerNorm(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, HartsyInference.Core.Tensors.Tensor b, float e) => throw new NotImplementedException();
        public void RmsNorm(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, float e) => throw new NotImplementedException();
        public void AdaInstanceNorm1d(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor g, HartsyInference.Core.Tensors.Tensor b, float e) => throw new NotImplementedException();
        public void LeakyRelu(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, float s) => throw new NotImplementedException();
        public void ScaledDotProductAttention(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor q, HartsyInference.Core.Tensors.Tensor k, HartsyInference.Core.Tensors.Tensor v, HartsyInference.Core.Tensors.Tensor? m, float s) => throw new NotImplementedException();
        public void Gelu(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Silu(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Sigmoid(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Tanh(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Elu(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, float a) => throw new NotImplementedException();
        public void Snake(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor a, HartsyInference.Core.Tensors.Tensor? b) => throw new NotImplementedException();
        public void Conv1d(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, HartsyInference.Core.Tensors.Tensor? b, int s, int pl, int pr, int d, int g) => throw new NotImplementedException();
        public void ConvTranspose1d(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor w, HartsyInference.Core.Tensors.Tensor? b, int s, int pl, int pr, int d) => throw new NotImplementedException();
        public void Add(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor a, HartsyInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void Mul(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor a, HartsyInference.Core.Tensors.Tensor b) => throw new NotImplementedException();
        public void Scale(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, float s) => throw new NotImplementedException();
        public void Clamp(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, float a, float b) => throw new NotImplementedException();
        public void Transpose2D(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, int a, int b) => throw new NotImplementedException();
        public void Permute0213(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, int s, int h, int d) => throw new NotImplementedException();
        public void GeGlu(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void BroadcastAdd(HartsyInference.Core.Tensors.Tensor h, HartsyInference.Core.Tensors.Tensor b, int c, int s) => throw new NotImplementedException();
        public void Concat(HartsyInference.Core.Tensors.Tensor o, ReadOnlySpan<HartsyInference.Core.Tensors.Tensor> i, int d) => throw new NotImplementedException();
        public void Split(ReadOnlySpan<HartsyInference.Core.Tensors.Tensor> o, HartsyInference.Core.Tensors.Tensor i, int d) => throw new NotImplementedException();
        public void UpsampleNearest2D(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, int sH, int sW) => throw new NotImplementedException();
        public void UpsampleBilinear2D(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, int sH, int sW) => throw new NotImplementedException();
        public void CopyTo(HartsyInference.Core.Tensors.Tensor d, HartsyInference.Core.Tensors.Tensor s) => throw new NotImplementedException();
        public void Fill(HartsyInference.Core.Tensors.Tensor t, float v) => throw new NotImplementedException();
        public void Fft(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i) => throw new NotImplementedException();
        public void Stft(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, int n, int h, HartsyInference.Core.Tensors.Tensor w) => throw new NotImplementedException();
        public void MelFilterbank(HartsyInference.Core.Tensors.Tensor o, HartsyInference.Core.Tensors.Tensor i, HartsyInference.Core.Tensors.Tensor f) => throw new NotImplementedException();
        public HartsyInference.Core.Backends.IStreamingWeightCache? StreamingCache => null;
    }
}
