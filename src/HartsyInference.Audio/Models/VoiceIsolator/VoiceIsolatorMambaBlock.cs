using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Audio.Models.VoiceIsolator;

/// <summary>A dual-path Mamba block: a residual Mamba along time within each band, then one across the bands of each frame.</summary>
internal sealed class VoiceIsolatorMambaBlock : IDisposable
{
    private readonly VoiceIsolatorConfig _config;
    private readonly VoiceIsolatorBidirectionalMamba _time;
    private readonly VoiceIsolatorBidirectionalMamba _frequency;

    public VoiceIsolatorMambaBlock(VoiceIsolatorConfig config, IReadOnlyDictionary<string, Tensor> weights, string prefix)
    {
        _config = config;
        _time = new VoiceIsolatorBidirectionalMamba(config, weights, $"{prefix}.time_mamba", config.Causal);
        _frequency = new VoiceIsolatorBidirectionalMamba(config, weights, $"{prefix}.freq_mamba", causal: false);
    }

    /// <summary>Runs the block over band-major tokens <c>[bands * frames, width]</c>; <paramref name="carry"/> continues the time scan of a stream.</summary>
    public Tensor Forward(IBackend backend, Tensor x, int bands, int frames, VoiceIsolatorMambaCarry? carry)
    {
        int width = _config.Width;
        long n = (long)bands * frames;
        Tensor timeOut = _time.Forward(backend, x, bands, frames, carry);
        Tensor afterTime = new(new TensorShape(1, n, width), DType.F32);
        backend.Add(afterTime, x, timeOut);
        timeOut.Dispose();
        Tensor frameMajor = new(new TensorShape(frames, bands, width), DType.F32);
        VoiceIsolatorOps.SwapLeading(frameMajor, afterTime, bands, frames, width);
        afterTime.Dispose();
        Tensor frequencyOut = _frequency.Forward(backend, frameMajor, frames, bands, null);
        Tensor afterFrequency = new(new TensorShape(1, n, width), DType.F32);
        backend.Add(afterFrequency, frameMajor, frequencyOut);
        frequencyOut.Dispose();
        frameMajor.Dispose();
        Tensor bandMajor = new(new TensorShape(1, n, width), DType.F32);
        VoiceIsolatorOps.SwapLeading(bandMajor, afterFrequency, frames, bands, width);
        afterFrequency.Dispose();
        return bandMajor;
    }

    public void Dispose()
    {
        _time.Dispose();
        _frequency.Dispose();
    }
}
