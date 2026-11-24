namespace HartsyInference.Audio.Models.Separation;

/// <summary>The network step of an MDX-Net separator: a spectrogram in, the separated stem's spectrogram out. Behind an interface so
/// the DSP around it can be tested without weights.</summary>
public interface IMdxSpectrogramModel
{
    /// <summary>Maps <paramref name="spectrogram"/> to <paramref name="prediction"/>, both <c>[4, DimF, Frames]</c> row-major
    /// (channels are left real, left imaginary, right real, right imaginary).</summary>
    void Run(float[] spectrogram, float[] prediction);
}
