namespace HartsyInference.Audio.Models.Denoise;

/// <summary>How RNNoise's network runs, chosen when its weights load (<see cref="RnnoiseWeights.LoadFile"/>).</summary>
public enum RnnoisePrecision
{
    /// <summary>Every layer in F32, from the PyTorch checkpoint. The default, and what the wake stack runs.</summary>
    Float,

    /// <summary>conv2 and the three GRUs on the int8 tables upstream's default C build compiles in
    /// (<see cref="RnnoiseInt8Tables"/>), with uint8 activations; conv1 and the two heads stay F32, as they do there.
    /// It reads about a quarter of the weight bytes per frame. The voice front end selects it.</summary>
    Int8,
}
