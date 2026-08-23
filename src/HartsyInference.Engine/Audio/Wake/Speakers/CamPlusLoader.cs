using HartsyInference.Audio.Models.CosyVoice;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Logging;
using HartsyInference.Core.Tensors;
using HartsyInference.ModelAssets.PyTorch;
using HartsyInference.ModelAssets.SafeTensors;

namespace HartsyInference.Engine.Audio.Wake.Speakers;

/// <summary>The CAM++ checkpoint load and fbank→embedding pass shared by the wake-word speaker path and the STT diarizer; each caller keeps its own log tag and its own span/clip guards.</summary>
internal static class CamPlusLoader
{
    /// <summary>Opens a CAM++ checkpoint (safetensors or PyTorch pickle, chosen by extension) and loads the encoder, auto-detecting whether the file is a standalone <c>campplus</c> or a CosyVoice/Chatterbox bundle that nests it under <c>speaker_encoder</c>. The returned loaders own the weight memory and must outlive the encoder.</summary>
    internal static (CamPlusSpeakerEncoder Encoder, IDisposable[] Loaders) Load(string path, string logTag, int embeddingDimension)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        IDisposable[] loaders;
        IReadOnlyDictionary<string, Tensor> weights;
        try
        {
            if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            {
                SafeTensorsLoader safetensors = new SafeTensorsLoader();
                safetensors.Load(path);
                weights = safetensors.GetAllTensors();
                loaders = [safetensors];
            }
            else
            {
                PytorchPickleLoader pickle = new PytorchPickleLoader();
                pickle.Load(path);
                weights = pickle.GetAllTensors();
                loaders = [pickle];
            }
        }
        catch (Exception ex)
        {
            Logs.Error($"{logTag} Failed to read the CAM++ checkpoint '{path}'", ex);
            throw;
        }

        // Standalone campplus checkpoints are unprefixed; the CosyVoice/Chatterbox bundle nests it under speaker_encoder.
        string prefix = weights.ContainsKey("xvector.dense.linear.weight") ? string.Empty : "speaker_encoder";
        CamPlusSpeakerEncoder encoder = new CamPlusSpeakerEncoder(embeddingDimension);
        try
        {
            encoder.LoadWeights(weights, prefix);
        }
        catch (Exception ex)
        {
            Logs.Error($"{logTag} '{path}' does not carry CAM++ weights under prefix '{prefix}'", ex);
            encoder.Dispose();
            foreach (IDisposable loader in loaders)
            {
                loader.Dispose();
            }
            throw;
        }
        Logs.Info($"{logTag} Loaded the CAM++ speaker encoder from '{path}'.");
        return (encoder, loaders);
    }

    /// <summary>Per-bin cepstral mean normalization over <paramref name="fbank"/> into the <c>[1, frames, bins]</c> input CAM++ expects, then one forward pass reduced to an L2-normalized embedding. Callers own the frame-count guard — this assumes the span is already long enough.</summary>
    internal static unsafe float[] Embed(IBackend backend, CamPlusSpeakerEncoder encoder, float[,] fbank)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(fbank);
        int frames = fbank.GetLength(0);
        int bins = fbank.GetLength(1);
        Tensor features = new Tensor(new TensorShape(1, frames, bins), DType.F32);
        try
        {
            float* destination = (float*)features.DataPointer;
            for (int bin = 0; bin < bins; bin++)
            {
                // Cepstral mean normalization, per bin over time — what CosyVoice feeds CAM++.
                double mean = 0d;
                for (int frame = 0; frame < frames; frame++)
                {
                    mean += fbank[frame, bin];
                }
                mean /= frames;
                for (int frame = 0; frame < frames; frame++)
                {
                    destination[(long)frame * bins + bin] = (float)(fbank[frame, bin] - mean);
                }
            }
            Tensor embedding = encoder.Forward(backend, features);
            try
            {
                int dimension = (int)embedding.Shape[embedding.Shape.Rank - 1];
                float[] vector = new float[dimension];
                new ReadOnlySpan<float>((float*)embedding.DataPointer, dimension).CopyTo(vector);
                SpeakerEmbeddingMath.NormalizeInPlace(vector);
                return vector;
            }
            finally
            {
                embedding.Dispose();
            }
        }
        finally
        {
            features.Dispose();
        }
    }
}
