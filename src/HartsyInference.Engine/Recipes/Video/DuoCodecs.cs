using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Duo;
using HartsyInference.Diffusion.Models.Vae;

namespace HartsyInference.Engine.Recipes.Video;

/// <summary>The loaded codecs of a Duo pipeline: a factory for each stream decoder, which every run builds fresh, and the codec weights to
/// keep resident while a run decodes.</summary>
internal sealed record DuoCodecs(Func<IBackend, DuoVideoStreamDecoder> Video, Func<IBackend, DuoAudioStreamDecoder> Audio,
    Func<IEnumerable<Tensor>> Weights);
