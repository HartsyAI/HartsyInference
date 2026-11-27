using HartsyInference.Core.Backends.Cpu;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Pipelines.LongCat;
using Xunit;

namespace HartsyInference.Diffusion.Tests;

/// <summary>A multi-speaker forward of the tiny avatar DiT: every block's noisy tokens take the positions its own reference map gives, the
/// rotation reaches the output, and the forward refuses what the reference cannot run.</summary>
[Trait("Category", "SyntheticSmoke")]
public sealed class LongCatSpeakerDitTests
{
    [Fact]
    public void TheDitPlacesEachNoisyTokenByItsMapAndRefusesWhatTheReferenceCannotRun()
    {
        LongCatConfig config = LongCatTestData.TinyAvatar;
        Dictionary<string, Tensor> weights = LongCatTestData.AvatarWeights(config, seed: 5);
        try
        {
            using CpuBackend backend = new();
            using LongCatTransformer dit = new(config);
            dit.LoadWeights(weights);
            const int Frames = 3, Tracks = 2;
            using Tensor latent = LongCatTestData.Latent(config, Frames, 4, 8, seed: 6);
            using Tensor encoder = LongCatTestData.Encoder(config, 5, seed: 7);
            (int regions, float[] masks) = LongCatSpeakerMasks.TokenMasks(16, 32, null, null, [], 16, 32, 2, 4);
            LongCatSpeakers speakers = new(Tracks, regions, 2, 4, masks);
            using Tensor audio = LongCatTestData.Random(new TensorShape(Frames * Tracks * config.AudioContextTokens, config.AudioOutputDim), seed: 8);
            LongCatConditioning fromImage = new(NumCondLatents: 1);
            Dictionary<string, float[]> got = [];
            dit.Probe = (stage, t) => got[stage] = t.AsReadOnlySpan<float>().ToArray();

            using Tensor velocity = dit.Forward(backend, latent, encoder, [500f], in fromImage, audioTokens: audio, speakers: speakers);

            dit.Probe = null;
            int noisy = (Frames - 1) * 8;
            for (int block = 0; block < config.Depth; block++)
            {
                float[] map = got[$"block{block}.ref_attn_map"];
                Assert.Equal(regions * noisy, map.Length);
                Assert.Equal(LongCatSpeakerRope.QueryPositions(map, regions, noisy), got[$"block{block}.query_positions"]);
                Assert.Equal((Frames - 1) * Tracks * config.AudioContextTokens, got[$"block{block}.key_positions"].Length);
            }

            // The L-RoPE reaches the output: the same tokens attended without rotation give another result.
            using Tensor unrotated = dit.Forward(backend, latent, encoder, [500f], in fromImage, audioTokens: audio);
            Assert.NotEqual(velocity.AsReadOnlySpan<float>().ToArray(), unrotated.AsReadOnlySpan<float>().ToArray());

            LongCatConditioning none = new();
            Assert.Throws<ArgumentException>(() => dit.Forward(backend, latent, encoder, [500f], in none, audioTokens: audio, speakers: speakers));
            Assert.Throws<ArgumentException>(() => dit.Forward(backend, latent, encoder, [500f], in fromImage, audioTokens: audio,
                speakers: new LongCatSpeakers(Tracks, regions, 4, 2, masks)));
            Assert.Throws<ArgumentException>(() => dit.Forward(backend, latent, encoder, [500f], in fromImage, audioTokens: audio,
                speakers: new LongCatSpeakers(3, regions, 2, 4, masks)));
        }
        finally
        {
            LongCatTestData.DisposeAll(weights);
        }
    }
}
