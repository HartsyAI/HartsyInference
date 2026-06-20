using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Tests.Common;
using HartsyInference.Video.Pipelines;

namespace HartsyInference.Video.Tests;

/// <summary>End-to-end structural test for the Wan-Animate pipeline: tiny-config <see cref="WanAnimateTransformer"/>
/// (motion + face + pose pathway) + reused Wan VAE encode/decode driven through <see cref="WanAnimatePipeline"/> on CPU
/// (pose video + face video → frames). Numerics validation-pending.</summary>
public unsafe class WanAnimatePipelineTests
{
    [Fact]
    public void GenerateAnimation_TinyConfig_ProducesFrames()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 48, OutChannels = 48, VaeLatentChannels = 48,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 5, PatchSize = (1, 2, 2),
            VaeSpatialCompression = 16, VaeTemporalCompression = 4,
            NumInferenceSteps = 2, GuidanceScale = 5, FlowShift = 5,
        };
        const int poseCh = 48, motionSize = 16, motionStyle = 8, motionVec = 4, motionOut = 8, motionBlocks = 2;
        const int faceHidden = 16, faceHeads = 2, inject = 5;
        Dictionary<int, int> motionChannels = new() { [4] = 8, [8] = 8, [16] = 8 };

        WanAnimateTransformer transformer = new(cfg, poseLatentChannels: poseCh, motionEncoderSize: motionSize,
            motionDim: motionOut, faceHiddenDim: faceHidden, faceNumHeads: faceHeads, injectFaceLatentsBlocks: inject,
            motionChannelSizes: motionChannels, motionVecDim: motionVec, motionBlocks: motionBlocks);
        transformer.LoadWeights(BuildAnimateWeights(cfg, poseCh, motionSize, motionStyle, motionVec, motionOut, motionBlocks,
            motionChannels, faceHidden, faceHeads, inject));

        int[] dimMult = [1, 2, 4, 4];
        Wan22VaeDecoder vae = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalUpsample: [false, true, true]);
        vae.LoadWeights(LanceSyntheticWeights.BuildVae(8, 48, dimMult, 2, [false, true, true]));
        Wan22VaeEncoder encoder = new(dim: 8, zDim: 48, dimMult: dimMult, numResBlocks: 2, temperalDownsample: [true, true, false]);
        encoder.LoadWeights(LanceSyntheticWeights.BuildVaeEncoder(8, 48, dimMult, 2, [true, true, false]));

        WanAnimatePipeline pipeline = new(backend, transformer, vae, encoder, cfg);

        Tensor promptEmbeds = RandRows(3, cfg.TextDim, seed: 1);
        Tensor negEmbeds = RandRows(2, cfg.TextDim, seed: 2);
        Tensor poseClip = Rand5d(1, 3, 5, 32, 32, seed: 9);                 // 5 pose frames → 2 latent frames (S=2*2*2=8)
        Tensor faceClip = Rand5d(1, 3, 4, motionSize, motionSize, seed: 10);  // 4 face frames → motion T=2 (divides S=8)
        TextToImageRequest req = new() { Prompt = "x", Width = 32, Height = 32, Steps = 2, CfgScale = 5, Seed = 42 };

        (byte[][] frames, int w, int h, _) = pipeline.GenerateAnimation(promptEmbeds, negEmbeds, poseClip, faceClip, req);
        Assert.Equal(5, frames.Length);
        Assert.Equal(32, w);
        Assert.Equal(32, h);
    }

    private static Dictionary<string, Tensor> BuildAnimateWeights(WanVideoConfig c, int poseCh, int motionSize, int style,
        int motionVec, int motionOut, int motionBlocks, Dictionary<int, int> motionChannels, int faceHidden, int faceHeads, int inject)
    {
        Dictionary<string, Tensor> w = WanSyntheticWeights.BuildTransformer(c);
        int dim = c.InnerDim;
        w["pose_patch_embedding.weight"] = R([dim, poseCh, c.PatchSize.T, c.PatchSize.H, c.PatchSize.W]);
        w["pose_patch_embedding.bias"] = R([dim]);

        int ch = motionChannels[motionSize];
        w["motion_encoder.conv_in.weight"] = R([ch, 3, 1, 1]); w["motion_encoder.conv_in.act_fn.bias"] = R([ch]);
        w["motion_encoder.conv_out.weight"] = R([style, ch, 4, 4]);
        w["motion_encoder.motion_synthesis_weight"] = R([motionOut, motionVec]);
        int logSize = (int)Math.Round(Math.Log2(motionSize)), idx = 0;
        for (int i = logSize; i > 2; i--)
        {
            string b = $"motion_encoder.res_blocks.{idx}";
            w[$"{b}.conv1.weight"] = R([ch, ch, 3, 3]); w[$"{b}.conv1.act_fn.bias"] = R([ch]);
            w[$"{b}.conv2.weight"] = R([ch, ch, 3, 3]); w[$"{b}.conv2.act_fn.bias"] = R([ch]);
            w[$"{b}.conv_skip.weight"] = R([ch, ch, 1, 1]);
            idx++;
        }
        for (int i = 0; i < motionBlocks - 1; i++) { w[$"motion_encoder.motion_network.{i}.weight"] = R([style, style]); w[$"motion_encoder.motion_network.{i}.bias"] = R([style]); }
        w[$"motion_encoder.motion_network.{motionBlocks - 1}.weight"] = R([motionVec, style]); w[$"motion_encoder.motion_network.{motionBlocks - 1}.bias"] = R([motionVec]);

        w["face_encoder.conv1_local.weight"] = R([faceHidden * faceHeads, motionOut, 3]); w["face_encoder.conv1_local.bias"] = R([faceHidden * faceHeads]);
        w["face_encoder.conv2.weight"] = R([faceHidden, faceHidden, 3]); w["face_encoder.conv2.bias"] = R([faceHidden]);
        w["face_encoder.conv3.weight"] = R([faceHidden, faceHidden, 3]); w["face_encoder.conv3.bias"] = R([faceHidden]);
        w["face_encoder.out_proj.weight"] = R([dim, faceHidden]); w["face_encoder.out_proj.bias"] = R([dim]);
        w["face_encoder.padding_tokens"] = R([1, 1, 1, dim]);

        int numAdapters = c.NumLayers / inject;
        for (int i = 0; i < numAdapters; i++)
        {
            string p = $"face_adapter.{i}";
            w[$"{p}.to_q.weight"] = R([dim, dim]); w[$"{p}.to_q.bias"] = R([dim]);
            w[$"{p}.to_k.weight"] = R([dim, dim]); w[$"{p}.to_k.bias"] = R([dim]);
            w[$"{p}.to_v.weight"] = R([dim, dim]); w[$"{p}.to_v.bias"] = R([dim]);
            w[$"{p}.to_out.weight"] = R([dim, dim]); w[$"{p}.to_out.bias"] = R([dim]);
            w[$"{p}.norm_q.weight"] = R([c.HeadDim]); w[$"{p}.norm_k.weight"] = R([c.HeadDim]);
        }
        return w;
    }

    private static int _seed = 950;
    private static Tensor R(int[] dims)
    {
        long[] d = Array.ConvertAll(dims, x => (long)x);
        Tensor t = new Tensor(new TensorShape(d), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(_seed++);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 0.1 - 0.05);
        return t;
    }

    private static Tensor RandRows(int rows, int cols, int seed)
    {
        Tensor t = new Tensor(new TensorShape(rows, cols), DType.F32);
        float* p = (float*)t.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < t.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor Rand5d(int b, int c, int t, int h, int w, int seed)
    {
        Tensor x = new Tensor(new TensorShape([(long)b, c, t, h, w]), DType.F32);
        float* p = (float*)x.DataPointer;
        Random rng = new(seed);
        for (long i = 0; i < x.Shape.ElementCount; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
