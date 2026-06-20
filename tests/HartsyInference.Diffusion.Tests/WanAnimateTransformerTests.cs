using Xunit;
using HartsyInference.Core.Tensors;
using HartsyInference.Cpu;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Tests.Common;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Tiny-config end-to-end smoke test for the Wan-Animate DiT (<see cref="WanAnimateTransformer"/>): exercises
/// the pose patch-embed addition, the motion-encoder → face-encoder → face-adapter pathway, and the base block loop,
/// producing a finite, correctly-shaped velocity. Numerics validation-pending.</summary>
public unsafe class WanAnimateTransformerTests
{
    [Fact]
    public void Forward_TinyConfig_FaceAndPoseConditioningProducesLatentShape()
    {
        CpuBackend backend = new();
        WanVideoConfig cfg = new()
        {
            NumHeads = 2, HeadDim = 12, InChannels = 8, OutChannels = 8,
            TextDim = 16, FreqDim = 16, FfnDim = 32, NumLayers = 5, PatchSize = (1, 2, 2),
        };
        const int poseCh = 8, motionSize = 16, motionStyle = 8, motionVec = 4, motionOut = 8, motionBlocks = 2;
        const int faceHidden = 16, faceHeads = 2, inject = 5;
        Dictionary<int, int> motionChannels = new() { [4] = 8, [8] = 8, [16] = 8 };

        WanAnimateTransformer transformer = new(cfg, poseLatentChannels: poseCh, motionEncoderSize: motionSize,
            motionDim: motionOut, faceHiddenDim: faceHidden, faceNumHeads: faceHeads, injectFaceLatentsBlocks: inject,
            motionChannelSizes: motionChannels, motionVecDim: motionVec, motionBlocks: motionBlocks);
        transformer.LoadWeights(BuildWeights(cfg, poseCh, motionSize, motionStyle, motionVec, motionOut, motionBlocks,
            motionChannels, faceHidden, faceHeads, inject));

        Tensor latent = Rand5d(1, cfg.InChannels, 2, 4, 4, seed: 61);          // [1,8,2,4,4] → S=8 tokens
        Tensor pose = Rand5d(1, poseCh, 1, 4, 4, seed: 62);                     // [1,8,T-1=1,4,4]
        Tensor face = Rand5d(1, 3, 4, motionSize, motionSize, seed: 63);       // [1,3,4,16,16] → motion T=2 (divides S=8)
        Tensor encoder = RandRows(3, cfg.TextDim, seed: 64);

        Tensor outVel = transformer.Forward(backend, latent, pose, face, encoder, timestep: 0.5f);

        Assert.Equal(cfg.OutChannels, (int)outVel.Shape[1]);
        Assert.Equal(2, (int)outVel.Shape[2]);
        Assert.Equal(4, (int)outVel.Shape[3]);
        float* p = (float*)outVel.DataPointer;
        for (long i = 0; i < outVel.Shape.ElementCount; i++) Assert.True(float.IsFinite(p[i]), $"non-finite at {i}");
    }

    private static Dictionary<string, Tensor> BuildWeights(WanVideoConfig c, int poseCh, int motionSize, int style,
        int motionVec, int motionOut, int motionBlocks, Dictionary<int, int> motionChannels, int faceHidden, int faceHeads, int inject)
    {
        Dictionary<string, Tensor> w = WanSyntheticWeights.BuildTransformer(c);
        int dim = c.InnerDim;
        // pose patch embed
        w["pose_patch_embedding.weight"] = R([dim, poseCh, c.PatchSize.T, c.PatchSize.H, c.PatchSize.W]);
        w["pose_patch_embedding.bias"] = R([dim]);

        // motion encoder
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

        // face encoder
        w["face_encoder.conv1_local.weight"] = R([faceHidden * faceHeads, motionOut, 3]); w["face_encoder.conv1_local.bias"] = R([faceHidden * faceHeads]);
        w["face_encoder.conv2.weight"] = R([faceHidden, faceHidden, 3]); w["face_encoder.conv2.bias"] = R([faceHidden]);
        w["face_encoder.conv3.weight"] = R([faceHidden, faceHidden, 3]); w["face_encoder.conv3.bias"] = R([faceHidden]);
        w["face_encoder.out_proj.weight"] = R([dim, faceHidden]); w["face_encoder.out_proj.bias"] = R([dim]);
        w["face_encoder.padding_tokens"] = R([1, 1, 1, dim]);

        // face adapter blocks
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

    private static int _seed = 900;
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
