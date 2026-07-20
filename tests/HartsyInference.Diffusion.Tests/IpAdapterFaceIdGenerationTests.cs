using System.Diagnostics;
using Xunit;
using Xunit.Abstractions;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Cuda;
using HartsyInference.Diffusion.Adapters;
using HartsyInference.Diffusion.Models.Denoisers;
using HartsyInference.Diffusion.Models.TextEncoders;
using HartsyInference.Diffusion.Models.Vae;
using HartsyInference.Diffusion.Pipelines;
using HartsyInference.Diffusion.Requests;
using HartsyInference.Diffusion.Utilities;
using HartsyInference.ModelAssets.CheckpointConverters;
using HartsyInference.ModelAssets.Lora;
using HartsyInference.ModelAssets.SafeTensors;
using HartsyInference.Tests.Common;
using HartsyInference.ModelAssets.Tokenizers;
using HartsyInference.Vision.Codec;
using HartsyInference.Vision.Detection;
using HartsyInference.Vision.Face;

namespace HartsyInference.Diffusion.Tests;

/// <summary>Real-weight end-to-end IP-Adapter <b>FaceID</b> generation: SDXL base + FaceID .bin (ArcFace-embed
/// MLP projection + K/V) + the companion rank-128 UNet LoRA, driven by a real portrait photo through the full
/// detection→alignment→embedding chain (YOLO11-pose keypoints → ArcFace 112×112 similarity alignment → IR-50
/// embedding). Asserts a coherent non-black output and reports the ArcFace cosine between the reference identity
/// and the face detected in the generated image. Skips cleanly when any asset is missing. GPU (CUDA) only —
/// run with CUDA_VISIBLE_DEVICES pinned to a free device.</summary>
[Trait("Category", "Integration")]
public class IpAdapterFaceIdGenerationTests
{
    private readonly ITestOutputHelper _output;

    public IpAdapterFaceIdGenerationTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void Gpu_Sdxl_FaceId_PortraitIdentity()
    {
        string sdxlPath = TestPaths.Sdxl.SingleFile;
        string faceIdPath = TestPaths.IpAdapter.SdxlFaceId;
        string loraPath = TestPaths.IpAdapter.SdxlFaceIdLora;
        string arcFacePath = TestPaths.IpAdapter.ArcFace;
        string posePath = TestPaths.IpAdapter.PoseWeights;
        string portraitPath = TestPaths.IpAdapter.PortraitImage;
        foreach ((string what, string path) in new[]
        {
            ("SDXL checkpoint", sdxlPath), ("FaceID .bin", faceIdPath), ("FaceID LoRA", loraPath),
            ("ArcFace weights", arcFacePath), ("pose weights", posePath), ("portrait image", portraitPath),
            ("CLIP tokenizer", TestPaths.Tokenizers.ClipVocab),
        })
        {
            if (!File.Exists(path)) { _output.WriteLine($"SKIPPED: {what} not found: {path}"); return; }
        }
        string ptxDir = Path.Combine(Path.GetDirectoryName(typeof(IpAdapterFaceIdGenerationTests).Assembly.Location)!, "Ptx");
        if (!Directory.Exists(ptxDir)) { _output.WriteLine($"SKIPPED: PTX directory not found: {ptxDir}"); return; }

        using CudaBackend backend = new(deviceOrdinal: 0, ptxDir: ptxDir);
        _output.WriteLine($"CUDA device: {backend.Capabilities.Name}");

        // 1. Reference identity: portrait → pose keypoints → aligned 112×112 crop → normed ArcFace embedding.
        (byte[] portraitRgb, int pw, int ph) = PngDecoder.DecodeFromFile(portraitPath);
        using SafeTensorsLoader arcLoader = new();
        arcLoader.Load(arcFacePath);
        ArcFaceModel arcFace = new();
        arcFace.LoadWeights(arcLoader.GetAllTensors());

        using YoloPosePipeline pose = new(backend, YoloConfig.YoloV11nPose, posePath, inputSize: 640);
        Tensor? refEmbedMaybe = EmbedLargestFace(backend, pose, arcFace, portraitRgb, pw, ph, "reference");
        Assert.NotNull(refEmbedMaybe);
        Tensor refEmbed = refEmbedMaybe;

        // 2. SDXL + FaceID adapter + companion LoRA.
        Stopwatch sw = Stopwatch.StartNew();
        (SdxlCheckpointConverter.ConvertedWeights converted, SafeTensorsLoader loader) =
            SdxlCheckpointConverter.LoadAndConvert(sdxlPath);
        _output.WriteLine($"Checkpoint loaded in {sw.ElapsedMilliseconds}ms");

        using (loader)
        {
            Dictionary<string, Tensor> clipLF32 = CastWeights(converted.ClipL, DType.F32);
            Dictionary<string, Tensor> clipGF32 = CastWeights(converted.ClipG, DType.F32);
            Dictionary<string, Tensor> unetF16 = CastWeights(converted.UNet, DType.F16);
            Dictionary<string, Tensor> vaeBf16 = CastWeights(converted.Vae, DType.BF16);

            using LoraStack loraStack = new();
            loraStack.AddFromPath(loraPath, strength: 1.0f);
            int merged = loraStack.ApplyToWeights(backend, unetWeights: unetF16, clipLWeights: clipLF32, clipGWeights: clipGF32);
            _output.WriteLine($"FaceID LoRA merged into {merged} weights.");
            Assert.True(merged > 0, "FaceID companion LoRA merged 0 weights — kohya SDXL mapping regressed.");

            using ClipTokenizer tokenizer = new(TestPaths.Tokenizers.ClipVocab, TestPaths.Tokenizers.ClipMerges);
            string prompt = "professional portrait photo of a person in a library, warm light, detailed, sharp focus";
            string negPrompt = "blurry, low quality, deformed";
            int[] promptTokens = tokenizer.Encode(prompt);
            int[] negTokens = tokenizer.Encode(negPrompt);
            int promptEos = ClipTokenizer.FindEosPosition(promptTokens);
            int negEos = ClipTokenizer.FindEosPosition(negTokens);

            ClipTextEncoder clipL = new(ClipTextEncoderConfig.SdxlClipL);
            clipL.LoadWeights(clipLF32, "text_model");
            ClipTextEncoder clipG = new(ClipTextEncoderConfig.SdxlClipG);
            clipG.LoadWeights(clipGF32, "text_model");
            UNet unet = new(UNetConfig.SdxlBase);
            unet.LoadWeights(unetF16);
            VaeDecoder vae = new(VaeConfig.Sdxl);
            vae.LoadWeights(vaeBf16);

            using IpAdapterFile ipaFile = IpAdapterLoader.Load(faceIdPath);
            Assert.Equal(IpAdapterBaseModel.Sdxl, ipaFile.BaseModel);
            Assert.True(ipaFile.Config.IsFaceId);
            using IpAdapter ipa = new(ipaFile.Config);
            ipa.LoadWeights(ipaFile.Weights);
            _output.WriteLine($"FaceID: tokens={ipa.NumImageTokens}, layers={ipa.CrossAttentionLayerCount}");

            using Tensor imageTokens = ipa.ProjectImage(backend, refEmbed);
            AssertFinite("image_tokens", imageTokens);

            IpAdapterConditioning conditioning = new()
            {
                Adapter = ipa,
                ImageTokens = imageTokens,
                Scale = 0.8f,
            };

            using SdxlPipeline pipeline = new(backend, clipL, clipG, unet, vae);
            TextToImageRequest request = new()
            {
                Prompt = prompt,
                NegativePrompt = negPrompt,
                Width = 768,
                Height = 768,
                Steps = 14,
                CfgScale = 6.0f,
                Seed = 42,
            };

            sw.Restart();
            (byte[] rgbData, int width, int height, _) = pipeline.GenerateFromTokens(
                promptTokens, negTokens, promptTokens, negTokens, promptEos, negEos, request,
                progress => _output.WriteLine($"  step {progress.Step}/{progress.TotalSteps} ({progress.ElapsedMs:F0}ms)"),
                ipAdapters: [conditioning]);
            _output.WriteLine($"FaceID generation done in {sw.Elapsed.TotalSeconds:F1}s");

            (double mean, double std) = ByteStats(rgbData);
            _output.WriteLine($"FaceID output: mean={mean:F2}, std={std:F2}");
            Directory.CreateDirectory(TestPaths.OutputDir);
            string outPath = Path.Combine(TestPaths.OutputDir, $"sdxl_faceid_{DateTime.Now:yyyyMMdd_HHmmss}.bmp");
            ImagePostProcessor.SaveBmp(outPath, rgbData, width, height);
            _output.WriteLine($"Saved: {outPath}");
            Assert.True(std > 15, $"FaceID output collapsed (std={std:F2}) — image is flat/black.");
            Assert.True(mean > 10 && mean < 245, $"FaceID output mean={mean:F2} out of range.");

            // 3. Identity consistency: embed the generated image's face and compare to the reference.
            //    Unrelated identities score ~0; FaceID at scale 0.8 + LoRA should land well above that.
            Tensor? genEmbed = EmbedLargestFace(backend, pose, arcFace, rgbData, width, height, "generated");
            if (genEmbed is not null)
            {
                float* rp = (float*)refEmbed.DataPointer;
                float* gp = (float*)genEmbed.DataPointer;
                double cos = 0;
                for (int d = 0; d < ArcFaceModel.EmbeddingDim; d++) cos += (double)rp[d] * gp[d];
                _output.WriteLine($"Identity cosine (reference vs generated): {cos:F4}");
                genEmbed.Dispose();
                Assert.True(cos > 0.1, $"Generated face does not resemble the reference identity (cosine={cos:F4}).");
            }
            else
            {
                _output.WriteLine("WARNING: no face keypoints detected in the generated image — identity check skipped.");
            }
            refEmbed.Dispose();
        }
    }

    /// <summary>Detects the person with the widest eye span, aligns to the ArcFace template, and returns the
    /// L2-normalized <c>[1,512]</c> identity embedding (null when no usable keypoints are found).</summary>
    private Tensor? EmbedLargestFace(IBackend backend, YoloPosePipeline pose, ArcFaceModel arcFace,
        byte[] rgb, int width, int height, string tag)
    {
        IReadOnlyList<PoseDetection> people = pose.Detect(rgb, width, height);
        float bestEyeDist = -1f;
        float[] bestPoints = null!;
        foreach (PoseDetection person in people)
        {
            if (!FaceAlignment.TryGetAlignmentPoints(person, visThreshold: 0.3f, out float[] pts)) continue;
            float dx = pts[2] - pts[0], dy = pts[3] - pts[1];
            float eyeDist = MathF.Sqrt(dx * dx + dy * dy);
            if (eyeDist > bestEyeDist) { bestEyeDist = eyeDist; bestPoints = pts; }
        }
        _output.WriteLine($"{tag}: {people.Count} detections, best eye span={bestEyeDist:F1}px");
        if (bestEyeDist <= 0f) return null;

        byte[] aligned = FaceAlignment.AlignToTemplate(rgb, width, height, bestPoints);
        Tensor input = ArcFaceModel.PreprocessAligned(aligned);
        try
        {
            return arcFace.EmbedNormalized(backend, input);
        }
        finally
        {
            input.Dispose();
        }
    }

    private static unsafe void AssertFinite(string name, Tensor t)
    {
        float* p = (float*)t.DataPointer;
        for (long i = 0; i < t.ElementCount; i++)
        {
            if (!float.IsFinite(p[i]))
                throw new Xunit.Sdk.XunitException($"{name} contains non-finite value at {i}");
        }
    }

    private static (double mean, double std) ByteStats(byte[] data)
    {
        double sum = 0;
        foreach (byte b in data) sum += b;
        double mean = sum / data.Length;
        double var = 0;
        foreach (byte b in data) { double d = b - mean; var += d * d; }
        return (mean, Math.Sqrt(var / data.Length));
    }

    private static Dictionary<string, Tensor> CastWeights(Dictionary<string, Tensor> weights, DType dtype)
    {
        Dictionary<string, Tensor> result = new(weights.Count);
        foreach ((string key, Tensor tensor) in weights)
        {
            result[key] = tensor.DType == dtype ? tensor : tensor.CastTo(dtype);
        }
        return result;
    }
}
