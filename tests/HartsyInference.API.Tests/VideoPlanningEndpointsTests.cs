using System.Net;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using HartsyInference.Engine;
using HartsyInference.Engine.Audio;
using HartsyInference.Engine.Dispatch;
using HartsyInference.Engine.Planning;
using HartsyInference.Engine.Recipes;
using HartsyInference.Engine.Requests;
using HartsyInference.Engine.Services;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Xunit;

namespace HartsyInference.API.Tests;

/// <summary>HTTP contract coverage for native video planning, pre-SSE validation, enum/casing, and body limits.</summary>
public sealed class VideoPlanningEndpointsTests : IClassFixture<WebApplicationFactory<Program>>, IDisposable
{
    private readonly WebApplicationFactory<Program> _factory;
    private readonly string _tempDirectory;

    public VideoPlanningEndpointsTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder => builder.UseSetting("HartsyInference:Backend", "cpu"));
        _tempDirectory = Path.Combine(Path.GetTempPath(), $"hartsy-api-video-plan-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDirectory);
    }

    [Fact]
    public async Task Plan_UsesCamelCaseAndStringEnums()
    {
        string checkpoint = WriteH3Folder();
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/video/plan", new
        {
            model = "minimax-h3",
            modelPath = checkpoint,
            request = new { prompt = "a lighthouse", audioFlowShift = 2.75 },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.True(body.TryGetProperty("effectiveSettings", out JsonElement settings));
        Assert.Equal(2.75, settings.GetProperty("audioFlowShift").GetDouble(), 3);
        Assert.Equal("Unknown", body.GetProperty("profile").GetProperty("task").GetString());
        Assert.Equal("Dense", body.GetProperty("profile").GetProperty("attention").GetString());
    }

    [Fact]
    public async Task Plan_DoesNotExposeServerPathsHashesOrRawMetadata()
    {
        string checkpoint = WriteH3Folder();
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/video/plan", new
        {
            model = "minimax-h3",
            modelPath = checkpoint,
            request = new { prompt = "safe projection" },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string json = await response.Content.ReadAsStringAsync();
        Assert.DoesNotContain(checkpoint, json, StringComparison.Ordinal);
        Assert.DoesNotContain("localPath", json, StringComparison.Ordinal);
        Assert.DoesNotContain("componentPaths", json, StringComparison.Ordinal);
        Assert.DoesNotContain("artifactHashes", json, StringComparison.Ordinal);
        Assert.DoesNotContain("artifactMetadata", json, StringComparison.Ordinal);
        Assert.DoesNotContain("checkpointMetadata", json, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("/v1/native/video/plan")]
    [InlineData("/v1/native/video/stream")]
    public async Task InvalidProfile_ReturnsTyped422BeforeStreaming(string route)
    {
        string checkpoint = WriteH3Folder();
        using HttpClient client = _factory.CreateClient();
        HttpResponseMessage response = await client.PostAsJsonAsync(route, new
        {
            model = "minimax-h3",
            modelPath = checkpoint,
            modelProfile = "not-the-detected-profile",
            request = new { prompt = "test" },
        });

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        Assert.DoesNotContain("text/event-stream", response.Content.Headers.ContentType?.MediaType ?? "");
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(422, body.GetProperty("status").GetInt32());
        Assert.Equal("https://hartsy.ai/problems/video-plan-invalid", body.GetProperty("type").GetString());
        JsonElement issue = body.GetProperty("issues").EnumerateArray()
            .First(item => item.GetProperty("code").GetString() == "video.profile.mismatch");
        Assert.Equal("Error", issue.GetProperty("severity").GetString());
    }

    [Fact]
    public async Task Stream_UnregisteredVideoFamily_ReturnsTyped422BeforeSse()
    {
        using HttpClient client = _factory.CreateClient();
        using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/video/stream", new
        {
            // SDXL is a real, resolvable Engine family but deliberately has no video recipe. This exercises the
            // exact catalog/family resolution path without supplying a checkpoint or constructing any weights.
            model = "sdxl",
            request = new { prompt = "must remain an ordinary HTTP problem" },
        });

        Assert.Equal(HttpStatusCode.UnprocessableEntity, response.StatusCode);
        Assert.DoesNotContain("text/event-stream", response.Content.Headers.ContentType?.MediaType ?? "");
        JsonElement body = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.Equal(422, body.GetProperty("status").GetInt32());
        JsonElement issue = body.GetProperty("issues").EnumerateArray()
            .Single(item => item.GetProperty("code").GetString() == "video.family.unregistered");
        Assert.Equal("Error", issue.GetProperty("severity").GetString());
        Assert.Equal("Requested", issue.GetProperty("field").GetString());
    }

    [Fact]
    public async Task VideoBodyLimit_RejectsBeforeJsonBinding()
    {
        using WebApplicationFactory<Program> limited = _factory.WithWebHostBuilder(builder =>
            builder.UseSetting("HartsyInference:MaxVideoRequestBodyBytes", "64"));
        using HttpClient client = limited.CreateClient();
        using StringContent content = new StringContent(
            JsonSerializer.Serialize(new { model = "x", request = new { prompt = new string('p', 512) } }),
            Encoding.UTF8, "application/json");

        HttpResponseMessage response = await client.PostAsync("/v1/native/video/plan", content);
        Assert.Equal(HttpStatusCode.RequestEntityTooLarge, response.StatusCode);
    }

    [Fact]
    public async Task Stream_OrdersProgressFrameAudioAndCompleteAndIncludesExecutionSummary()
    {
        FakeVideoService fakeVideo = new FakeVideoService();
        using WebApplicationFactory<Program> fakeFactory = _factory.WithWebHostBuilder(builder =>
            builder.ConfigureServices(services =>
            {
                services.RemoveAll<IInferenceEngine>();
                services.AddSingleton<IInferenceEngine>(new FakeVideoEngine(fakeVideo));
            }));
        using HttpClient client = fakeFactory.CreateClient();
        using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/video/stream", new
        {
            model = "fake-video",
            save = false,
            request = new { prompt = "contract test" },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        Assert.Equal("text/event-stream", response.Content.Headers.ContentType?.MediaType);
        string body = await response.Content.ReadAsStringAsync();
        int progress = body.IndexOf("event: progress", StringComparison.Ordinal);
        int frame = body.IndexOf("event: frame", StringComparison.Ordinal);
        int audio = body.IndexOf("event: audio", StringComparison.Ordinal);
        int complete = body.IndexOf("event: complete", StringComparison.Ordinal);
        Assert.True(progress >= 0 && frame > progress && audio > frame && complete > audio, body);
        Assert.Contains("\"index\":0", body, StringComparison.Ordinal);
        Assert.Contains("\"execution\":{\"profileId\":\"fake-h3\"", body, StringComparison.Ordinal);
        Assert.DoesNotContain("event: error", body, StringComparison.Ordinal);
        Assert.Equal(1, fakeVideo.PlanCalls);
        Assert.Equal(1, fakeVideo.PlannedGenerateCalls);
        Assert.Equal(0, fakeVideo.LegacyGenerateCalls);
        Assert.Same(fakeVideo.LastPlan, fakeVideo.LastExecutedPlan);
    }

    [Fact]
    public async Task Stream_GenerationFailureUsesTypedTerminalErrorEvent()
    {
        FakeVideoService fakeVideo = new() { ThrowOnGenerate = true };
        using WebApplicationFactory<Program> fakeFactory = _factory.WithWebHostBuilder(builder =>
            builder.ConfigureServices(services =>
            {
                services.RemoveAll<IInferenceEngine>();
                services.AddSingleton<IInferenceEngine>(new FakeVideoEngine(fakeVideo));
            }));
        using HttpClient client = fakeFactory.CreateClient();
        using HttpResponseMessage response = await client.PostAsJsonAsync("/v1/native/video/stream", new
        {
            model = "fake-video",
            save = false,
            request = new { prompt = "failing contract test" },
        });

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        string body = await response.Content.ReadAsStringAsync();
        Assert.Contains("event: error", body, StringComparison.Ordinal);
        Assert.Contains("\"message\":\"fake generation failure\"", body, StringComparison.Ordinal);
        Assert.DoesNotContain("event: complete", body, StringComparison.Ordinal);
    }

    [Fact]
    public async Task OpenApi_DocumentsNamedVideoSsePayloadsAndTyped422()
    {
        using HttpClient client = _factory.CreateClient();
        JsonElement document = await client.GetFromJsonAsync<JsonElement>("/openapi/v1.json");
        JsonElement paths = document.GetProperty("paths");
        JsonElement responses = paths.GetProperty("/v1/native/video/stream").GetProperty("post")
            .GetProperty("responses");
        JsonElement streamSchema = responses.GetProperty("200").GetProperty("content")
            .GetProperty("text/event-stream").GetProperty("schema");
        Assert.Contains("event: <name>", streamSchema.GetProperty("description").GetString(),
            StringComparison.Ordinal);

        Dictionary<string, string> expectedEvents = new(StringComparer.Ordinal)
        {
            ["progress"] = "StepPreviewPayload",
            ["frame"] = "NativeVideoFrameEvent",
            ["audio"] = "NativeVideoAudioEvent",
            ["complete"] = "NativeVideoCompleteEvent",
            ["error"] = "NativeSseErrorEvent",
        };
        JsonElement[] eventSchemas = streamSchema.GetProperty("oneOf").EnumerateArray().ToArray();
        Assert.Equal(expectedEvents.Count, eventSchemas.Length);
        foreach (JsonElement eventSchema in eventSchemas)
        {
            string eventName = eventSchema.GetProperty("title").GetString()!;
            Assert.True(expectedEvents.Remove(eventName, out string? component), eventName);
            JsonElement[] payloadSchemas = eventSchema.GetProperty("allOf").EnumerateArray().ToArray();
            Assert.Single(payloadSchemas);
            Assert.Equal($"#/components/schemas/{component}", payloadSchemas[0].GetProperty("$ref").GetString());
        }
        Assert.Empty(expectedEvents);

        JsonElement schemas = document.GetProperty("components").GetProperty("schemas");
        AssertProperties(schemas.GetProperty("StepPreviewPayload"), "step", "total", "previewPng",
            "previewFramesPng", "previewWidth", "previewHeight");
        AssertProperties(schemas.GetProperty("NativeVideoFrameEvent"), "index", "png");
        AssertProperties(schemas.GetProperty("NativeVideoAudioEvent"), "sampleRate", "channels", "wav");
        AssertProperties(schemas.GetProperty("NativeVideoCompleteEvent"), "frames", "savedPath", "execution");
        AssertProperties(schemas.GetProperty("NativeSseErrorEvent"), "message");
        JsonElement execution = schemas.GetProperty("NativeVideoCompleteEvent").GetProperty("properties")
            .GetProperty("execution").GetProperty("oneOf")[0];
        Assert.Equal("#/components/schemas/VideoExecutionSummary", execution.GetProperty("$ref").GetString());
        AssertProperties(schemas.GetProperty("VideoExecutionSummary"), "profileId", "task", "acceleration",
            "attention", "width", "height", "frames", "fps", "seed", "steps", "cfgScale", "flowShift",
            "audioFlowShift", "sampler", "scheduler", "executionPath", "componentFormats");

        string[] typedProblemRoutes = ["/v1/native/video/plan", "/v1/native/video/stream"];
        foreach (string route in typedProblemRoutes)
        {
            JsonElement problem = paths.GetProperty(route).GetProperty("post").GetProperty("responses")
                .GetProperty("422").GetProperty("content").GetProperty("application/json").GetProperty("schema");
            Assert.Equal("#/components/schemas/NativeVideoPlanProblem", problem.GetProperty("$ref").GetString());
        }
    }

    private string WriteH3Folder()
    {
        string root = Path.Combine(_tempDirectory, Guid.NewGuid().ToString("N"));
        string transformerDirectory = Path.Combine(root, "transformer");
        string videoVaeDirectory = Path.Combine(root, "video_vae");
        string textEncoderDirectory = Path.Combine(root, "text_encoder");
        Directory.CreateDirectory(transformerDirectory);
        Directory.CreateDirectory(videoVaeDirectory);
        Directory.CreateDirectory(textEncoderDirectory);
        string transformer = Path.Combine(transformerDirectory, "model.safetensors");
        Dictionary<string, object> transformerHeader = new(StringComparer.Ordinal)
        {
            ["__metadata__"] = new Dictionary<string, string> { ["task"] = "fl2va" },
            ["video_patch_proj.weight"] = Descriptor([5376, 96]),
            ["video_patch_proj.bias"] = Descriptor([5376]),
            ["audio_patch_proj.weight"] = Descriptor([5376, 32]),
            ["audio_patch_proj.bias"] = Descriptor([5376]),
            ["condition_proj.weight"] = Descriptor([5376, 5120]),
            ["condition_proj.bias"] = Descriptor([5376]),
            ["rope.inv_freq"] = Descriptor([16]),
            ["time_embedder.proj_in.weight"] = Descriptor([5376, 256]),
            ["time_embedder.proj_in.bias"] = Descriptor([5376]),
            ["time_embedder.proj_out.weight"] = Descriptor([2688, 5376]),
            ["time_embedder.proj_out.bias"] = Descriptor([2688]),
            ["token_refiner.final_norm.weight"] = Descriptor([5376]),
            ["final_layer.norm.weight"] = Descriptor([5376]),
            ["final_layer.adaln_proj.linear.weight"] = Descriptor([10752, 2688]),
            ["final_layer.adaln_proj.linear.bias"] = Descriptor([10752]),
            ["final_layer.video_out.weight"] = Descriptor([96, 5376]),
            ["final_layer.video_out.bias"] = Descriptor([96]),
            ["final_layer.audio_out.weight"] = Descriptor([32, 5376]),
            ["final_layer.audio_out.bias"] = Descriptor([32]),
        };
        for (int i = 0; i < 50; i++)
        {
            AddH3Block(transformerHeader, $"blocks.{i}", includeAdaln: true);
        }
        for (int i = 0; i < 2; i++)
        {
            AddH3Block(transformerHeader, $"token_refiner.blocks.{i}", includeAdaln: false);
        }
        WriteHeader(transformer, transformerHeader);
        WriteHeader(Path.Combine(videoVaeDirectory, "model.safetensors"), BuildVideoVaeHeader());
        WriteHeader(Path.Combine(textEncoderDirectory, "model.safetensors"), BuildTextEncoderHeader());
        return transformer;
    }

    private static void AddH3Block(Dictionary<string, object> header, string block, bool includeAdaln)
    {
        header[block + ".norm1.weight"] = Descriptor([5376]);
        header[block + ".norm2.weight"] = Descriptor([5376]);
        header[block + ".attn.qkv_proj.weight"] = Descriptor([21504, 5376]);
        header[block + ".attn.q_norm.weight"] = Descriptor([128]);
        header[block + ".attn.k_norm.weight"] = Descriptor([128]);
        header[block + ".attn.out_proj.weight"] = Descriptor([5376, 7168]);
        header[block + ".mlp.fc1.weight"] = Descriptor([28672, 5376]);
        header[block + ".mlp.fc2.weight"] = Descriptor([5376, 14336]);
        if (includeAdaln)
        {
            header[block + ".adaln_proj.linear.weight"] = Descriptor([96768, 2688]);
            header[block + ".adaln_proj.linear.bias"] = Descriptor([96768]);
        }
    }

    private static Dictionary<string, object> BuildVideoVaeHeader()
    {
        Dictionary<string, object> header = new(StringComparer.Ordinal)
        {
            ["latents_mean"] = Descriptor([24]),
            ["latents_std"] = Descriptor([24]),
            ["post_quant_conv.weight"] = Descriptor([24, 24, 1, 1, 1]),
            ["post_quant_conv.bias"] = Descriptor([24]),
            ["decoder.x_embedder.weight"] = Descriptor([2048, 24]),
            ["decoder.x_embedder.bias"] = Descriptor([2048]),
            ["decoder.register_tokens"] = Descriptor([1, 4, 2048]),
            ["decoder.norm_out.weight"] = Descriptor([2048]),
            ["decoder.norm_out.bias"] = Descriptor([2048]),
            ["decoder.proj_out.weight"] = Descriptor([3072, 2048]),
            ["decoder.proj_out.bias"] = Descriptor([3072]),
            ["encoder.conv_in.weight"] = Descriptor([128, 3, 3, 3, 3]),
            ["encoder.conv_in.bias"] = Descriptor([128]),
            ["encoder.norm_out.weight"] = Descriptor([1024]),
            ["encoder.norm_out.bias"] = Descriptor([1024]),
            ["encoder.conv_out.weight"] = Descriptor([48, 1024, 3, 3, 3]),
            ["encoder.conv_out.bias"] = Descriptor([48]),
            ["quant_conv.weight"] = Descriptor([48, 48, 1, 1, 1]),
            ["quant_conv.bias"] = Descriptor([48]),
        };
        for (int i = 0; i < 36; i++)
        {
            string block = $"decoder.transformer_blocks.{i}";
            header[block + ".norm1.weight"] = Descriptor([2048]);
            header[block + ".norm2.weight"] = Descriptor([2048]);
            header[block + ".scale1"] = Descriptor([2048]);
            header[block + ".scale2"] = Descriptor([2048]);
            header[block + ".attn.to_qkv.weight"] = Descriptor([6144, 2048]);
            header[block + ".attn.to_qkv.bias"] = Descriptor([6144]);
            header[block + ".attn.to_out.weight"] = Descriptor([2048, 2048]);
            header[block + ".attn.to_out.bias"] = Descriptor([2048]);
            header[block + ".ff.w1.weight"] = Descriptor([16384, 2048]);
            header[block + ".ff.w1.bias"] = Descriptor([16384]);
            header[block + ".ff.w2.weight"] = Descriptor([2048, 8192]);
            header[block + ".ff.w2.bias"] = Descriptor([2048]);
        }
        int[] channels = [128, 256, 256, 512, 512, 1024];
        int inputChannels = 128;
        for (int stage = 0; stage < channels.Length; stage++)
        {
            int outputChannels = channels[stage];
            for (int blockIndex = 0; blockIndex < 2; blockIndex++)
            {
                string block = $"encoder.down.{stage}.block.{blockIndex}";
                int blockInput = blockIndex == 0 ? inputChannels : outputChannels;
                header[block + ".norm1.weight"] = Descriptor([blockInput]);
                header[block + ".norm1.bias"] = Descriptor([blockInput]);
                header[block + ".conv1.weight"] = Descriptor([outputChannels, blockInput, 3, 3, 3]);
                header[block + ".conv1.bias"] = Descriptor([outputChannels]);
                header[block + ".norm2.weight"] = Descriptor([outputChannels]);
                header[block + ".norm2.bias"] = Descriptor([outputChannels]);
                header[block + ".conv2.weight"] = Descriptor([outputChannels, outputChannels, 3, 3, 3]);
                header[block + ".conv2.bias"] = Descriptor([outputChannels]);
                if (blockInput != outputChannels)
                {
                    header[block + ".nin_shortcut.weight"] = Descriptor([outputChannels, blockInput, 1, 1, 1]);
                    header[block + ".nin_shortcut.bias"] = Descriptor([outputChannels]);
                }
            }
            if (stage < 4)
            {
                string downsample = $"encoder.down.{stage}.downsample.conv";
                header[downsample + ".weight"] = Descriptor([outputChannels, outputChannels, 3, 3, 3]);
                header[downsample + ".bias"] = Descriptor([outputChannels]);
            }
            inputChannels = outputChannels;
        }
        return header;
    }

    private static Dictionary<string, object> BuildTextEncoderHeader()
    {
        Dictionary<string, object> header = new(StringComparer.Ordinal)
        {
            ["model.embed_tokens.weight"] = Descriptor([151936, 5120]),
            ["visual.patch_embed.proj.weight"] = Descriptor([1152, 3, 2, 16, 16]),
            ["visual.merger.norm.weight"] = Descriptor([1152]),
            ["visual.merger.linear_fc1.weight"] = Descriptor([4608, 4608]),
            ["visual.merger.linear_fc2.weight"] = Descriptor([5120, 4608]),
        };
        for (int i = 0; i < 50; i++)
        {
            string layer = $"model.layers.{i}";
            header[layer + ".input_layernorm.weight"] = Descriptor([5120]);
            header[layer + ".post_attention_layernorm.weight"] = Descriptor([5120]);
            header[layer + ".self_attn.q_norm.weight"] = Descriptor([128]);
            header[layer + ".self_attn.k_norm.weight"] = Descriptor([128]);
            header[layer + ".self_attn.q_proj.weight"] = Descriptor([8192, 5120]);
            header[layer + ".self_attn.k_proj.weight"] = Descriptor([1024, 5120]);
            header[layer + ".self_attn.v_proj.weight"] = Descriptor([1024, 5120]);
            header[layer + ".self_attn.o_proj.weight"] = Descriptor([5120, 8192]);
            header[layer + ".mlp.gate_proj.weight"] = Descriptor([25600, 5120]);
            header[layer + ".mlp.up_proj.weight"] = Descriptor([25600, 5120]);
            header[layer + ".mlp.down_proj.weight"] = Descriptor([5120, 25600]);
        }
        return header;
    }

    private static void AssertProperties(JsonElement schema, params string[] expected)
    {
        JsonElement properties = schema.GetProperty("properties");
        foreach (string property in expected)
        {
            Assert.True(properties.TryGetProperty(property, out _), $"Schema is missing property '{property}'.");
        }
    }

    private static Dictionary<string, object> Descriptor(long[] shape) => new(StringComparer.Ordinal)
    {
        ["dtype"] = "F32",
        ["shape"] = shape,
        ["data_offsets"] = new long[] { 0, 0 },
    };

    private static void WriteHeader(string path, Dictionary<string, object> header)
    {
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using FileStream stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((long)json.Length));
        stream.Write(json);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDirectory))
        {
            Directory.Delete(_tempDirectory, recursive: true);
        }
    }

    private sealed class FakeVideoEngine(FakeVideoService video) : IInferenceEngine
    {
        private readonly FakeVideoService _video = video;

        public string BackendSelector => "fake";
        public string BackendDescription => "fake video contract backend";
        public IReadOnlyCollection<string> LoadedPipelineKeys => [];
        public IVideoService Video => _video;
        public IVideoPlanningService VideoPlanning => _video;
        public IImagesService Images => throw new NotSupportedException();
        public ITextService Text => throw new NotSupportedException();
        public IMusicService Music => throw new NotSupportedException();
        public ISpeechService Speech => throw new NotSupportedException();
        public ITranscribeService Transcribe => throw new NotSupportedException();
        public IVoiceConversionService VoiceConversion => throw new NotSupportedException();
        public IFxService Fx => throw new NotSupportedException();
        public IModelPrefetchService ModelPrefetch => throw new NotSupportedException();
        public IVisionService Vision => throw new NotSupportedException();
        public IRestoreService Restore => throw new NotSupportedException();
        public IMeshService Mesh => throw new NotSupportedException();
        public IWorldService World => throw new NotSupportedException();
        public IEmbeddingService Embeddings => throw new NotSupportedException();

        public bool IsSupported(Modality modality) => modality == Modality.Video;
        public void SetBackend(string selector) => throw new NotSupportedException();
        public void FreeMemory() { }
        public void Dispose() { }
    }

    private sealed class FakeVideoService : IVideoService, IVideoPlanningService
    {
        private static readonly VideoEffectiveSettings Effective = new()
        {
            Width = 1,
            Height = 1,
            Frames = 1,
            Fps = 25,
            Steps = 4,
            CfgScale = 1,
            FlowShift = 12,
            AudioFlowShift = 3,
            Sampler = "euler",
            Scheduler = "normal",
            Seed = 123,
            ReferenceSizing = VideoReferenceSizing.Native,
            LockedFields = VideoLockedFields.None,
        };

        private static readonly VideoModelProfile Profile = new()
        {
            Id = "fake-h3",
            DisplayName = "Fake H3",
            FamilyId = "minimax-h3",
            Task = VideoTaskFamily.T2Va,
            Acceleration = VideoAccelerationKind.None,
            Attention = VideoAttentionKind.Dense,
            Defaults = VideoDefaults.Standard,
            Features = VideoFeatures.None,
        };

        private int _planCalls;
        private int _plannedGenerateCalls;
        private int _legacyGenerateCalls;
        private VideoPlan? _lastPlan;
        private VideoPlan? _lastExecutedPlan;

        public int PlanCalls => _planCalls;
        public int PlannedGenerateCalls => _plannedGenerateCalls;
        public int LegacyGenerateCalls => _legacyGenerateCalls;
        public VideoPlan? LastPlan => _lastPlan;
        public VideoPlan? LastExecutedPlan => _lastExecutedPlan;
        public bool ThrowOnGenerate { get; init; }

        public Task<VideoPlan> PlanAsync(ModelSpec spec, VideoRequest request, CancellationToken cancel = default)
        {
            _planCalls++;
            _lastPlan = new VideoPlan
            {
                Model = spec,
                Profile = Profile,
                EffectiveSettings = Effective,
                Issues = [],
                CacheIdentity = "fake",
            };
            return Task.FromResult(_lastPlan);
        }

        public Task<VideoGenerationResult> GenerateAsync(ModelSpec spec, VideoRequest request,
            IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
        {
            _legacyGenerateCalls++;
            return Task.FromResult(Result());
        }

        public Task<VideoGenerationResult> GenerateAsync(VideoPlan plan, VideoRequest request,
            IProgress<StepPreview>? progress = null, CancellationToken cancel = default)
        {
            _plannedGenerateCalls++;
            _lastExecutedPlan = plan;
            if (ThrowOnGenerate)
            {
                throw new InvalidOperationException("fake generation failure");
            }
            progress?.Report(new StepPreview { Step = 1, TotalSteps = 4 });
            return Task.FromResult(Result());
        }

        private static VideoGenerationResult Result() => new()
        {
            Frames = [new VideoFrame { Rgb = [1, 2, 3], Width = 1, Height = 1, Index = 0 }],
            Audio = AudioBuffer.FromChannels([[0.25f, -0.25f]], 32_000),
            Fps = 25,
            Execution = new VideoExecutionSummary
            {
                ProfileId = "fake-h3",
                Task = VideoTaskFamily.T2Va,
                Acceleration = VideoAccelerationKind.None,
                Attention = VideoAttentionKind.Dense,
                Width = 1,
                Height = 1,
                Frames = 1,
                Fps = 25,
                Seed = 123,
                Steps = 4,
                CfgScale = 1,
                FlowShift = 12,
                AudioFlowShift = 3,
                Sampler = "euler",
                Scheduler = "normal",
                ExecutionPath = "None",
                ComponentFormats = new Dictionary<string, string> { ["transformer"] = "bf16" },
            },
        };

        public async IAsyncEnumerable<VideoFrame> GenerateFramesAsync(ModelSpec spec, VideoRequest request,
            IProgress<StepPreview>? progress = null,
            [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancel = default)
        {
            await Task.CompletedTask;
            yield break;
        }
    }
}
