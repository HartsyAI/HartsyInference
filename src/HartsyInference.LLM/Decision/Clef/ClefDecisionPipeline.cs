using System.Text.Json;
using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.LLM.Ssm;
using HartsyInference.ModelAssets.Tokenizers;

namespace HartsyInference.LLM.Decision.Clef;

/// <summary>Cloudflare Clef end to end: tokenize the record, run the Qwen3.5 trunk, score every option with the joint schema
/// head, and answer in the Jev / SystemOne response shape (<c>systemone</c> in <c>joint_schema_model.py</c>).</summary>
public sealed unsafe class ClefDecisionPipeline : IDisposable
{
    private readonly Qwen35Model _backbone;
    private readonly ClefJointHead _head;
    private readonly ClefRecordEncoder _encoder;
    private readonly Tensor _outputEmbedding;
    private readonly int _hiddenSize;
    private readonly int _maxLength;
    private int _disposed;

    private ClefDecisionPipeline(Qwen35Model backbone, ClefJointHead head, ClefRecordEncoder encoder, Tensor outputEmbedding,
        int hiddenSize, int maxLength)
    {
        _backbone = backbone;
        _head = head;
        _encoder = encoder;
        _outputEmbedding = outputEmbedding;
        _hiddenSize = hiddenSize;
        _maxLength = maxLength;
    }

    /// <summary>Builds the pipeline from the release's tensors: <paramref name="backbone"/> holds <c>model.language_model.*</c>
    /// and <c>lm_head.weight</c> (vision tensors are ignored), <paramref name="head"/> the joint head's state dict.</summary>
    public static ClefDecisionPipeline Create(IReadOnlyDictionary<string, Tensor> backbone, IReadOnlyDictionary<string, Tensor> head,
        Qwen35HfConfig backboneConfig, ClefJointHeadConfig headConfig, GgufTokenizer tokenizer, int maxLength = ClefRecordEncoder.DefaultMaxLength)
    {
        Qwen35Model trunk = Qwen35Model.FromHuggingFace(backbone, backboneConfig, maxLength);
        ClefJointHead jointHead = new(headConfig);
        jointHead.LoadWeights(head);
        return new ClefDecisionPipeline(trunk, jointHead, new ClefRecordEncoder(tokenizer), backbone["lm_head.weight"],
            backboneConfig.HiddenSize, maxLength);
    }

    /// <summary>Answers a <c>/v1/systemone</c> request body with the matching response body.</summary>
    public string Decide(IBackend backend, string requestJson)
    {
        using JsonDocument doc = JsonDocument.Parse(requestJson);
        JsonElement request = doc.RootElement;
        if (request.ValueKind != JsonValueKind.Object || !request.TryGetProperty("model", out JsonElement model)
            || model.ValueKind != JsonValueKind.String || !request.TryGetProperty("state", out _))
        {
            throw new ArgumentException("model and state are required");
        }
        if (!request.TryGetProperty("questions", out JsonElement questions) || questions.ValueKind != JsonValueKind.Object
            || !questions.EnumerateObject().Any())
        {
            throw new ArgumentException("at least one question is required");
        }
        foreach (JsonProperty q in questions.EnumerateObject())
        {
            string? type = q.Value.TryGetProperty("type", out JsonElement t) && t.ValueKind == JsonValueKind.String ? t.GetString() : null;
            if (type is not ("noul" or "choice" or "score"))
            {
                throw new ArgumentException($"{q.Name}: type must be noul, choice, or score");
            }
            if (type != "noul" && (!q.Value.TryGetProperty("criteria", out JsonElement c) || IsEmpty(c)))
            {
                throw new ArgumentException($"{q.Name}: criteria must not be empty");
            }
        }

        ClefEncodedRecord record = _encoder.Encode(request, _maxLength);
        float[] hidden = _backbone.ForwardHiddenStates(backend, record.InputIds);
        float[][] logits = _head.Forward(backend, hidden, record.InputIds.Length, record.InputIds, record.Questions, OutputRow);

        using MemoryStream stream = new();
        using (Utf8JsonWriter w = new(stream))
        {
            w.WriteStartObject();
            w.WriteString("model", model.GetString());
            w.WritePropertyName("answers");
            w.WriteStartObject();
            for (int i = 0; i < record.Questions.Length; i++)
            {
                ClefQuestionSpans span = record.Questions[i];
                w.WritePropertyName(span.QuestionId);
                WriteAnswer(w, questions.GetProperty(span.QuestionId), span.OptionIds, Softmax(logits[i]));
            }
            w.WriteEndObject();
            w.WritePropertyName("usage");
            w.WriteStartObject();
            w.WriteNumber("input_tokens", record.InputIds.Length);
            w.WriteNumber("output_tokens", 0);
            w.WriteEndObject();
            w.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }

    /// <summary>Releases the trunk and head.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }
        _backbone.Dispose();
        _head.Dispose();
    }

    private float[] OutputRow(int token)
    {
        float[] row = new float[_hiddenSize];
        fixed (float* dst = row)
        {
            Qwen35Model.CopyEmbeddingRow(_outputEmbedding, token, _hiddenSize, dst);
        }
        return row;
    }

    private static bool IsEmpty(JsonElement criteria) => criteria.ValueKind switch
    {
        JsonValueKind.Object => !criteria.EnumerateObject().Any(),
        JsonValueKind.Array => criteria.GetArrayLength() == 0,
        JsonValueKind.Null => true,
        JsonValueKind.String => criteria.GetString() is { Length: 0 },
        _ => false,
    };

    private static double[] Softmax(float[] logits)
    {
        double max = logits.Max();
        double[] p = logits.Select(l => Math.Exp(l - max)).ToArray();
        double sum = p.Sum();
        return p.Select(v => v / sum).ToArray();
    }

    private static double Round(double value) => Math.Round(value, 4, MidpointRounding.ToEven);

    private static void WriteAnswer(Utf8JsonWriter w, JsonElement question, string[] optionIds, double[] probabilities)
    {
        string type = question.GetProperty("type").GetString()!;
        Dictionary<string, double> byOption = new(StringComparer.Ordinal);
        for (int i = 0; i < optionIds.Length; i++)
        {
            byOption[optionIds[i]] = probabilities[i];
        }
        w.WriteStartObject();
        w.WriteString("type", type);
        if (type == "noul")
        {
            w.WriteNumber("noul", Round(byOption["true"]));
        }
        else if (type == "choice")
        {
            string[] options = [.. question.GetProperty("criteria").EnumerateObject().Select(p => p.Name)];
            string choice = options[0];
            foreach (string option in options)
            {
                if (byOption[option] > byOption[choice])
                {
                    choice = option;
                }
            }
            w.WriteString("choice", choice);
            w.WriteNumber("confidence", Round(byOption[choice]));
            WriteProbabilities(w, options, byOption);
        }
        else
        {
            JsonElement criteria = question.GetProperty("criteria");
            string[] levels = [.. Enumerable.Range(0, criteria.GetArrayLength()).Select(i => i.ToString(System.Globalization.CultureInfo.InvariantCulture))];
            double expected = 0, best = 0;
            for (int i = 0; i < levels.Length; i++)
            {
                expected += i * byOption[levels[i]];
                best = Math.Max(best, byOption[levels[i]]);
            }
            w.WriteNumber("score", Round(expected));
            w.WriteNumber("confidence", Round(best));
            w.WritePropertyName("legend");
            w.WriteStartObject();
            for (int i = 0; i < levels.Length; i++)
            {
                w.WritePropertyName(levels[i]);
                criteria[i].WriteTo(w);
            }
            w.WriteEndObject();
            WriteProbabilities(w, levels, byOption);
        }
        w.WriteEndObject();
    }

    private static void WriteProbabilities(Utf8JsonWriter w, string[] keys, Dictionary<string, double> byOption)
    {
        w.WritePropertyName("probabilities");
        w.WriteStartObject();
        foreach (string key in keys)
        {
            w.WriteNumber(key, Round(byOption[key]));
        }
        w.WriteEndObject();
    }
}
