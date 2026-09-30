using System.Globalization;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text.Json;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools;

/// <summary>An <see cref="IToolHandler"/> over a C# delegate: binds the model's JSON arguments to the delegate's parameters by name (exact, then case-insensitive), converts scalars and enums, fills optional parameters from their defaults, invokes, and renders the return value (string, Task/ValueTask of string, or any convertible value) as the tool result.</summary>
internal sealed class DelegateToolHandler : IToolHandler
{
    private readonly Delegate _method;
    private readonly ParameterInfo[] _parameters;
    private readonly ToolDefinition _definition;

    /// <summary>Creates the handler; the schema is derived once through <see cref="ToolSchema.FromDelegate"/>.</summary>
    public DelegateToolHandler(string name, Delegate method, string? description)
    {
        _definition = ToolSchema.FromDelegate(name, method, description);
        _method = method;
        _parameters = method.Method.GetParameters();
    }

    /// <inheritdoc/>
    public string Name => _definition.Name;

    /// <inheritdoc/>
    public string Description => _definition.Description;

    /// <inheritdoc/>
    public string JsonSchema => _definition.JsonSchema;

    /// <summary>The definition the registry offers to the model.</summary>
    public ToolDefinition Definition => _definition;

    /// <inheritdoc/>
    public async Task<string> InvokeAsync(string argumentsJson, CancellationToken cancel)
    {
        object?[] arguments = Bind(argumentsJson, cancel);
        object? result;
        try
        {
            result = _method.DynamicInvoke(arguments);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
        return await RenderAsync(result).ConfigureAwait(false);
    }

    private object?[] Bind(string argumentsJson, CancellationToken cancel)
    {
        object?[] values = new object?[_parameters.Length];
        using JsonDocument doc = JsonDocument.Parse(string.IsNullOrWhiteSpace(argumentsJson) ? "{}" : argumentsJson);
        JsonElement root = doc.RootElement;
        if (root.ValueKind != JsonValueKind.Object)
            throw new ArgumentException($"Tool '{Name}' expects a JSON object of arguments, got {root.ValueKind}.");
        for (int i = 0; i < _parameters.Length; i++)
        {
            ParameterInfo parameter = _parameters[i];
            if (ToolSchema.IsSkipped(parameter))
            {
                values[i] = cancel;
                continue;
            }
            string parameterName = parameter.Name!;
            if (!TryFind(root, parameterName, out JsonElement element) || element.ValueKind == JsonValueKind.Null)
            {
                if (parameter.HasDefaultValue) values[i] = parameter.DefaultValue;
                else if (Nullable.GetUnderlyingType(parameter.ParameterType) is not null) values[i] = null;
                else throw new ArgumentException($"Tool '{Name}' is missing the required argument '{parameterName}'.");
                continue;
            }
            values[i] = Convert(element, ToolSchema.EffectiveType(parameter.ParameterType), parameterName);
        }
        return values;
    }

    private static bool TryFind(JsonElement root, string name, out JsonElement element)
    {
        if (root.TryGetProperty(name, out element)) return true;
        foreach (JsonProperty property in root.EnumerateObject())
        {
            if (string.Equals(property.Name, name, StringComparison.OrdinalIgnoreCase))
            {
                element = property.Value;
                return true;
            }
        }
        return false;
    }

    private object Convert(JsonElement element, Type type, string parameterName)
    {
        try
        {
            if (type.IsEnum)
            {
                return element.ValueKind == JsonValueKind.Number
                    ? Enum.ToObject(type, element.GetInt64())
                    : Enum.Parse(type, element.GetString() ?? "", ignoreCase: true);
            }
            switch (Type.GetTypeCode(type))
            {
                case TypeCode.String:
                    return element.ValueKind == JsonValueKind.String ? element.GetString()! : element.GetRawText();
                case TypeCode.Boolean:
                    return element.ValueKind == JsonValueKind.String
                        ? bool.Parse(element.GetString()!)
                        : element.GetBoolean();
                case TypeCode.Byte or TypeCode.SByte or TypeCode.Int16 or TypeCode.UInt16 or TypeCode.Int32 or TypeCode.UInt32
                    or TypeCode.Int64 or TypeCode.UInt64:
                    long integer = element.ValueKind == JsonValueKind.String
                        ? long.Parse(element.GetString()!, NumberStyles.Integer, CultureInfo.InvariantCulture)
                        : element.GetInt64();
                    return System.Convert.ChangeType(integer, type, CultureInfo.InvariantCulture);
                case TypeCode.Single or TypeCode.Double or TypeCode.Decimal:
                    double number = element.ValueKind == JsonValueKind.String
                        ? double.Parse(element.GetString()!, NumberStyles.Float, CultureInfo.InvariantCulture)
                        : element.GetDouble();
                    return System.Convert.ChangeType(number, type, CultureInfo.InvariantCulture);
                default:
                    throw new NotSupportedException($"Unsupported parameter type {type.Name}.");
            }
        }
        catch (Exception ex) when (ex is FormatException or InvalidOperationException or OverflowException or ArgumentException)
        {
            throw new ArgumentException($"Tool '{Name}' argument '{parameterName}' does not convert to {type.Name}: {ex.Message}", ex);
        }
    }

    private static async Task<string> RenderAsync(object? result)
    {
        switch (result)
        {
            case null:
                return "";
            case string text:
                return text;
            case Task<string> textTask:
                return await textTask.ConfigureAwait(false);
            case ValueTask<string> textValueTask:
                return await textValueTask.ConfigureAwait(false);
            case ValueTask valueTask:
                await valueTask.ConfigureAwait(false);
                return "";
            case Task task:
                await task.ConfigureAwait(false);
                return Render(TaskResult(task));
            default:
                if (AsTask(result) is { } boxed)
                {
                    await boxed.ConfigureAwait(false);
                    return Render(TaskResult(boxed));
                }
                return Render(result);
        }
    }

    /// <summary>A <c>ValueTask&lt;T&gt;</c> result as its <see cref="Task"/>, or null for any other value.</summary>
    private static Task? AsTask(object value)
    {
        Type type = value.GetType();
        if (!type.IsGenericType || type.GetGenericTypeDefinition() != typeof(ValueTask<>)) return null;
        return (Task?)type.GetMethod("AsTask", Type.EmptyTypes)?.Invoke(value, null);
    }

    /// <summary>The <c>Result</c> of a completed <c>Task&lt;T&gt;</c>, or null for a plain <see cref="Task"/> (whose runtime box is a <c>Task&lt;VoidTaskResult&gt;</c>).</summary>
    private static object? TaskResult(Task task)
    {
        Type type = task.GetType();
        PropertyInfo? property = type.IsGenericType ? type.GetProperty("Result") : null;
        if (property is null || property.PropertyType.Name == "VoidTaskResult") return null;
        return property.GetValue(task);
    }

    private static string Render(object? value)
        => value is null ? "" : System.Convert.ToString(value, CultureInfo.InvariantCulture) ?? "";
}
