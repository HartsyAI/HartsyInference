using System.ComponentModel;
using System.Reflection;
using System.Text;
using System.Text.Json;
using HartsyInference.Engine.Requests;

namespace HartsyInference.Tools;

/// <summary>Builds a <see cref="ToolDefinition"/> from a C# delegate: each parameter becomes a schema property (string, integer, number, boolean, or a string enum listing the enum names), a <see cref="DescriptionAttribute"/> on a parameter or on the method supplies the description, and a parameter with a default value or a nullable type is optional. <see cref="CancellationToken"/> parameters are omitted. Unsupported parameter types fail here, at registration, rather than at call time.</summary>
public static class ToolSchema
{
    /// <summary>The definition for <paramref name="method"/> under <paramref name="name"/>; <paramref name="description"/> defaults to the method's <see cref="DescriptionAttribute"/>, else empty.</summary>
    public static ToolDefinition FromDelegate(string name, Delegate method, string? description = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(method);
        return new ToolDefinition
        {
            Name = name,
            Description = description ?? method.Method.GetCustomAttribute<DescriptionAttribute>()?.Description ?? "",
            JsonSchema = ParametersSchema(method.Method),
        };
    }

    /// <summary>The JSON schema (<c>{"type":"object","properties":{…},"required":[…]}</c>) for the parameters of <paramref name="method"/>.</summary>
    public static string ParametersSchema(MethodInfo method)
    {
        ArgumentNullException.ThrowIfNull(method);
        using MemoryStream buffer = new();
        using (Utf8JsonWriter writer = new(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString("type", "object");
            writer.WriteStartObject("properties");
            List<string> required = [];
            foreach (ParameterInfo parameter in method.GetParameters())
            {
                if (IsSkipped(parameter)) continue;
                string parameterName = parameter.Name ?? throw new ArgumentException("Every tool parameter needs a name.", nameof(method));
                writer.WriteStartObject(parameterName);
                WriteType(writer, parameter);
                string? description = parameter.GetCustomAttribute<DescriptionAttribute>()?.Description;
                if (!string.IsNullOrEmpty(description)) writer.WriteString("description", description);
                writer.WriteEndObject();
                if (!IsOptional(parameter)) required.Add(parameterName);
            }
            writer.WriteEndObject();
            writer.WriteStartArray("required");
            foreach (string item in required) writer.WriteStringValue(item);
            writer.WriteEndArray();
            writer.WriteEndObject();
        }
        return Encoding.UTF8.GetString(buffer.GetBuffer(), 0, (int)buffer.Length);
    }

    /// <summary>True for parameters that never come from the model.</summary>
    internal static bool IsSkipped(ParameterInfo parameter) => parameter.ParameterType == typeof(CancellationToken);

    /// <summary>True when the model may omit the parameter: it has a default or its type is nullable.</summary>
    internal static bool IsOptional(ParameterInfo parameter)
        => parameter.HasDefaultValue || Nullable.GetUnderlyingType(parameter.ParameterType) is not null;

    /// <summary>The non-nullable type the model's value converts to.</summary>
    internal static Type EffectiveType(Type type) => Nullable.GetUnderlyingType(type) ?? type;

    private static void WriteType(Utf8JsonWriter writer, ParameterInfo parameter)
    {
        Type type = EffectiveType(parameter.ParameterType);
        if (type.IsEnum)
        {
            writer.WriteString("type", "string");
            writer.WriteStartArray("enum");
            foreach (string value in Enum.GetNames(type)) writer.WriteStringValue(value);
            writer.WriteEndArray();
            return;
        }
        writer.WriteString("type", Type.GetTypeCode(type) switch
        {
            TypeCode.String => "string",
            TypeCode.Boolean => "boolean",
            TypeCode.Byte or TypeCode.SByte or TypeCode.Int16 or TypeCode.UInt16 or TypeCode.Int32 or TypeCode.UInt32
                or TypeCode.Int64 or TypeCode.UInt64 => "integer",
            TypeCode.Single or TypeCode.Double or TypeCode.Decimal => "number",
            _ => throw new NotSupportedException(
                $"Tool parameter '{parameter.Name}' has type {type.Name}; supported: string, integer and floating-point numbers, bool, enums, and their nullable forms."),
        });
    }
}
