using System.Buffers.Binary;
using System.ComponentModel;
using System.Net.Sockets;
using System.Text;
using HartsyInference.Audio.Io;
using Spectre.Console;
using Spectre.Console.Cli;

namespace HartsyInference.Cli.Commands;

/// <summary>Acts as a voice satellite: streams a WAV to the wake listener in real time and prints what comes
/// back. This is the reference client for <c>docs/Research/WAKE_SATELLITE_PROTOCOL.md</c> — useful for trying
/// the system before any firmware exists, for measuring false accepts over long recordings, and as a worked
/// example of the framing to port to a device.</summary>
public sealed class WakeClientCommand : AsyncCommand<WakeClientCommand.Settings>
{
    /// <summary>Options for <c>hartsy wake-client</c>.</summary>
    public sealed class Settings : CommandSettings
    {
        /// <summary>WAV file to stream as if it were live microphone audio.</summary>
        [CommandArgument(0, "<audio>")]
        [Description("WAV file to stream (any sample rate; downmixed and resampled to 16 kHz mono).")]
        public string Audio { get; init; } = "";

        [CommandOption("--host")]
        [Description("Wake listener host.")]
        public string Host { get; init; } = "127.0.0.1";

        [CommandOption("-p|--port")]
        [Description("Wake listener port.")]
        public int Port { get; init; } = 10_800;

        [CommandOption("-d|--device-id")]
        [Description("Device id; must be stable across reconnects for the server to resume the session.")]
        public string DeviceId { get; init; } = "cli-test";

        [CommandOption("--frame-ms")]
        [Description("Audio frame size in milliseconds.")]
        public int FrameMs { get; init; } = 20;

        [CommandOption("--fast")]
        [Description("Stream as fast as the socket accepts instead of in real time. Detection results are the same; only pacing changes.")]
        public bool Fast { get; init; }

        [CommandOption("--loop")]
        [Description("Repeat the file forever — for false-accept soak tests.")]
        public bool Loop { get; init; }

        [CommandOption("--token")]
        [Description("Shared auth token, when the listener has WakeServiceOptions.AuthToken / HartsyInference__WakeAuthToken set — omit only against a listener with no token configured.")]
        public string? Token { get; init; }
    }

    public override async Task<int> ExecuteAsync(CommandContext context, Settings settings)
    {
        if (!File.Exists(settings.Audio))
        {
            AnsiConsole.MarkupLine($"[red]No such file:[/] {settings.Audio}");
            return 1;
        }

        float[] pcm = LoadInt16Scaled(settings.Audio);
        int frameSamples = Math.Max(160, 16_000 * settings.FrameMs / 1000);
        AnsiConsole.MarkupLine($"[grey]{pcm.Length / 16_000.0:F1}s of audio, {frameSamples} samples/frame, device '{settings.DeviceId}'[/]");

        using TcpClient client = new() { NoDelay = true };
        await client.ConnectAsync(settings.Host, settings.Port);
        using NetworkStream stream = client.GetStream();
        AnsiConsole.MarkupLine($"[green]Connected[/] to {settings.Host}:{settings.Port}");

        using CancellationTokenSource cancel = new();
        Task reader = Task.Run(() => ReadServerEventsAsync(stream, cancel.Token));

        // Token field omitted entirely (not sent as empty/null) when unset, matching what a
        // real satellite against a no-auth listener sends — WakeListener.IsTokenValid() only
        // requires it when the listener actually has one configured.
        string tokenField = string.IsNullOrEmpty(settings.Token) ? "" : $",\"token\":\"{settings.Token}\"";
        await SendAsync(stream, $"{{\"type\":\"hello\",\"data\":{{\"device_id\":\"{settings.DeviceId}\",\"rate\":16000,\"width\":2,\"channels\":1,\"firmware\":\"cli\"{tokenField}}}}}");

        long sequence = 0;
        byte[] payload = new byte[frameSamples * 2];
        do
        {
            for (int offset = 0; offset + frameSamples <= pcm.Length; offset += frameSamples)
            {
                for (int i = 0; i < frameSamples; i++)
                    BinaryPrimitives.WriteInt16LittleEndian(payload.AsSpan(i * 2), (short)Math.Clamp(pcm[offset + i], -32768f, 32767f));

                await SendAsync(stream, $"{{\"type\":\"audio-chunk\",\"data\":{{\"seq\":{sequence++}}},\"payload_length\":{payload.Length}}}", payload);
                // Real-time pacing by default: a soak test that streams at 100x tells you nothing about how
                // the listener behaves against actual devices.
                if (!settings.Fast) await Task.Delay(settings.FrameMs);
            }
        }
        while (settings.Loop && !cancel.IsCancellationRequested);

        // Give the server time to finish transcribing the final utterance before hanging up.
        await Task.Delay(TimeSpan.FromSeconds(6));
        await SendAsync(stream, "{\"type\":\"bye\"}");
        await cancel.CancelAsync();
        try { await reader; } catch (OperationCanceledException) { }
        return 0;
    }

    private static async Task SendAsync(NetworkStream stream, string header, byte[]? payload = null)
    {
        await stream.WriteAsync(Encoding.UTF8.GetBytes(header + "\n"));
        if (payload is not null) await stream.WriteAsync(payload);
        await stream.FlushAsync();
    }

    private static async Task ReadServerEventsAsync(NetworkStream stream, CancellationToken cancel)
    {
        byte[] buffer = new byte[8192];
        StringBuilder line = new();
        try
        {
            while (!cancel.IsCancellationRequested)
            {
                int read = await stream.ReadAsync(buffer, cancel);
                if (read == 0) return;
                for (int i = 0; i < read; i++)
                {
                    if (buffer[i] != (byte)'\n') { line.Append((char)buffer[i]); continue; }
                    string text = line.ToString();
                    line.Clear();
                    if (text.Contains("\"detection\"")) AnsiConsole.MarkupLine($"[bold green]DETECTION[/] {text.EscapeMarkup()}");
                    else if (text.Contains("\"ping\"")) await SendAsync(stream, "{\"type\":\"pong\"}");
                    else AnsiConsole.MarkupLine($"[grey]{text.EscapeMarkup()}[/]");
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex) { AnsiConsole.MarkupLine($"[yellow]Connection closed:[/] {ex.Message.EscapeMarkup()}"); }
    }

    /// <summary>Reads a WAV as 16 kHz mono at int16 scale — what the wire format and the wake models expect.</summary>
    private static float[] LoadInt16Scaled(string path)
    {
        WavFile.DecodedAudio decoded = WavFile.Read(path);
        float[] mono = decoded.ToMono();
        if (decoded.SampleRate != 16_000)
        {
            Resampler resampler = Resampler.Create(decoded.SampleRate, 16_000);
            mono = resampler.Resample(mono);
        }
        float[] scaled = new float[mono.Length];
        for (int i = 0; i < mono.Length; i++) scaled[i] = mono[i] * 32768f;
        return scaled;
    }
}
