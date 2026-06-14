namespace HartsyInference.Server;

public sealed class Program
{
    public static void Main(string[] args)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

        // TODO: builder.Services.AddHartsyInference(options => { ... });

        WebApplication app = builder.Build();

        app.MapGet("/health", () => Results.Ok(new { status = "healthy" }));
        app.MapGet("/ready", () => Results.Ok(new { status = "ready" }));

        // TODO: app.MapHartsyInferenceEndpoints();

        app.Run();
    }
}
