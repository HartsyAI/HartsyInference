namespace HartsyInference.Server;

public sealed class Program
{
    public static void Main(string[] args)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

        builder.Services.AddHartsyInference(options =>
        {
            // Bind from the "HartsyInference" configuration section (appsettings.json / env vars).
            builder.Configuration.GetSection("HartsyInference").Bind(options);
        });

        WebApplication app = builder.Build();

        // Maps /health, /ready, /v1/models*, /v1/images/* (and optional API-key auth over /v1).
        app.MapHartsyInferenceEndpoints();

        app.Run();
    }
}
