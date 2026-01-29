using Serilog;
using API.Services;
using System;
using RabbitMQ.Client;
using MassTransit;
using MassTransit.Configuration;



Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug() // Set the default minimum level
    .WriteTo.Console()
    .CreateBootstrapLogger(); // Write logs to the console


try {
    var builder = WebApplication.CreateBuilder(args);

    // Add Serilog as the logging provider
    builder.Host.UseSerilog((context, services, configuration) =>
    {
        configuration
            .ReadFrom.Configuration(context.Configuration)
            .Enrich.FromLogContext()
            .WriteTo.Console()
            .WriteTo.File("logs/log-.txt", rollingInterval: RollingInterval.Day);
    }); // Use Serilog for logging

    builder.Services.Configure<MailSettings>(builder.Configuration.GetSection("MailSettings"));
    builder.Services.AddTransient<IEmailService, EmailService>(); // Register the email service
    builder.Services.AddTransient<FinAidQueuePublisher>(); // Register the FinAidQueuePublisher
    builder.Services.AddTransient<MLQueuePublisher>(); // Register the MLQueuePublisher
    builder.Services.AddTransient<FinAidQueueConsumer>(); // Register the FinAidQueueConsumer
    builder.Services.AddHttpClient(); // Register HttpClient for making external API calls
    //Add Controllers to the container.
    builder.Services.AddControllers();
    builder.Services.AddMassTransit(busConfigurator =>
    {
        busConfigurator.SetKebabCaseEndpointNameFormatter();
        busConfigurator.AddConsumer<FinAidQueueConsumer>(); // Register the consumer
        busConfigurator.UsingRabbitMq((context, configurator) =>
        {
            configurator.Host(new Uri(builder.Configuration["MessageBroker:Host"]!), h =>
            {
                h.Username(builder.Configuration["MessageBroker:Username"]!);
                h.Password(builder.Configuration["MessageBroker:Password"]!);
            });


            configurator.ConfigureEndpoints(context);

        });
    }); 

    // Add services to the container.
    // Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
    builder.Services.AddOpenApi();

    var app = builder.Build();

    // Configure the HTTP request pipeline.
    if (app.Environment.IsDevelopment())
    {
        app.MapOpenApi();
    }

    app.UseHttpsRedirection();

    // Enable Serilog's request logging middleware
    app.UseSerilogRequestLogging(options =>
    {
        options.MessageTemplate = "Handled {RequestPath} in {Elapsed:0.0000} ms";
    });


    app.MapControllers();

    app.Run();
}
catch (Exception ex)
{
    Log.Fatal(ex, "Application start-up failed");
}
finally
{
    Log.CloseAndFlush();
}
