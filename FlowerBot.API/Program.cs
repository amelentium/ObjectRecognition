using FlowerBot;
using FlowerBot.Services;
using FlowerBot.Services.Interfaces;
using Microsoft.Extensions.Options;
using Telegram.Bot;

var builder = WebApplication.CreateBuilder(args);
var configuration = builder.Configuration;

builder.Services.AddControllers()
    .AddNewtonsoftJson();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.Configure<TelegramBotOptions>(configuration.GetSection("TelegramBotOptions"));
builder.Services.AddTransient(ser => ser.GetService<IOptions<TelegramBotOptions>>().Value);
builder.Services.AddSingleton<ITelegramBotClient>(
    x =>
    {
        var settings = x.GetRequiredService<TelegramBotOptions>();
        return new TelegramBotClient(settings.Token);
    });
builder.Services.AddScoped<ITelegramBotService, TelegramBotService>();

var app = builder.Build();

app.UseTelegramBotWebhook();

if (app.Environment.IsDevelopment())
{
	app.UseSwagger();
	app.UseSwaggerUI();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
