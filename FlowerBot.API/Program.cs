using FlowerBot;
using FlowerBot.API.Services.Interfaces;
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

builder.Services.Configure<TelegramBotOptions>(configuration.GetSection(TelegramBotOptions.Configuration));

builder.Services.AddHttpClient("telegram_bot_client")
				.AddTypedClient<ITelegramBotClient>((httpClient, sp) =>
				{
					TelegramBotOptions botOptions = sp.GetService<IOptions<TelegramBotOptions>>().Value;
					TelegramBotClientOptions options = new(botOptions.Token);
					return new TelegramBotClient(options, httpClient);
				});

builder.Services.AddScoped<ITelegramBotService, TelegramBotService>();
builder.Services.AddScoped<IFileManagerService, FileManagerService>();

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
