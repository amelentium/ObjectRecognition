using ObjectRecognition;
using ObjectRecognition.Options;
using ObjectRecognition.Services;
using ObjectRecognition.Services.Interfaces;
using Microsoft.Extensions.Options;
using Telegram.Bot;

var builder = WebApplication.CreateBuilder(args);
var configuration = builder.Configuration;

builder.Environment.SetConstants();
builder.Services.AddControllers()
                .AddNewtonsoftJson();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();

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

if (!app.Environment.IsDevelopment())
{
	app.UseExceptionHandler("/Error");
	app.UseHsts();
}

app.UseStaticFiles();

app.UseRouting();

app.UseAuthorization();

app.MapBlazorHub();

app.MapFallbackToPage("/_Host");

app.MapControllers();

app.Run();
