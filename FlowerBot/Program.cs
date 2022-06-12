using FlowerBot;
using FlowerBot.Services;
using FlowerBot.Services.Interfaces;
using Microsoft.Extensions.Options;
using Telegram.Bot;

var builder = WebApplication.CreateBuilder(args);
var configuration = builder.Configuration;

builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddRazorPages();
builder.Services.AddServerSideBlazor();

builder.Services.Configure<TelegramBotOptions>(configuration.GetSection("TelegramBotOptions"));
builder.Services.AddTransient(ser => ser.GetService<IOptions<TelegramBotOptions>>().Value);
builder.Services.AddSingleton<ITelegramBotClient>(
    x =>
    {
        var settings = x.GetRequiredService<TelegramBotOptions>();
        return new TelegramBotClient(settings.Token);
    });
builder.Services.AddScoped<ITelegramBotService, TelegramBotService>();
builder.Services.AddScoped<IFileManagerService, FileManagerService>();

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
	app.UseExceptionHandler("/Error");
	app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();

app.UseRouting();

app.MapBlazorHub();
app.MapFallbackToPage("/_Host");
app.MapControllers();

app.Run();
