using FlowerBot.Options;
using Microsoft.Extensions.Options;
using Newtonsoft.Json;
using Telegram.Bot;

namespace FlowerBot;

public static class StartupExtensions
{
	public static IApplicationBuilder UseTelegramBotWebhook(this IApplicationBuilder applicationBuilder)
	{
		var services = applicationBuilder.ApplicationServices;
		var logger = services.GetRequiredService<ILogger<Program>>();
		var lifetime = services.GetRequiredService<IHostApplicationLifetime>();
		var telegramBot = services.GetRequiredService<ITelegramBotClient>();
		var botOptions = services.GetService<IOptions<TelegramBotOptions>>().Value;
		var address = botOptions.WebHookAddress;

		lifetime.ApplicationStarted.Register(async () =>
			{
				logger.LogInformation("Removing webhook");
				await telegramBot.DeleteWebhookAsync();
				logger.LogInformation("Webhook removed");

				logger.LogInformation($"Setting webhook to {address}");
				await telegramBot.SetWebhookAsync(address);
				logger.LogInformation($"Webhook is set to {address}");

				var webhookInfo = await telegramBot.GetWebhookInfoAsync();
				logger.LogInformation($"Webhook info: {JsonConvert.SerializeObject(webhookInfo)}");
			});

		lifetime.ApplicationStopping.Register(async () =>
			{
				logger.LogInformation("Removing webhook");
				await telegramBot.DeleteWebhookAsync();
				logger.LogInformation("Webhook removed");
			});

		return applicationBuilder;
	}
}
