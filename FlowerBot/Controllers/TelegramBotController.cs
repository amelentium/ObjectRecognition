using Microsoft.AspNetCore.Mvc;
using Telegram.Bot;
using Telegram.Bot.Types;

namespace FlowerBot.Controllers
{
	[ApiController]
	[Route("api/telegram-bot/message/update")]
	public class TelegramBotController : ControllerBase
	{
		private readonly ILogger<TelegramBotController> _logger;
		private readonly ITelegramBotClient _telegramBot;

		public TelegramBotController(
			ILogger<TelegramBotController> logger,
			ITelegramBotClient telegramBot)
		{
			_logger = logger;
			_telegramBot = telegramBot;
		}

		[HttpPost]
		public async Task<IActionResult> TelegramWebhook([FromBody] Update update)
        {
            if (update != null)
				await _telegramBot.SendTextMessageAsync(update.Message.Chat.Id, update.Message.Text);

            return Ok();
        }
    }
}