using FlowerBot.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Telegram.Bot.Types;

namespace FlowerBot.Controllers
{
	[ApiController]
	[Route("api/telegram-bot/message/update")]
	public class TelegramBotController : ControllerBase
	{
		private readonly ILogger<TelegramBotController> _logger;
		private readonly ITelegramBotService _telegramBotService;

		public TelegramBotController(
			ILogger<TelegramBotController> logger,
			ITelegramBotService telegramBotService)
		{
			_logger = logger;
			_telegramBotService = telegramBotService;
		}

		[HttpPost]
		public IActionResult TelegramWebhook([FromBody] Update update)
        {
			if (update != null)
			{
				_ = _telegramBotService.MakePredictAsync(update);
			}

			return Ok();
        }
    }
}