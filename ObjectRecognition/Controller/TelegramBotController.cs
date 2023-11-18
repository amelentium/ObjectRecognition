using ObjectRecognition.Services.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using Telegram.Bot.Types;

namespace ObjectRecognition.Controllers
{
	[AllowAnonymous]
	[ApiController]
	[Route("api/telegram-bot")]
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

		[HttpPost("message/update")]
		public IActionResult TelegramWebhook([FromBody] Update update)
        {
			if (!string.IsNullOrEmpty(update.Message?.Photo?.Last()?.FileId))
			{
				_ = _telegramBotService.MakePredictAsync(update);
			}

			return Ok();
        }
    }
}