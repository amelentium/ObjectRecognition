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
		private readonly ITelegramBotService _telegramBotService;

		public TelegramBotController(ITelegramBotService telegramBotService)
		{
			_telegramBotService = telegramBotService;
		}

		[HttpPost("message/update")]
		public async Task<IActionResult> TelegramWebhook([FromBody] Update update)
        {
			if (!string.IsNullOrEmpty(update.Message?.Photo?.Last()?.FileId))
			{
				await _telegramBotService.MakePredictAsync(update);
			}

			return Ok();
        }
    }
}