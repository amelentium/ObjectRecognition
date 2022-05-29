using FlowerBot.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;
using Telegram.Bot.Types;

namespace FlowerBot.Controllers
{
	[ApiController]
	[Route("api/file-manager")]
	public class FileManagerController : ControllerBase
	{
		private readonly ILogger<FileManagerController> _logger;

		public FileManagerController(
			ILogger<FileManagerController> logger
			)
		{
			_logger = logger;
		}
    }
}