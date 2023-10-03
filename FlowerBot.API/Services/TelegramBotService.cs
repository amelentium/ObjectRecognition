using FlowerBot.Exceptions;
using FlowerBot.Services.Interfaces;
using System.Diagnostics;
using Telegram.Bot;
using Telegram.Bot.Types;

namespace FlowerBot.Services
{
	public class TelegramBotService : ITelegramBotService
	{
		private readonly ILogger<TelegramBotService> _logger;
		private readonly ITelegramBotClient _telegramBot;
		private readonly IWebHostEnvironment _env;

		public TelegramBotService(
			ILogger<TelegramBotService> logger,
			ITelegramBotClient telegramBot,
			IWebHostEnvironment env)
		{
			_logger = logger;
			_telegramBot = telegramBot;
			_env = env;
		}

		public async Task MakePredictAsync(Update update)
		{
			var chatId = update.Message.Chat.Id;
			var imageId = update.Message.Photo.Last().FileId;

			try
			{
				await SaveUserImageAsync(imageId);
				await ExecutePredictScript(imageId);
				await SendPredictResult(chatId, imageId);
				ClearUserImages(imageId);
			}
			catch (Exception ex)
			{
				await _telegramBot.SendTextMessageAsync(chatId, ex.Message);
			}
		}

		private async Task SaveUserImageAsync(string imageId)
		{
			var rootPath = _env.ContentRootPath;

			if (!Directory.Exists($"{rootPath}Images\\user_images"))
			{
				Directory.CreateDirectory($"{rootPath}Images\\user_images");
			}

			var imagePath = $"{rootPath}Images\\user_images\\{imageId}.jpg";
			using (var imageStream = System.IO.File.OpenWrite(imagePath))
			{
				try
				{
					var image = await _telegramBot.GetInfoAndDownloadFileAsync(imageId, imageStream);
				}
				catch
				{
					throw new TelegramImageDownloadException();
				}
			}
		}

		private async Task ExecutePredictScript(string imageId)
		{
			var cmdProcess = Process.Start(new ProcessStartInfo
			{
				WorkingDirectory = $"{_env.ContentRootPath}Scripts",
				FileName = "image_predict.py",
				Arguments = imageId,
				CreateNoWindow = true,
				UseShellExecute = true
			});

			await cmdProcess.WaitForExitAsync();
		}

		private async Task SendPredictResult(long chatId, string imageId)
		{
			await using Stream stream = System.IO.File.OpenRead($"{_env.ContentRootPath}Images\\user_images\\{imageId}_result.jpg");

			await _telegramBot.SendPhotoAsync(chatId, new InputFileStream(content: stream, fileName: "predict_result.jpg"));
		}

		private void ClearUserImages(string imageId)
		{
			System.IO.File.Delete($"{_env.ContentRootPath}Images\\user_images\\{imageId}.jpg");
			System.IO.File.Delete($"{_env.ContentRootPath}Images\\user_images\\{imageId}_result.jpg");
		}
	}
}
