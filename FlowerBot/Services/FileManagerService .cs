using FlowerBot.Services.Interfaces;
using System.Diagnostics;

namespace FlowerBot.Services
{
	public class FileManagerService : IFileManagerService
	{
		private readonly ILogger<FileManagerService> _logger;
		private readonly IWebHostEnvironment _env;

		public FileManagerService(
			ILogger<FileManagerService> logger,
			IWebHostEnvironment env)
		{
			_logger = logger;
			_env = env;
		}

		public async Task ExecuteTrainScript()
		{
			var cmdProcess = Process.Start(new ProcessStartInfo
			{
				WorkingDirectory = $"{_env.ContentRootPath}Scripts",
				FileName = "model_train_test.py",
				CreateNoWindow = false,
				UseShellExecute = true
			});

			await cmdProcess.WaitForExitAsync();
		}
	}
}
