using ObjectRecognition.Services.Interfaces;
using System.Diagnostics;

namespace ObjectRecognition.Services
{
    public class FileManagerService : IFileManagerService
	{

		public async Task ExecuteTrainScript()
		{
			var cmdProcess = Process.Start(new ProcessStartInfo
			{
				WorkingDirectory = $"{Constants.ContentRootPath}Scripts",
				FileName = "model_train_test.py",
				CreateNoWindow = false,
				UseShellExecute = true
			});

			await cmdProcess.WaitForExitAsync();
		}
	}
}
