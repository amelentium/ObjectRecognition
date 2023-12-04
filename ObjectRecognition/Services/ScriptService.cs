using ObjectRecognition.Services.Interfaces;
using System.Diagnostics;

namespace ObjectRecognition.Services
{
    public class ScriptService : IScriptService
    {
        public async void ExecuteImagePrediction(string modelPath, string imagePath, Action<string> onResultRecived)
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "C:\\Programs\\Python\\python.exe",
                Arguments = $"{Constants.SctiptsPath}\\image_predict.py {modelPath} {imagePath}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
            };

            using Process process = Process.Start(processInfo);
            using StreamReader reader = process.StandardOutput;

            string result = await reader.ReadToEndAsync();
            onResultRecived.Invoke(result);
        }

        public async void ExecuteModelTraining(string modelPath, string trainImagesPath, string testImagesPath, Action<string> onTrainStepCompleted)
        {
            var processInfo = new ProcessStartInfo
            {
                FileName = "C:\\Programs\\Python\\python.exe",
                Arguments = $"{Constants.SctiptsPath}\\model_train.py {modelPath} {trainImagesPath} {testImagesPath}",
                UseShellExecute = false,
                RedirectStandardOutput = true,
            };

            using Process process = Process.Start(processInfo);
            using StreamReader reader = process.StandardOutput;

            while (!process.HasExited)
            {
                string result = await reader.ReadLineAsync();

                if (!string.IsNullOrEmpty(result))
                    onTrainStepCompleted.Invoke(result);
                else
                    await Task.Delay(TimeSpan.FromSeconds(3));
            }
        }
    }
}
