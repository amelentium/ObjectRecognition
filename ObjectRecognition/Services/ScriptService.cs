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
    }
}
