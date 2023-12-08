using Microsoft.AspNetCore.Components;
using ObjectRecognition.Services.Interfaces;
using System.Diagnostics;

namespace ObjectRecognition.Services
{
    public class ScriptService : IScriptService
    {
        public async Task ExecuteImagePredictionAsync(string modelPath, string imagePath, EventCallback<string> onResultRecived)
        {
            var process = new Process();
            var processInfo = process.StartInfo;

            processInfo.FileName = "C:\\Programs\\Python\\python.exe";
            processInfo.Arguments = $"{Constants.SctiptsPath}\\image_predict.py {modelPath} {imagePath}";
            processInfo.UseShellExecute = false;
            processInfo.RedirectStandardOutput = true;

            process.Start();
            var result = await process.StandardOutput.ReadToEndAsync();
            await onResultRecived.InvokeAsync(result);
            await process.WaitForExitAsync();
        }

        public async Task ExecuteModelTrainingAsync(string modelPath, string trainImagesPath, string testImagesPath, EventCallback<string> onTrainStepCompleted)
        {
            var process = new Process();
            var processInfo = process.StartInfo;

            processInfo.FileName = "C:\\Programs\\Python\\python.exe";
            processInfo.Arguments = $"-u  {Constants.SctiptsPath}\\model_train.py {modelPath} {trainImagesPath} {testImagesPath}";
            processInfo.UseShellExecute = false;
            processInfo.RedirectStandardOutput = true;

            process.OutputDataReceived += async (sender, args) =>
            {
                await onTrainStepCompleted.InvokeAsync(args.Data);
            };

            process.Start();
            process.BeginOutputReadLine();
            await process.WaitForExitAsync();
        }
    }
}
