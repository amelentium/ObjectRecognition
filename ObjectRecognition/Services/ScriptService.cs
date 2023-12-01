using System.Diagnostics;
using ObjectRecognition.Services.Interfaces;

namespace ObjectRecognition.Services
{
    public class ScriptService : IScriptService
    {
        public Task<Dictionary<string, string>> ExecuteImagePrediction(string modelPath, string imagePath, string resultPath)
        {
            return Task.Run(async () =>
            {
                var cmdProcess = Process.Start(new ProcessStartInfo
                {
                    WorkingDirectory = Constants.SctiptsPath,
                    FileName = "image_predict.py",
                    Arguments = $"{modelPath} {imagePath} {resultPath}",
                    UseShellExecute = true,
                });

                await cmdProcess.WaitForExitAsync();

                var result = new Dictionary<string, string>();
                var lines = await File.ReadAllLinesAsync(resultPath);

                foreach (var line in lines)
                {
                    var parts = line.Split('\t');
                    var @class = parts[0];
                    var prob = parts[1];

                    result.TryAdd(@class, prob);
                }

                return result;
            });
        }
    }
}
