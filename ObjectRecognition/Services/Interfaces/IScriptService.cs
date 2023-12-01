namespace ObjectRecognition.Services.Interfaces
{
    public interface IScriptService
    {
        Task<Dictionary<string, string>> ExecuteImagePrediction(string modelPath, string imagePath, string resultPath);
    }
}