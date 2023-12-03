namespace ObjectRecognition.Services.Interfaces
{
    public interface IScriptService
    {
        void ExecuteImagePrediction(string modelPath, string imagePath, Action<string> onResultRecived);
    }
}