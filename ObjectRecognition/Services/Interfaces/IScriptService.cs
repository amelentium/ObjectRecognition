namespace ObjectRecognition.Services.Interfaces
{
    public interface IScriptService
    {
        void ExecuteImagePrediction(string modelPath, string imagePath, Action<string> onResultRecived);

        void ExecuteModelTraining(string modelPath, string trainImagesPath, string testImagesPath, Action<string> onTrainStepCompleted);
    }
}