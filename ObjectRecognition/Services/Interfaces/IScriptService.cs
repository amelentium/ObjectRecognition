using Microsoft.AspNetCore.Components;

namespace ObjectRecognition.Services.Interfaces
{
    public interface IScriptService
    {
        Task ExecuteImagePredictionAsync(string modelPath, string imagePath, EventCallback<string> onResultRecived);

        Task ExecuteModelTrainingAsync(string modelPath, string trainImagesPath, string testImagesPath, EventCallback<string> onTrainStepCompleted);
    }
}