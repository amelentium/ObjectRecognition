using Microsoft.AspNetCore.Components;

namespace ObjectRecognition.Services.Interfaces
{
    public interface IScriptService
    {
        void ExecuteImagePrediction(string modelPath, string imagePath, EventCallback<string> onResultRecived);

        void ExecuteModelTraining(string modelPath, string trainImagesPath, string testImagesPath, EventCallback<string> onTrainStepCompleted);
    }
}