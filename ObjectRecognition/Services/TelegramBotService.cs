using Microsoft.AspNetCore.Components;
using ObjectRecognition.Exceptions;
using ObjectRecognition.Helpers;
using ObjectRecognition.Services.Interfaces;
using Telegram.Bot;
using Telegram.Bot.Types;

namespace ObjectRecognition.Services
{
    public class TelegramBotService : ITelegramBotService
    {
        private readonly ITelegramBotClient _telegramBot;
        private readonly IScriptService _scriptService;

        public TelegramBotService(
            ITelegramBotClient telegramBot,
            IScriptService scriptService)
        {
            _telegramBot = telegramBot;
            _scriptService = scriptService;
        }

        public async Task MakePredictAsync(Update update)
        {
            var chatId = update.Message.Chat.Id;
            var imageId = update.Message.Photo.Last().FileId;
            var imagePath = FileSystemHelper.GetItemPath(Enums.ItemType.UserImage, $"{imageId}.png");

            try
            {
                await SaveUserImageAsync(imageId, imagePath);
                ExecutePredictScript(chatId, imagePath);
                ClearUserImages(imageId);
            }
            catch (Exception ex)
            {
                await _telegramBot.SendTextMessageAsync(chatId, ex.Message);
            }
        }

        private async Task SaveUserImageAsync(string imageId, string imagePath)
        {
            try
            {
                using var imageStream = new FileStream(imagePath, FileMode.Create);

                var image = await _telegramBot.GetInfoAndDownloadFileAsync(imageId, imageStream);
            }
            catch
            {
                throw new TelegramImageDownloadException();
            }
        }

        private async void ExecutePredictScript(long chatId, string imagePath)
        {
            var ResultRecivedCallback = new EventCallback<string>(null, (MulticastDelegate)Delegate.Combine(new Delegate[]
            {
                new Action<string> (async (result) => await SendResult(chatId, result)),
            }));

            await _scriptService.ExecuteImagePredictionAsync("C:\\Users\\amele\\source\\repos\\ObjectRecognition\\ObjectRecognition\\wwwroot\\models\\tiny_image_net.tar", imagePath, ResultRecivedCallback);
        }

        private async Task SendResult(long chatId, string result)
        {
            await _telegramBot.SendTextMessageAsync(chatId, result);
        }

        private static void ClearUserImages(string imagePath)
        {
            System.IO.File.Delete(imagePath);
        }
    }
}
