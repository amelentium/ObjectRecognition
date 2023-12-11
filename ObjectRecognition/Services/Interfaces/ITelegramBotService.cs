using Telegram.Bot.Types;

namespace ObjectRecognition.Services.Interfaces
{
	public interface ITelegramBotService
	{
		Task MakePredictAsync(Update update);
	}
}
