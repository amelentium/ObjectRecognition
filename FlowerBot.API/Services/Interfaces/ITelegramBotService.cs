using Telegram.Bot.Types;

namespace FlowerBot.Services.Interfaces
{
	public interface ITelegramBotService
	{
		Task MakePredictAsync(Update update);
	}
}
