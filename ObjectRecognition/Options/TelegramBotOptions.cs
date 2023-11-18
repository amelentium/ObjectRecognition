namespace ObjectRecognition.Options
{
    public class TelegramBotOptions
    {
        public const string Configuration = "TelegramBotOptions";
        public string Token { get; set; }
        public string WebHookAddress { get; set; }
    }
}
