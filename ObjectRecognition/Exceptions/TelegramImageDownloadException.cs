using System.Runtime.Serialization;

namespace ObjectRecognition.Exceptions
{
	[Serializable]
	internal class TelegramImageDownloadException : Exception
	{
		public TelegramImageDownloadException() : base("Something get wrong during image download.\nPlease try again later")
		{
		}

		public TelegramImageDownloadException(string message) : base(message)
		{
		}

		public TelegramImageDownloadException(string message, Exception innerException) : base(message, innerException)
		{
		}

		protected TelegramImageDownloadException(SerializationInfo info, StreamingContext context) : base(info, context)
		{
		}
	}
}