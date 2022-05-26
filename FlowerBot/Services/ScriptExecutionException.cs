using System.Runtime.Serialization;

namespace FlowerBot.Services
{
	[Serializable]
	internal class ScriptExecutionException : Exception
	{
		public ScriptExecutionException() : base("Something get wrong during script execution.\nPlease try again later")
		{
		}

		public ScriptExecutionException(string message) : base(message)
		{
		}

		public ScriptExecutionException(string message, Exception innerException) : base(message, innerException)
		{
		}

		protected ScriptExecutionException(SerializationInfo info, StreamingContext context) : base(info, context)
		{
		}
	}
}