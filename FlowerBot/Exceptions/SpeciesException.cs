using System.Runtime.Serialization;

namespace FlowerBot.Exceptions
{
	[Serializable]
	internal class SpeciesException : Exception
	{
		public SpeciesException()
		{
		}

		public SpeciesException(string message) : base(message)
		{
		}

		public SpeciesException(string message, Exception innerException) : base(message, innerException)
		{
		}

		protected SpeciesException(SerializationInfo info, StreamingContext context) : base(info, context)
		{
		}
	}
}