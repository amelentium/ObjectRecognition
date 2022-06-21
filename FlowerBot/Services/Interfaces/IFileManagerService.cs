using FlowerBot.Data;

namespace FlowerBot.Services.Interfaces
{
	public interface IFileManagerService
	{
		Task ExecuteTrainScript();
		Task AddSpecie(SpecieUpload specie);
		void RemoveSpecie(string specieName);
	}
}