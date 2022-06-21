namespace FlowerBot.Services.Interfaces
{
	public interface ISpeciesContext
	{
		Dictionary<string, string> Species { get; }
		Task<string> AddSpecie(string name);
		Task RemoveSpecie(string name);
		bool IsExistSpecie(string name);
	}
}
