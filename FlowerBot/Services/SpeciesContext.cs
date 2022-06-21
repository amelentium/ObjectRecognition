using FlowerBot.Exceptions;
using FlowerBot.Services.Interfaces;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Unicode;

namespace FlowerBot.Services
{
	public class SpeciesContext : ISpeciesContext
	{
		private static readonly JsonSerializerOptions _serializerOptions;
		private static readonly Dictionary<string, string> _species = new();
		private const string SpeciesFilePath = "./Scripts/cat_to_name.json";
		public Dictionary<string, string> Species => _species;

		static SpeciesContext()
		{
			_serializerOptions = new JsonSerializerOptions()
			{
				Encoder = JavaScriptEncoder.Create(UnicodeRanges.BasicLatin, UnicodeRanges.Cyrillic),
				PropertyNameCaseInsensitive = true,
				WriteIndented = true
			};

			try
			{
				using var stream = new StreamReader(SpeciesFilePath);
				var json = stream.ReadToEnd();

				_species = JsonSerializer.Deserialize<Dictionary<string, string>>(json, _serializerOptions);
			}
			catch
			{
				_species = new();
				Save();
			}
		}

		public async Task<string> AddSpecie(string name)
		{
			if (_species.ContainsValue(name.ToLower()))
				throw new SpeciesException("Specie with same name already exist");

			var id = (int.Parse(_species.Last().Key) + 1).ToString();
			_species.Add(id.ToString(), name);
			await SaveAsync();

			return id;
		}

		public async Task RemoveSpecie(string name)
		{
			var specie = _species.FirstOrDefault(x => x.Value == name.ToLower());
			if (int.Parse(specie.Key) == 0)
				throw new SpeciesException("Specie with same name does not exist");

			_species.Remove(specie.Key);
			await SaveAsync();
		}

		public bool IsExistSpecie(string name)
		{
			return _species.ContainsValue(name.ToLower());
		}

		private static void Save()
		{
			var json = JsonSerializer.Serialize(_species, _serializerOptions);
			File.WriteAllText(SpeciesFilePath, json);
		}

		private static async Task SaveAsync()
		{
			var json = JsonSerializer.Serialize(_species, _serializerOptions);
			await File.WriteAllTextAsync(SpeciesFilePath, json);
		}
	}
}
