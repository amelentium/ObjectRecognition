using FlowerBot.Data;
using FlowerBot.Exceptions;
using FlowerBot.Services.Interfaces;
using System.Diagnostics;

namespace FlowerBot.Services
{
	public class FileManagerService : IFileManagerService
	{
		private readonly ISpeciesContext _speciesContext;
		private readonly ILogger<FileManagerService> _logger;
		private readonly IWebHostEnvironment _env;
		private const string ImagePath = "./wwwroot/images/flower_images/{0}/{1}";

		public FileManagerService(
			ISpeciesContext speciesContext,
			ILogger<FileManagerService> logger,
			IWebHostEnvironment env)
		{
			_logger = logger;
			_env = env;
		}

		public async Task AddSpecie(SpecieUpload specie)
		{
			if (_speciesContext.IsExistSpecie(specie.Name))
				throw new SpeciesException("Specie with same name already exist");

			var specieId = _speciesContext.AddSpecie(specie.Name);
			var imgCount = 0;

			foreach (var image in specie.Images)
			{
				var imgType = (++imgCount % 4 == 0) ? "valid" : "train";
				var imagePath = string.Format(ImagePath, imgType, specieId);

				using (var stream = File.Create(imagePath))
				{
					await image.CopyToAsync(stream);
				}
			}
		}

		public void RemoveSpecie(string specieName)
		{
			if (!_speciesContext.IsExistSpecie(specieName))
				throw new SpeciesException("Specie with same name does not exist");

			var specieId = _speciesContext.Species.FirstOrDefault(x => x.Value == specieName).Key;

			Directory.Delete(string.Format(ImagePath, "train", specieId), true);
			Directory.Delete(string.Format(ImagePath, "valid", specieId), true);
		}

		public async Task ExecuteTrainScript()
		{
			var cmdProcess = Process.Start(new ProcessStartInfo
			{
				WorkingDirectory = $"{_env.ContentRootPath}Scripts",
				FileName = "model_train_test.py",
				CreateNoWindow = false,
				UseShellExecute = true
			});

			await cmdProcess.WaitForExitAsync();
		}
	}
}
