using FlowerBot.API.Services.Interfaces;
using Microsoft.AspNetCore.Mvc;

namespace FlowerBot.Controllers
{
    [ApiController]
	[Route("api/file-manager")]
	public class FileManagerController : ControllerBase
	{
		private readonly ILogger<FileManagerController> _logger;
        private readonly IFileManagerService _fileManagerService;

        public FileManagerController(
			ILogger<FileManagerController> logger,
			IFileManagerService fileManagerService
			)
		{
			_logger = logger;
            _fileManagerService = fileManagerService;
        }

        [HttpPost("model/train")]
		public IActionResult RunTraining()
        {
			_ = _fileManagerService.ExecuteTrainScript();
			return Ok();
        }
    }
}