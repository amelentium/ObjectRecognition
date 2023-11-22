using Microsoft.AspNetCore.Mvc;
using ObjectRecognition.Services.Interfaces;

namespace ObjectRecognition.Controllers
{
    [ApiController]
    [Route("api/file-manager")]
    public class FileManagerController : ControllerBase
    {
        private readonly IWebHostEnvironment _environment;
        private readonly IFileManagerService _fileManagerService;

        public FileManagerController(
            IFileManagerService fileManagerService,
            IWebHostEnvironment environment)
        {
            _fileManagerService = fileManagerService;
            _environment = environment;
        }

        [HttpPost("model/train")]
        public IActionResult RunTraining()
        {
            _ = _fileManagerService.ExecuteTrainScript();
            return Ok();
        }

        [HttpPost("upload")]
        public async Task<IActionResult> UploadFilesAsync(IFormFile[] files, [FromQuery] string CurrentDirectory)
        {
            try
            {
                if (HttpContext.Request.Form.Files.Any())
                {
                    foreach (var file in files)
                    {
                        string RequestedPath = CurrentDirectory.ToLower().Replace(_environment.WebRootPath.ToLower(), "");

                        if (RequestedPath.Contains("\\images\\"))
                        {
                            RequestedPath = RequestedPath.Replace("\\images\\", "");
                        }
                        else
                        {
                            RequestedPath = "";
                        }

                        string path = Path.Combine(_environment.WebRootPath, "images", RequestedPath, file.FileName);

                        using var stream = new FileStream(path, FileMode.Create);

                        await file.CopyToAsync(stream);
                    }
                }
                return StatusCode(200);
            }
            catch (Exception ex)
            {
                return StatusCode(500, ex.Message);
            }
        }
    }
}