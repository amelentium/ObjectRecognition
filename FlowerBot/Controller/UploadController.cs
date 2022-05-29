using Microsoft.AspNetCore.Mvc;
using System.Web;

namespace BlazorUsersRoles.Controllers
{
	[Route("api/[controller]")]
    [ApiController]
    public class UploadController : Controller
    {
        private readonly IWebHostEnvironment _environment;

        public UploadController(IWebHostEnvironment environment)
        {
            _environment = environment;
        }

        [HttpPost("[action]")]
        public async Task<IActionResult> MultipleAsync(IFormFile[] files, [FromQuery] string CurrentDirectory)
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

                        using (var stream = new FileStream(path, FileMode.Create))
                        {
                            await file.CopyToAsync(stream);
                        }
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