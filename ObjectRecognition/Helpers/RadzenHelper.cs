namespace ObjectRecognition.Helpers
{
    public static class RadzenHelper
    {
        public static string NormalizeImagePath(string imagePath) => imagePath.Replace(Constants.WebRootPath, string.Empty);
    }
}
