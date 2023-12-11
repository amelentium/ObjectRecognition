namespace ObjectRecognition
{
    public static class Constants
    {
        public static string WebRootPath = string.Empty;
        public static string ContentRootPath = string.Empty;

        public static string DatasetsPath => WebRootPath + "\\images\\data_sets";
        public static string UserImagesPath => WebRootPath + "\\images\\user_images";
        public static string ModelsPath => WebRootPath + "\\models";

        public static string[] ImageFileExtentions = new[] { ".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp" };
        public const string TrainImagesFolderName = "train";
        public const string TestImagesFolderName = "test";
        public const long DatasetImageFileMaxSize = 3 * 1024 * 1024;
        public const long UserImageFileMaxSize = 15 * 1024 * 1024;

        public static string SctiptsPath => ContentRootPath + "Scripts";
        public static string PredictionSctiptName => "image_predict.py";
        public static string TrainSctiptName => "model_train.py";

        public static void SetConstants(this IWebHostEnvironment env)
        {
            WebRootPath = env.WebRootPath;
            ContentRootPath = env.ContentRootPath;
        }
    }
}
