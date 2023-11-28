﻿namespace ObjectRecognition
{
    public static class Constants
    {
        public static string WebRootPath = string.Empty;
        public static string ContentRootPath = string.Empty;

        public static string DatasetsPath => WebRootPath + "\\images\\data_sets";
        public const string TrainImagesFolderName = "train";
        public const string TestImagesFolderName = "test";

        public static string[] ImageFileExtentions = new[] { ".jpeg", ".jpg", ".png" };
        public const long ImageFileMaxSize = 3 * 1024 * 1024;


        public static void SetConstants(this IWebHostEnvironment env)
        {
            WebRootPath = env.WebRootPath;
            ContentRootPath = env.ContentRootPath;
        }
    }
}
