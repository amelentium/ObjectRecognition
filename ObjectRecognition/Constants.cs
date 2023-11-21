namespace ObjectRecognition
{
    public static class Constants
    {
        public static string WebRootPath = string.Empty;

        public static string DatasetsPath => WebRootPath + "\\images\\data_sets";

        public static string DatasetClassesPathTemplate => DatasetsPath + "\\{0}\\classes.json";

        public static void SetConstants(this IWebHostEnvironment env)
        {
            WebRootPath = env.WebRootPath;
        }
    }
}
