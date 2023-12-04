using ObjectRecognition.Enums;

namespace ObjectRecognition.Helpers
{
    public static class FileSystemHelper
    {
        public static string GetItemPath(ItemType itemType, params string[] itemRelativeHierarchy)
        {
            var pathTemplate = itemType switch
            {
                    ItemType.Dataset => Constants.DatasetsPath + "\\{0}",
                    ItemType.DatasetClass => Constants.DatasetsPath + "\\{0}\\{1}\\{2}",
                    ItemType.DatasetImage => Constants.DatasetsPath + "\\{0}\\{1}\\{2}\\{3}",

                    ItemType.UserImage => Constants.UserImagesPath + "\\{0}",

                    ItemType.Model => Constants.ModelsPath + "\\{0}",

                    _ => string.Empty,
            };

            return string.Format(pathTemplate, itemRelativeHierarchy);
        }
    }
}
