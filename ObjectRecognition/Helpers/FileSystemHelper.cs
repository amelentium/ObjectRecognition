using ObjectRecognition.Enums;

namespace ObjectRecognition.Helpers
{
    public static class FileSystemHelper
    {
        public static string GetItemPath(ItemType itemType, params string[] itemRelativeHierarchy)
        {
            return itemType switch
            {
                    ItemType.Dataset => Constants.DatasetsPath,
                    ItemType.DatasetClass or
                    ItemType.DatasetImage => string.Format(Constants.DatasetClassesPathTemplate, itemRelativeHierarchy),

                    _ => Constants.WebRootPath,
            };
        }
    }
}
