using System.Text.RegularExpressions;

namespace ObjectRecognition.Extentions
{
    public static class StringExtention
    {
        private static readonly Regex _camelCaseRegexp;

        static StringExtention()
        {
            _camelCaseRegexp = new Regex(@"
                (?<=[A-Z])(?=[A-Z][a-z]) |
                (?<=[^A-Z])(?=[A-Z]) |
                (?<=[A-Za-z])(?=[^A-Za-z])",
                RegexOptions.IgnorePatternWhitespace);
        }

        public static string SplitCamelCase(this string str)
        {
            return _camelCaseRegexp.Replace(str, " ");
        }
    }
}
