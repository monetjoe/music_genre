using System;
using System.Text;
using System.IO;

namespace edge_no_recoverwin
{
    class Program
    {
        static void Main(string[] args)
        {
            String tag = "C:\\Users\\" + System.Environment.UserName + "\\AppData\\Local\\Microsoft\\Edge\\User Data\\Default\\Preferences";

            //防止文本字符中有ANSI。必须用Encoding.UTF8
            StreamReader reader = new StreamReader(tag, Encoding.UTF8);
            String a = reader.ReadToEnd();

            if (a.Contains("Normal"))
            {
                return;
            }

            if (a.Contains("Crashed"))
            {
                //将 Preferences 文件中 Crashed 替换为 Normal。
                a = a.Replace("Crashed", "Normal");
            }
            else if (a.Contains("SessionEnded"))
            {
                //将 Preferences 文件中 SessionEnded 替换为 Normal。
                a = a.Replace("SessionEnded", "Normal");
            }

            StreamWriter readTxt = new StreamWriter(tag + ".bad", false, Encoding.UTF8);
            readTxt.Write(a);
            readTxt.Flush();
            readTxt.Close();
            reader.Close();

            //将Preferences.bad复制覆盖到Preferences
            File.Copy(tag + ".bad", tag, true);
        }
    }
}