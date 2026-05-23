using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

namespace STS2MapDrawerLauncher
{
    internal static class Program
    {
        [STAThread]
        private static void Main()
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string runCmd = Path.Combine(baseDir, "run.cmd");

            if (!File.Exists(runCmd))
            {
                MessageBox.Show(
                    "run.cmd was not found next to this launcher.",
                    "STS2 Map Drawer",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
                return;
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = "cmd.exe",
                Arguments = "/c \"\"" + runCmd + "\" gui\"",
                WorkingDirectory = baseDir,
                UseShellExecute = false,
                CreateNoWindow = true,
                WindowStyle = ProcessWindowStyle.Hidden
            };

            try
            {
                Process.Start(startInfo);
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    ex.Message,
                    "STS2 Map Drawer",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error
                );
            }
        }
    }
}
