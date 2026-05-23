@echo off
set CSC=%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if not exist "%CSC%" set CSC=%WINDIR%\Microsoft.NET\Framework\v4.0.30319\csc.exe
if not exist "%CSC%" (
  echo csc.exe was not found.
  exit /b 1
)
"%CSC%" /nologo /target:winexe /reference:System.Windows.Forms.dll /win32icon:sts2_drawer\assets\app_icon.ico /out:STS2MapDrawer.exe launcher\STS2MapDrawerLauncher.cs
