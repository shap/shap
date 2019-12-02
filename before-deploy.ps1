# Appveyor script to install hadoop extra package to run spark on windows
# ========================== Hadoop bin package
$hadoopVer = "2.6.4"
$hadoopPath = "$tools\hadoop"
if (!(Test-Path $hadoopPath)) {
    New-Item -ItemType Directory -Force -Path $hadoopPath | Out-Null
}
Push-Location $hadoopPath

Start-FileDownload "https://github.com/steveloughran/winutils/archive/master.zip" "winutils-master.zip"

# extract
Invoke-Expression "7z.exe x winutils-master.zip"

# add hadoop bin to environment variables
$env:HADOOP_HOME = "$hadoopPath/winutils-master/hadoop-$hadoopVer"
$env:Path += ";$env:HADOOP_HOME\bin"

Pop-Location