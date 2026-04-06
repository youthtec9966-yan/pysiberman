[CmdletBinding()]
param(
    [switch]$SkipClean,
    [switch]$SkipPyInstaller,
    [switch]$SkipInno
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)] [string]$Name,
        [Parameter(Mandatory = $true)] [string]$FilePath,
        [Parameter(Mandatory = $false)] [string[]]$Arguments = @()
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name 执行失败，退出码: $LASTEXITCODE"
    }
}

$pythonExe = Join-Path $projectRoot ".venv312\Scripts\python.exe"
$specFile = Join-Path $projectRoot "pySiberMan.spec"
$issFile = Join-Path $projectRoot "pySiberMan.iss"
$mainPy = Join-Path $projectRoot "main.py"
$asrPy = Join-Path $projectRoot "asr_worker.py"

$requiredFiles = @(
    $pythonExe,
    $specFile,
    $issFile,
    $mainPy,
    $asrPy
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        throw "缺少必要文件: $file"
    }
}

$pyMajorMinor = (& $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
if ($pyMajorMinor -ne "3.12") {
    throw ".venv312 Python version mismatch: $pyMajorMinor (expected 3.12)"
}

$isccCandidates = @(
    "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    "C:\Program Files\Inno Setup 6\ISCC.exe"
)

$isccExe = $null
if (-not $SkipInno) {
    $isccExe = $isccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $isccExe) {
        throw "未找到 Inno Setup 编译器 ISCC.exe"
    }
}

if (-not $SkipClean) {
    $distDir = Join-Path $projectRoot "dist"
    $buildDir = Join-Path $projectRoot "build"
    if (Test-Path $distDir) { Remove-Item $distDir -Recurse -Force }
    if (Test-Path $buildDir) { Remove-Item $buildDir -Recurse -Force }
}

Write-Host "[1/6] 依赖导入自检"
Invoke-Checked -Name "依赖导入自检" -FilePath $pythonExe -Arguments @(
    "-c",
    "import dashscope,openai,websockets,pvrecorder,python_speech_features,fastdtw,scipy; import importlib.util; print('sherpa_onnx=' + ('ok' if importlib.util.find_spec('sherpa_onnx') else 'missing'))"
)

Write-Host "[2/6] 语法检查 main.py / asr_worker.py"
& $pythonExe -m py_compile $mainPy $asrPy

if (-not $SkipPyInstaller) {
    Write-Host "[3/6] 执行 PyInstaller"
    Invoke-Checked -Name "PyInstaller" -FilePath $pythonExe -Arguments @("-m", "PyInstaller", "--clean", "--noconfirm", $specFile)
} else {
    Write-Host "[3/6] 跳过 PyInstaller"
}

$verifyPaths = @(
    (Join-Path $projectRoot "dist\pySiberMan\pySiberMan.exe"),
    (Join-Path $projectRoot "dist\pySiberMan\_internal\templates\wake"),
    (Join-Path $projectRoot "dist\pySiberMan\_internal\templates\interrupt"),
    (Join-Path $projectRoot "dist\pySiberMan\_internal\player"),
    (Join-Path $projectRoot "dist\pySiberMan\_internal\sherpa\keywords.txt"),
    (Join-Path $projectRoot "dist\pySiberMan\_internal\pvrecorder\lib\windows\amd64\libpv_recorder.dll")
)

Write-Host "[4/6] 校验关键打包产物"
foreach ($path in $verifyPaths) {
    if (-not (Test-Path $path)) {
        throw "关键产物缺失: $path"
    }
}

if (-not $SkipInno) {
    Write-Host "[5/6] 执行 Inno Setup"
    $innoStart = Get-Date
    $ok = $false
    for ($i = 1; $i -le 3; $i++) {
        try {
            Invoke-Checked -Name "Inno Setup(第 $i 次)" -FilePath $isccExe -Arguments @($issFile)
            $ok = $true
            break
        } catch {
            if ($i -eq 3) { throw }
            Start-Sleep -Seconds 3
        }
    }
    if (-not $ok) {
        throw "Inno Setup 多次重试后仍失败"
    }
} else {
    Write-Host "[5/6] 跳过 Inno Setup"
}

Write-Host "[6/6] 输出最新安装包"
$setupCandidates = Get-ChildItem -Path (Join-Path $projectRoot "output\pySiberMan-setup-*.exe") -ErrorAction Stop
if ($SkipInno) {
    $latestSetup = $setupCandidates | Sort-Object LastWriteTime -Descending | Select-Object -First 1
} else {
    $latestSetup = $setupCandidates |
        Where-Object { $_.LastWriteTime -ge $innoStart.AddSeconds(-1) } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

if (-not $latestSetup) {
    throw "未找到安装包输出文件"
}

Write-Host "安装包路径: $($latestSetup.FullName)"
Write-Host "安装包大小: $([Math]::Round($latestSetup.Length / 1MB, 2)) MB"
Write-Host "完成"
