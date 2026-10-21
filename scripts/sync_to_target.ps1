# 把项目从开发目录同步到目标目录（例如 E:\项目一）
#
# 用法：
#   powershell -ExecutionPolicy Bypass -File scripts\sync_to_target.ps1 -Target "E:\项目一"
#
# 说明：
#   - 只复制源码/文档/示例库与（可选）评测结果，跳过 __pycache__ / pytest 缓存 / 虚拟环境 / traces
#   - /XO 跳过目标端更新的同名文件，不会删除目标端多余文件

param(
    [Parameter(Mandatory = $true)][string]$Target,
    [switch]$IncludeResults,     # 需要把评测结果（JSONL/报告）一起带过去时加上
    [switch]$DryRun              # 只看会复制什么，不实际写
)

$source = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path -LiteralPath $source)) { throw "源目录不存在: $source" }

# 排除项一律用「目录名模式」，不要用带通配符的完整路径：
# robocopy 的 /XD 会拒绝 `E:\path\pytest-cache-files-*` 这种参数（退出码 16）。
$excludeDirs = @("__pycache__", ".pytest_cache", ".venv", "venv", "traces", "pytest-cache-files-*")
if (-not $IncludeResults) { $excludeDirs += "results" }
$xd = $excludeDirs

# 注意：不要把参数数组命名为 $args（PowerShell 自动变量）；
# 也不要用行尾 `+` 续行（Windows PowerShell 5.1 不支持），逐项追加最稳。
$rcArgs = @($source, $Target, "/E", "/XO", "/NFL", "/NDL", "/NJH", "/NJS", "/NP")
$rcArgs += "/XD"
foreach ($d in $xd) { $rcArgs += $d }
$rcArgs += "/XF"
$rcArgs += "*.pyc"
$rcArgs += "*.log"
if ($DryRun) { $rcArgs += "/L" }

Write-Host "源      : $source" -ForegroundColor Cyan
Write-Host "目标    : $Target" -ForegroundColor Cyan
Write-Host "排除目录: $($excludeDirs -join ', ')" -ForegroundColor DarkGray

if (-not (Test-Path -LiteralPath $Target)) {
    if ($DryRun) { Write-Host "[DryRun] 将创建目标目录" } else { New-Item -ItemType Directory -Force -Path $Target | Out-Null }
}

robocopy @rcArgs | Out-Null
$code = $LASTEXITCODE
if ($code -ge 8) {
    Write-Host "[FAIL] robocopy 退出码 $code" -ForegroundColor Red
    exit 1
}

if (-not $DryRun) {
    $files = Get-ChildItem -LiteralPath $Target -Recurse -File -ErrorAction SilentlyContinue
    $totalKB = [math]::Round((($files | Measure-Object Length -Sum).Sum / 1KB), 0)
    Write-Host ("完成：robocopy 退出码 {0}（0~7 均为成功）｜目标端 {1} 个文件，{2} KB" -f $code, $files.Count, $totalKB) -ForegroundColor Green
    Get-ChildItem -LiteralPath $Target -Force | Select-Object Mode, Name | Format-Table -AutoSize | Out-String | Write-Host
}
exit 0
