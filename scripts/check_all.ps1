# 一键自检：题库校验 -> 单测 -> 阶段验收 -> trace 校验
# 用法： powershell -ExecutionPolicy Bypass -File scripts\check_all.ps1
# 注意：本脚本不写 C 盘；所有产物都落在项目目录内。

param(
    [string]$Python = "C:\Users\22855\miniconda3\envs\llm\python.exe",
    [switch]$SkipLlm   # 跳过需要真实模型的阶段二验收
)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Step($name, $block) {
    Write-Host "`n================ $name ================" -ForegroundColor Cyan
    & $block
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $name (exit $LASTEXITCODE)" -ForegroundColor Red
        $script:failed += 1
    } else {
        Write-Host "[ OK ] $name" -ForegroundColor Green
    }
}

$script:failed = 0

Step "1/5 生成示例库" { & $Python -m app.db.seed }
Step "2/5 校验评测题库（60 题的标准 SQL）" { & $Python scripts\build_eval_set.py }
Step "3/5 单元测试（护栏/指标/召回/Agent/API）" { & $Python -m pytest tests -q -p no:cacheprovider }
Step "4/5 trace 记账与逐题归零校验" { & $Python scripts\verify_trace.py }
Step "5/5 阶段一验收（护栏 + 执行器超时中断）" { & $Python scripts\verify_phase1.py }

if (-not $SkipLlm) {
    Step "附加 阶段二验收（真实模型：Schema Linking + 候选投票，约 1~6 分钟）" {
        & $Python scripts\verify_phase2.py
    }
}

Write-Host ""
if ($script:failed -eq 0) {
    Write-Host "全部检查通过 ✅" -ForegroundColor Green
    exit 0
} else {
    Write-Host "有 $script:failed 个步骤失败 ❌" -ForegroundColor Red
    exit 1
}
