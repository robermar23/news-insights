param(
    [string]$ProjectRoot = "D:\\code\\news-insights",
    [string]$TaskNameIngest = "news-insights-ingest",
    [string]$TaskNamePredict = "news-insights-predict"
)

$ingestScript = Join-Path $ProjectRoot "scripts/ingest.ps1"
$predictScript = Join-Path $ProjectRoot "scripts/predict.ps1"

$actionIngest = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ingestScript`""
$triggerIngest = New-ScheduledTaskTrigger -Daily -At 3am

$actionPredict = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$predictScript`""
$triggerPredict = New-ScheduledTaskTrigger -Daily -At 4am

Register-ScheduledTask -TaskName $TaskNameIngest -Action $actionIngest -Trigger $triggerIngest -Description "Ingest news articles nightly" -User "$env:USERNAME" -RunLevel LeastPrivilege -Force
Register-ScheduledTask -TaskName $TaskNamePredict -Action $actionPredict -Trigger $triggerPredict -Description "Predict on ingested articles nightly" -User "$env:USERNAME" -RunLevel LeastPrivilege -Force

