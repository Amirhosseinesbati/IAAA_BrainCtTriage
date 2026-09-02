[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$HostName,
    [Parameter(Mandatory = $true)][ValidateRange(1, 65535)][int]$Port,
    [Parameter(Mandatory = $true)][string]$IdentityFile,
    [Parameter(Mandatory = $true)][string]$KnownHostsFile,
    [Parameter(Mandatory = $true)][string]$RemoteManifest,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{64}$')][string]$ExpectedManifestSha256,
    [Parameter(Mandatory = $true)][string]$LocalDirectory,
    [string]$UserName = 'root'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$requiredArtifacts = @(
    'training_manifest',
    'launcher_status',
    'fixed_epoch_checkpoint',
    'report',
    'epoch_metrics',
    'run_log'
)

function Assert-SafeRemotePath([string]$PathValue) {
    if (-not $PathValue.StartsWith('/workspace/', [StringComparison]::Ordinal)) {
        throw "Remote artifact path must stay under /workspace/: $PathValue"
    }
    if ($PathValue.IndexOfAny([char[]]"`"`r`n") -ge 0) {
        throw 'Remote artifact path contains unsafe quoting or newline characters'
    }
}

function ConvertTo-SftpLocalPath([string]$PathValue) {
    $normalized = [IO.Path]::GetFullPath($PathValue).Replace('\', '/')
    if ($normalized.IndexOfAny([char[]]"`"`r`n") -ge 0) {
        throw 'Local artifact path contains unsafe quoting or newline characters'
    }
    return $normalized
}

function Invoke-SftpBatch([string[]]$Lines) {
    $batchPath = [IO.Path]::GetTempFileName()
    try {
        [IO.File]::WriteAllLines($batchPath, $Lines, [Text.UTF8Encoding]::new($false))
        $arguments = @(
            '-i', (Resolve-Path -LiteralPath $IdentityFile).Path,
            '-o', 'IdentitiesOnly=yes',
            '-o', "UserKnownHostsFile=$((Resolve-Path -LiteralPath $KnownHostsFile).Path)",
            '-o', 'BatchMode=yes',
            '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=6',
            '-P', [string]$Port,
            '-b', $batchPath,
            "$UserName@$HostName"
        )
        & sftp @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "SFTP batch failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Remove-Item -LiteralPath $batchPath -Force -ErrorAction SilentlyContinue
    }
}

if (-not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
    throw "SSH identity file does not exist: $IdentityFile"
}
if (-not (Test-Path -LiteralPath $KnownHostsFile -PathType Leaf)) {
    throw "Known-hosts file does not exist: $KnownHostsFile"
}
Assert-SafeRemotePath $RemoteManifest

$localRoot = [IO.Path]::GetFullPath($LocalDirectory)
[IO.Directory]::CreateDirectory($localRoot) | Out-Null
$manifestLocal = Join-Path $localRoot 'transfer_manifest.json'
$manifestSftpLocal = ConvertTo-SftpLocalPath $manifestLocal
Invoke-SftpBatch @("reget `"$RemoteManifest`" `"$manifestSftpLocal`"")

$actualManifestSha256 = (Get-FileHash -LiteralPath $manifestLocal -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualManifestSha256 -ne $ExpectedManifestSha256) {
    throw "Downloaded manifest checksum mismatch: $actualManifestSha256"
}

$manifest = Get-Content -LiteralPath $manifestLocal -Raw | ConvertFrom-Json
if ($manifest.schema_version -ne 1 -or $manifest.status -ne 'ready_for_checksum_transfer') {
    throw 'Transfer manifest is not ready or uses an unsupported schema'
}
$artifactNames = @($manifest.artifacts.PSObject.Properties.Name | Sort-Object)
$expectedNames = @($requiredArtifacts | Sort-Object)
if (Compare-Object -ReferenceObject $expectedNames -DifferenceObject $artifactNames) {
    throw 'Transfer manifest does not contain exactly the six canonical artifacts'
}

$fixedEpoch = [int]$manifest.fixed_audit_epoch
$canonicalNames = @{
    training_manifest      = 'training_manifest.yaml'
    launcher_status        = 'launcher_status.json'
    fixed_epoch_checkpoint = ('mls_multitask_epoch_{0:D3}.pth' -f $fixedEpoch)
    report                 = 'report.md'
    epoch_metrics          = 'epoch_metrics.jsonl'
    run_log                = 'run.log'
}
$downloadLines = [Collections.Generic.List[string]]::new()
foreach ($name in $requiredArtifacts) {
    $metadata = $manifest.artifacts.$name
    $remotePath = [string]$metadata.source_path
    $fileName = [string]$metadata.transfer_filename
    Assert-SafeRemotePath $remotePath
    if ($fileName -ne $canonicalNames[$name] -or [IO.Path]::GetFileName($fileName) -ne $fileName) {
        throw "Unsafe or non-canonical transfer filename for $name"
    }
    $destination = Join-Path $localRoot $fileName
    $destinationSftpLocal = ConvertTo-SftpLocalPath $destination
    $downloadLines.Add("reget `"$remotePath`" `"$destinationSftpLocal`"")
}
Invoke-SftpBatch $downloadLines.ToArray()

$verificationPath = Join-Path $localRoot 'transfer_verification.json'
& uv run python scripts/verify_mls_run_transfer.py `
    --manifest $manifestLocal `
    --expected-manifest-sha256 $ExpectedManifestSha256 `
    --artifact-dir $localRoot `
    --output $verificationPath
if ($LASTEXITCODE -ne 0) {
    throw "Local MLS transfer verifier failed with exit code $LASTEXITCODE"
}

$verification = Get-Content -LiteralPath $verificationPath -Raw | ConvertFrom-Json
$verifiedCheckCount = @($verification.checks.PSObject.Properties).Count
if ($verification.status -ne 'verified' -or $verification.artifacts_expected -ne 6 -or $verification.artifacts_verified -ne 6 -or $verifiedCheckCount -ne 6) {
    throw 'Local MLS transfer verification did not prove 6/6 artifacts'
}
[pscustomobject]@{
    status = 'verified'
    manifest_sha256 = $actualManifestSha256
    artifacts_verified = [int]$verification.artifacts_verified
    local_directory = $localRoot
}
