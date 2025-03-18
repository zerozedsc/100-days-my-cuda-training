# CUDA build script

# Set compiler and flags
$nvcc = "nvcc"
$flags = "-G -g"
$output = "d5.exe"
$sources = "main.cu"

# Build command
$buildCmd = "$nvcc $flags $sources -o $output"

# Execute the build
Write-Host "Building with command: $buildCmd"
Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!"
    Write-Host "Running: .\$output"
    & ".\$output"
} else {
    Write-Host "Build failed with exit code $LASTEXITCODE"
}