# CUDA Build and Profiling Script

# Set compiler, flags, and filenames
$nvcc = "nvcc"
$flags = "-G -g"
$output = "d7.exe"
$sources = "main.cu"

# Build command
$buildCmd = "$nvcc $flags $sources -o $output"

# Execute the build
Write-Host "Building CUDA program..."
Invoke-Expression $buildCmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!"

    # Run the program and show output
    Write-Host "`nRunning program: .\$output `n"
    & .\$output  # Corrected execution

    # Run with nvprof and display profiling output
    Write-Host "`nProfiling with nvprof (kernel execution times)... `n"
    & nvprof .\$output  # Corrected execution

    # Run nvprof with detailed metrics
    Write-Host "`nRunning nvprof with detailed performance metrics... `n"
    & nvprof --metrics all .\$output  # Corrected execution

} else {
    Write-Host "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
