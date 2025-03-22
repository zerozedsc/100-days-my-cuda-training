# CUDA Build and Profiling Script

# Set compiler, flags, and filenames
#Turing GPUs (RTX 20xx, GTX 1660, etc.) → -arch=sm_75
#Ampere GPUs (RTX 30xx) → -arch=sm_80
#Ada Lovelace (RTX 40xx) → -arch=sm_89
#Older Pascal GPUs (GTX 10xx) → -arch=sm_61
$nvcc = "nvcc"
$flags = "-O3 --use_fast_math -arch=sm_61 -Xptxas=-v"  # Optimized for performance
$output = "d8.exe"
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
    & .\$output  # Run executable

    # Check if nvprof exists before running profiling
    if (Get-Command nvprof -ErrorAction SilentlyContinue) {
        Write-Host "`nProfiling with nvprof (kernel execution times)... `n"
        & nvprof .\$output  

        Write-Host "`nRunning nvprof with detailed performance metrics... `n"
        & nvprof --metrics all .\$output  
    } else {
        Write-Host "`n⚠ nvprof not found! Consider using NVIDIA Nsight Systems (nsys) instead."
    }
} else {
    Write-Host "❌ Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}
