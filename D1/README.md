# Day 1: Setting Up the CUDA Development Environment

## Summary
On Day 1, I set up the CUDA development environment on my Windows 11 machine. This involved installing the CUDA Toolkit, configuring the system `PATH` to include the necessary directories, and verifying the installation. Additionally, I ensured that the Microsoft Visual Studio Build Tools were installed and properly configured, as they are required for compiling CUDA programs on Windows.

---

## Detailed Steps

### 1. **Install the CUDA Toolkit**
   - Downloaded the CUDA Toolkit from the [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit).
   - Ran the installer and selected the following options:
     - **Custom Installation** to avoid unnecessary components.
     - **CUDA Toolkit** (ensure this is checked).
     - **CUDA Samples** (optional but useful for testing).
   - Completed the installation and verified the installation directory:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.1
     ```

### 2. **Add CUDA to the System PATH**
   - Opened the **Environment Variables** settings:
     - Pressed `Win + S`, typed `Environment Variables`, and selected **Edit the system environment variables**.
     - In the **System Properties** window, clicked **Environment Variables**.
   - Added the following directories to the `Path` variable under **System variables**:
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.1\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.1\libnvvp
     ```
   - Verified the changes by opening a new PowerShell window and running:
     ```powershell
     nvcc --version
     ```
     Output:
     ```
     nvcc: NVIDIA (R) Cuda compiler driver
     Copyright (c) 2005-2025 NVIDIA Corporation
     Built on Fri_Feb_21_20:42:46_Pacific_Standard_Time_2025
     Cuda compilation tools, release 12.8, V12.8.93
     Build cuda_12.8.r12.8/compiler.35583870_0
     ```

### 3. **Install Microsoft Visual Studio Build Tools**
   - Downloaded and installed the **Visual Studio Build Tools** from the [Microsoft website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
   - Selected the **Desktop development with C++** workload during installation.
   - Ensured the following components were installed:
     - **MSVC (Microsoft Visual C++) compiler**
     - **Windows SDK**
     - **C++ CMake tools for Windows** (optional but recommended).

### 4. **Add `cl.exe` to the System PATH**
   - Located the `cl.exe` compiler in the Visual Studio Build Tools directory:
     ```
     C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64
     ```
   - Added this directory to the system `PATH` using PowerShell (run as Administrator):
     ```powershell
     $pathToAdd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"
     $systemPath = [Environment]::GetEnvironmentVariable('Path', [EnvironmentVariableTarget]::Machine)
     if ($systemPath -split ';' -notcontains $pathToAdd) {
         $newPath = $systemPath + ";" + $pathToAdd
         [Environment]::SetEnvironmentVariable('Path', $newPath, [EnvironmentVariableTarget]::Machine)
         Write-Host "Added '$pathToAdd' to the system PATH."
     } else {
         Write-Host "'$pathToAdd' is already in the system PATH."
     }

### 5. **Test the CUDA Installation**
   - Created a simple CUDA program (`hello.cu`):
     ```
     Output:
     Hello World from GPU!
     ```

---

## Key Takeaways
- Successfully installed the CUDA Toolkit and Visual Studio Build Tools.
- Configured the system `PATH` to include CUDA and MSVC compiler directories.
- Verified the installation by compiling and running a simple CUDA program.

---