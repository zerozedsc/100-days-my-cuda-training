# 🚀 CUDA 100-Day Challenge – Day 1

## **Setting Up the CUDA Development Environment**

## **Objective**
On Day 1 of my CUDA challenge, I focused on:
- Installing the **CUDA Toolkit** on my Windows 11 system
- Configuring the system **PATH** for CUDA development
- Setting up the **Microsoft Visual Studio Build Tools**
- Verifying the installation with a simple CUDA program

---

## **1. Installing the CUDA Toolkit**
### **What I Did:**
I downloaded and installed the CUDA Toolkit from NVIDIA's website, setting up the foundation for CUDA development.

- **Installation Path:** `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.1`
- **Components Selected:**
  - CUDA Toolkit (core components)
  - CUDA Samples (for testing and reference)

*References:*
- [NVIDIA CUDA Toolkit website](https://developer.nvidia.com/cuda-toolkit)

## **2. Environment Configuration**

### **System PATH Setup:**
I added critical CUDA directories to the system PATH to ensure proper operation:
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.1\bin
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8.1\libnvvp
```

### **NVCC Verification:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:42:46_Pacific_Standard_Time_2025
Cuda compilation tools, release 12.8, V12.8.93
```

## **3. Installing Microsoft Visual Studio Build Tools**
### **What I Did:**
I installed the necessary Microsoft Visual C++ compiler tools required for CUDA development on Windows.

- **Components Installed:**
  - MSVC (Microsoft Visual C++) compiler
  - Windows SDK
  - C++ CMake tools for Windows

I added the compiler path to the system PATH:
```
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64
```

## **4. Testing the Installation**
### **What I Did:**
I created and compiled a simple CUDA "Hello World" program to verify the environment was correctly configured.

```
Output:
Hello World from GPU!
```

📌 **Related File:** `hello.cu`

---

## **Key Learnings from Day 1**
✅ **CUDA development requires specific tooling** - Both NVIDIA and Microsoft components are needed

✅ **Environment configuration is critical** - Proper PATH setup ensures tools work correctly

✅ **Simple test programs verify setup** - "Hello World" confirms the environment is ready for more complex tasks

---

## **Next Steps (Day 2)**
🔹 Explore **GPU architecture** and **CUDA programming basics**

🔹 Query **device properties** to understand my GPU's capabilities

🔹 Implement **basic arithmetic operations** on the GPU and compare with CPU
