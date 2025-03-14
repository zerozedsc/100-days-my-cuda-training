# 🚀 CUDA 100-Day Challenge – Day 2  

## **Exploring GPU Architecture & Comparing CPU vs. GPU Performance**

## **Objective**  
On Day 2 of my CUDA challenge, I focused on:  
- Understanding **GPU architecture** and **CUDA programming basics**.  
- Exploring **device properties** to gain insights into my GPU’s capabilities.  
- Implementing **basic arithmetic operations** on the GPU and comparing them with CPU execution.  

---

## **1. Querying GPU System Information**  
### **What I Did:**  
To better understand how my GPU works, I wrote a program to retrieve **hardware specifications** using CUDA’s built-in API.  

- **Compute Capability:** 6.1  
  This indicates the GPU's support for specific CUDA features and performance metrics. A compute capability of 6.1 means the GTX 1060 is built on NVIDIA's Pascal architecture, supporting advanced features like enhanced memory compression and improved instruction throughput.

- **Multiprocessors:** 10  
  The GTX 1060 contains 10 Streaming Multiprocessors (SMs), each responsible for executing parallel threads. Each SM in the Pascal architecture includes 128 CUDA cores, totaling 1,280 CUDA cores for this GPU, enabling efficient parallel processing.

- **Max Threads per Block:** 1024  
  This defines the maximum number of threads that can be organized within a single block, allowing for flexible parallel workload management.

- **Global Memory:** ~3.2 GB  
  The GPU offers approximately 3.2 GB of global memory, used for storing data accessible by all threads during kernel execution.

- **Warp Size:** 32  
  A warp consists of 32 threads that execute instructions simultaneously. Efficient CUDA programs ensure that threads within a warp follow the same execution path to maximize performance.

*References:*
- [NVIDIA GeForce GTX 1060 Specifications](https://de.wikipedia.org/wiki/Nvidia-GeForce-10-Serie)
- [NVIDIA Pascal Architecture Overview](https://fr.wikipedia.org/wiki/Pascal_%28microarchitecture%29)

This information helps determine how to optimize CUDA programs for **best performance**.  

## **2.Code Overview & File Structure**

- **`main.cu`:**  
  Contains the main CUDA program. It initializes the GPU, executes arithmetic operations, and prints the results including system info and execution times.

- **`query_info.cu`:**  
  Likely contains functions to query and display GPU information such as device name, compute capability, memory properties, etc.

- **`share.h`:**  
  Holds shared declarations (for example, the prototype for `querySystemInfo()`). 

---

## **3. Implementing CUDA vs CPU Arithmetic Operations**  
### **What I Did:**  
I wrote a **CUDA kernel** to perform **basic arithmetic operations (+, -, *, /)** on large matrices (2048x2048) and compared execution times with **CPU calculations**.  

### **Observations from the Output:**  
| Operation | GPU Execution Time (ms) | CPU Execution Time (ms) | Speedup |
|-----------|------------------------|------------------------|---------|
| Addition (+) | 13.69 | 20.66 | ~1.5x Faster |
| Subtraction (-) | 10.63 | 18.42 | ~1.7x Faster |
| Multiplication (*) | 15.23 | 24.44 | ~1.6x Faster |
| Division (/) | 15.69 | 32.69 | ~2.1x Faster |

🚀 **CUDA significantly speeds up matrix operations, especially multiplication and division.**  

📌 **Related File:** `main.cu`  

---

## **. Key Learnings from Day 2**  
✅ **CUDA excels in parallel computing** – Handling thousands of threads at once boosts performance.  

✅ **Device properties matter** – Understanding GPU limitations helps optimize CUDA programs.  

✅ **Memory access is crucial** – Efficient memory usage can **further improve execution speed**.  

---

## **Next Steps (Day 3)**  
🔹 Explore **memory management in CUDA (global, shared, constant)**.  

🔹 Implement **optimized memory access techniques** for better performance.  

