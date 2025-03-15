# 🚀 CUDA 100-Day Challenge – Day 3  

## **Optimizing Memory Access: Global, Shared, and Constant Memory**  

## **Objective**  
On **Day 3**, I focused on optimizing memory access by comparing different **CUDA memory types** and their impact on performance.  
The goal was to:  
- Implement **matrix addition** using **global, shared, and constant memory**.  
- Measure execution times to analyze **performance differences**.  
- Ensure **constant memory fits within the 64KB limit** after previous errors.  

---

## **1. Understanding CUDA Memory Types**  
### **What I Did:**  
To optimize CUDA performance, I explored three key memory types:  

- **Global Memory** 🐢 (Accessible by all threads, slowest due to high latency).  
- **Shared Memory** ⚡ (Fast, shared within a block, reduces global memory accesses).  
- **Constant Memory** 🔥 (Read-only, cached, optimized for frequently accessed values).  

Each of these memory types has different access speeds, so choosing the right one can significantly improve execution time.  

*References:*  
- [CUDA Memory Model (NVIDIA Docs)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-hierarchy)  
- [CUDA Optimization Best Practices](https://developer.nvidia.com/blog/cuda-pro-tip-optimize-global-memory-access-with-these-strategies/)  

---

## **2. Code Overview & File Structure**  

### **Files Used:**  
📌 **`main.cu`** – Implements matrix addition using **global, shared, and constant memory**.  

### **Memory Type Breakdown:**  
- **Global Memory:** Standard CUDA implementation using `cudaMalloc()` and `cudaMemcpy()`.  
- **Shared Memory:** Uses `__shared__` memory to **reduce global memory accesses**.  
- **Constant Memory:** Uses `__constant__` memory but now **fits within the 64KB limit** by storing only a portion of the matrix.  

---

## **3. Observing Execution Times**  
### **Latest Results:**
| Memory Type | Execution Time (ms) |
|-------------|--------------------|
| **Global Memory** | 2.32 ms |
| **Shared Memory** | 0.93 ms |
| **Constant Memory** | 0.88 ms |

### **Analysis:**  
✔️ **Global memory execution time has significantly improved** (previously **27.67 ms**, now **2.32 ms**).  
✔️ **Shared memory is still faster than global memory** due to **low-latency access within a block**.  
✔️ **Constant memory performance is similar to shared memory**, as expected, since many threads **read the same values** from constant memory.  

---

## **4. Key Learnings from Day 3**  
✅ **Global memory optimizations (coalescing) significantly improve performance.**  
✅ **Shared memory remains one of the best ways to optimize repeated access patterns.**  
✅ **Constant memory is effective when many threads access the same values.**  
✅ **Choosing the right memory type can lead to over 10x performance improvement.**  

---

## **5. Next Steps (Day 4)**  
🔹 Further optimize **global memory access patterns** (e.g., coalescing).  
🔹 Implement **matrix multiplication** with optimized memory usage.  
🔹 Compare **multi-GPU performance** for memory-intensive tasks.  


