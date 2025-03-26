# 🚀 CUDA 100-Day Challenge – Day 9  

## **Parallel Prefix Sum Optimization (Global, Shared, and Warp-Level)**  

## **Objective**  
On **Day 9**, I focused on optimizing **parallel prefix sum (scan)** operations using different memory access strategies in CUDA.  
The goal was to:  
- Implement **global memory (Hillis-Steele) prefix sum**.  
- Optimize **shared memory** prefix sum using the **Hillis-Steele** approach.  
- Utilize **warp shuffle operations** for the fastest implementation.  
- Compare performance against a **CPU-based prefix sum**.  

---

## **1. Understanding Parallel Prefix Sum Optimization**  
### **What I Did:**  
Prefix sum (or scan) is a fundamental parallel computing operation used in:  
🔹 **Parallel computing algorithms** (sorting, filtering, graph traversal).  
🔹 **Financial modeling** (cumulative sums, running totals).  
🔹 **Scientific computing** (data reduction, inclusive/exclusive scans).  

---

## **2. Implementing and Comparing Different Prefix Sum Approaches**  

### **CPU Prefix Sum (Baseline)**
- **Naïve sequential approach.**
- **Time Complexity:** **O(N)**
- Uses **simple iteration** to accumulate sums.
- Acts as a **baseline for GPU comparisons**.

### **Global Memory Prefix Sum (Hillis-Steele)**
- Implements the **Hillis-Steele** method.
- Uses **global memory reads/writes** at every step.
- **Time Complexity:** **O(N log N)**
- **High memory bandwidth usage**, leading to slower execution.

### **Shared Memory Prefix Sum (Hillis-Steele)**
- Optimized **Hillis-Steele** implementation **using shared memory**.
- **Reduces redundant global memory accesses**.
- **Time Complexity:** **O(N log N)**
- **Better performance** but limited by shared memory size.

### **Warp Shuffle Prefix Sum (Optimized)**
- Uses **warp shuffle intrinsics** to **eliminate shared memory** overhead.
- **Each warp computes its own prefix sum** efficiently.
- **Time Complexity:** **O(N)**
- **Fastest approach**, achieving **3-4x speedup** over shared/global memory methods.

---

## **3. Execution Time Comparisons (N = 1,048,576 elements)**  

| Method | Execution Time (ms) | Speedup vs. CPU |
|----------------------|----------------|------------------|
| **CPU Prefix Sum** | **1.26 - 1.49 ms** | **Baseline** |
| **Global Memory Prefix Sum (Hillis-Steele)** | **0.88 - 0.91 ms** | **1.4x faster** |
| **Shared Memory Prefix Sum (Hillis-Steele)** | **0.88 - 0.91 ms** | **1.4x faster** |
| **Warp Shuffle Prefix Sum (Optimized)** | **0.26 - 0.29 ms** | **4.8x faster** |

---

## **4. Key Observations & Insights**  

✅ **Warp Shuffle Optimization is the fastest**  
- ~4.8x **speedup over the CPU approach**.  
- ~3.3x **speedup compared to Global/Shared Memory implementations**.  

✅ **Shared Memory does not significantly outperform Global Memory**  
- **Similar performance** between **global and shared memory** suggests that:  
  - **Global memory accesses are already coalesced**.  
  - **Shared memory adds some synchronization overhead**.  

✅ **CPU Performance is slightly inconsistent**  
- CPU execution time **fluctuates between 1.26 - 1.49 ms** due to:  
  - OS scheduling.  
  - Cache variations.  
  - Memory allocation fluctuations.  

---

## **5. Code Implementation & File Structure**  

### **Files Used:**  
📌 **main.cu** – Implements **all prefix sum approaches** (CPU, Global, Shared, Warp Shuffle).  

### **Implementation Summary:**  
- **CPU Prefix Sum** ✅ – **Baseline implementation (O(N))**  
- **Global Memory Prefix Sum (Hillis-Steele)** ❌ – **Slower due to memory bandwidth limits (O(N log N))**  
- **Shared Memory Prefix Sum (Hillis-Steele)** ✅ – **Better performance but not much improvement over Global (O(N log N))**  
- **Warp Shuffle Prefix Sum (Optimized)** 🚀 – **Fastest (O(N)), minimal memory overhead**  

---

## **6. Next Steps for Optimization**  
🔹 **Increase `N`** to test **scalability** at **1M+ elements**.  
🔹 **Use multi-block parallelism** to further improve shared memory performance.  
🔹 **Hybrid approach**:  
   - Use **Warp Shuffle** for **small sections**.  
   - Use **Multi-Block Reduction** for **large sections** (like **Thrust** does).  

---

## **7. Key Learnings from Day 9**  
✅ **Warp Shuffle intrinsics significantly improve performance**.  
✅ **Shared Memory does not always guarantee a performance boost**.  
✅ **Global memory optimization depends on access pattern and coalescing**.  
✅ **Choosing the right optimization depends on input size & hardware constraints**.  

---

## **8. Next Steps (Day 10 Preview)**  
🔹 Implement **multi-block prefix sum for even larger datasets**.  
🔹 Optimize **load balancing and dynamic memory handling**.  
🔹 Test **real-world applications using prefix sum (e.g., histogram equalization, parallel sorting, stream compaction)**.  

---
