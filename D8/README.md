# 🚀 CUDA 100-Day Challenge – Day 8  

## **Optimizing Hierarchical Reduction: Dynamic Block Size & Occupancy Analysis**  

## **Objective**  
On **Day 8**, I focused on refining **hierarchical reduction** by:  
- **Dynamically selecting block sizes** based on **CUDA occupancy analysis**
- **Minimizing atomic operations** to reduce overhead
- **Running multiple trials** to ensure consistent performance measurements
- **Using `cudaOccupancyMaxPotentialBlockSize()`** to find the optimal configuration
- **Profiling performance** with **NVIDIA Nsight Compute & nvprof**

Additionally, **Day 7** discussions on **warp shuffle optimizations** were integrated into this approach.  

---

## **1. Understanding Hierarchical Reduction & Warp Shuffle Optimization**  

### **What I Did:**  
Hierarchical reduction **combines warp shuffle and shared memory** to optimize summation.  
However, **block size and register pressure** are critical factors in **GPU performance**.  

- **Warp Shuffle Optimization** ✅
    - Used **`__shfl_xor_sync()`** for efficient **intra-warp reduction**
    - Reduced **atomic operations** to minimize **global memory contention**

- **Dynamic Block Size Selection** ✅
    - Used `cudaOccupancyMaxPotentialBlockSize()` to determine **best block size**
    - Tested **multiple block sizes (128, 192, 256, 320, 512, 1024)**
    - Chose the **fastest execution time** dynamically

- **Reducing Register Pressure** ✅
    - High register usage **reduces warp occupancy** → optimized register allocation
    - Analyzed register usage with **`nvcc -Xptxas=-v`** and Nsight Compute

- **Multiple Trials for Stability** ✅
    - Each kernel executed **10 times**, averaging execution time
    - Reduced **fluctuations** caused by **GPU scheduling variations**

## 📌 **Key Takeaways:**  
- **Warp shuffle significantly reduces reduction overhead**
- **Dynamic block size selection improves performance**
- **Register pressure must be carefully managed**
- **Running multiple trials ensures stable and reliable performance measurements**

---
## **2. How Dynamic Block Size Optimization Works**  
The **best block size** maximizes **GPU occupancy** while avoiding excessive **register pressure**.  

### **Steps Taken to Find the Best Block Size**  
1. **Used `cudaOccupancyMaxPotentialBlockSize()`** to determine **optimal launch configuration**:  
   ```cpp
   CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &maxBlockSize, 
           dotProductHierarchical, 
           (BLOCK_SIZE / WARP_SIZE) * sizeof(float), 
           0));
   ```

   - Ensures the best balance of active warps per multiprocessor
   - Prevents register overflow and unnecessary memory accesses

2. **Tested multiple block sizes (128, 192, 256, 320, 512, 1024)**
   - Measured execution time and selected the **fastest block size dynamically**
3. Used multiple trials `(NUM_TRIALS = 10)` to stabilize performance measurements

---

## **3. Code Overview & File Structure**
### Files Used:
📌 `main.cu` – Implements hierarchical reduction with warp shuffle + dynamic block size selection.

### Optimization Techniques Applied:
- ✅ **Hierarchical Reduction** – Combines warp shuffle & shared memory for efficiency
- ✅ **Warp Shuffle** (`__shfl_xor_sync`) – Reduces intra-warp communication overhead
- ✅ **Dynamic Block Size Selection** (`cudaOccupancyMaxPotentialBlockSize`) – Maximizes GPU occupancy
- ✅ **Multiple Trials** (10 runs per block size) – Ensures consistent execution time measurements

---

## 📊 **Performance Breakdown by Block Size**

| Block Size | Execution Time (ms) | Speedup Over CPU |
|------------|---------------------|------------------|
| 128        | 0.972 - 0.801       | 20.18x - 25.75x  |
| 192        | 0.959 - 0.829       | 20.45x - 24.88x  |
| 256        | 0.774 - 1.051       | 26.63x - 18.89x  |
| 320        | 0.789 - 0.998       | 26.13x - 20.30x  |
| 384        | 0.787 - 0.909       | 26.20x - 22.27x  |
| 512        | 0.775 - 0.883       | 26.61x - 22.35x  |
| 640        | 0.793 - 0.846       | 26.07x - 23.95x  |
| 768        | 0.805 - 0.840       | 25.62x - 23.48x  |
| 1024       | 0.776 - 0.821       | 26.36x - 23.90x  |

🔹 Block Size 256 and 1024 consistently performed the best  
🔹 Small variations in execution time were observed due to GPU resource contention and memory access patterns

---

## **5. Understanding Register Pressure & Its Impact**
### **What is Register Pressure?**
- Too many registers per thread → Low occupancy, spilling to local memory
- Too few registers per thread → More memory accesses, reducing performance

### **How We Optimized Register Usage**
- ✔ Used `nvcc -Xptxas=-v` to check register usage per thread
- ✔ Avoided unnecessary local variables that increase register demand
- ✔ Used shared memory to offload register demand
- ✔ Balanced register allocation to avoid spilling

---

## **6. Key Learnings from Day 8**
- ✅ Dynamic block size selection improves performance significantly
- ✅ Warp shuffle minimizes unnecessary memory accesses
- ✅ Avoiding register pressure is crucial for maximizing warp occupancy
- ✅ Running multiple trials eliminates fluctuations in execution time
- ✅ Using shared memory efficiently reduces register pressure

---

## **7. Next Steps (Day 9)**
- 🔹 Profile with Nsight Compute to analyze warp execution time
- 🔹 Experiment with additional block sizes (e.g., 640, 768, 896) to refine performance
- 🔹 Further optimize memory access patterns using different shuffle strategies (`__shfl_down_sync`, `__shfl_xor_sync`)
- 🔹 Explore alternative reduction approaches to further reduce atomic operations