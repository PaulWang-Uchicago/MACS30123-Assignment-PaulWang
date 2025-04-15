import numpy as np
import rasterio
import time
import matplotlib.pyplot as plt

# Import PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Load the Satellite Bands
red_band_path = '/project/macs30123/landsat8/LC08_B4.tif'  # red
nir_band_path = '/project/macs30123/landsat8/LC08_B5.tif'   # nir

with rasterio.open(red_band_path) as src:
    red_band = src.read(1).astype(np.float64)

with rasterio.open(nir_band_path) as src:
    nir_band = src.read(1).astype(np.float64)

# Prepare Data for GPU Computation
red_band_flat = red_band.ravel()
nir_band_flat = nir_band.ravel()
num_pixels = np.int32(red_band_flat.size)
ndvi_result_flat = np.empty_like(red_band_flat)

# Initialize GPU
start_gpu = time.time()

# CUDA Kernel
ndvi_kernel_code = """
__global__ void compute_ndvi(const double *red, const double *nir, double *ndvi, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double r = red[idx];
        double n = nir[idx];
        double total = r + n;
        if (total != 0.0)
            ndvi[idx] = (n - r) / total;
        else
            ndvi[idx] = 0.0;
    }
}
"""

# Compile the kernel
mod = SourceModule(ndvi_kernel_code)
compute_ndvi = mod.get_function("compute_ndvi")

# Allocate GPU memory for inputs and output
red_gpu = cuda.mem_alloc(red_band_flat.nbytes)
nir_gpu = cuda.mem_alloc(nir_band_flat.nbytes)
ndvi_gpu = cuda.mem_alloc(ndvi_result_flat.nbytes)

# Transfer input data to GPU
cuda.memcpy_htod(red_gpu, red_band_flat)
cuda.memcpy_htod(nir_gpu, nir_band_flat)

# Set block and grid sizes
block_size = 512
grid_size = ((int(num_pixels) + block_size - 1) // block_size, 1)

# Launch the kernel
compute_ndvi(
    cuda.In(red_band_flat),
    cuda.In(nir_band_flat),
    cuda.Out(ndvi_result_flat),
    num_pixels,
    block=(block_size, 1, 1),
    grid=grid_size
)
cuda.Context.synchronize()
end_gpu = time.time()
gpu_time = end_gpu - start_gpu
print("GPU NDVI computation time: {:.6f} seconds".format(gpu_time))

# Compare with a Serial CPU Implementation
start_cpu = time.time()
cpu_result = np.where(
    (red_band_flat + nir_band_flat) != 0,
    (nir_band_flat - red_band_flat) / (red_band_flat + nir_band_flat),
    0.0
)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu
print("CPU NDVI computation time: {:.6f} seconds".format(cpu_time))

# Save and Display the NDVI Image
ndvi_result = ndvi_result_flat.reshape(red_band.shape)
plt.imsave("ndvi_gpu.png", ndvi_result)
plt.figure(figsize=(8, 6))
plt.imshow(ndvi_result)
plt.title("NDVI Calculated on GPU")
plt.colorbar(label='NDVI')
plt.tight_layout()
plt.show()