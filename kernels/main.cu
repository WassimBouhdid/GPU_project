#include "main.cuh"

#include <cstdio>
#include <cmath>

#define BLOCK_W 16
#define BLOCK_H 16

// ============================================================================
// MÉMOIRE CONSTANTE — déclarée au niveau global du fichier
// Accessible depuis tous les kernels, cachée automatiquement
// ============================================================================
__constant__ CudaCamParams c_ref_params;           // paramètres de la caméra de référence
__constant__ CudaCamParams c_cam_params[MAX_CAMS]; // paramètres des caméras secondaires
__constant__ int c_cam_dims[MAX_CAMS * 2];         // [width0, height0, width1, height1, ...]
__constant__ int c_num_cams;                       // nombre de caméras secondaires


// ============================================================================
// 1. LE KERNEL (S'exécute en parallèle sur des milliers de cœurs du GPU)
// ============================================================================
__global__ void kernel_sweeping_plane_shared_const(
    uint8_t*  dev_ref_pixels,
    int       ref_width,
    int       ref_height,
    uint8_t** dev_pixel_ptrs,   // tableau de pointeurs vers les pixels des caméras
    float*    dev_cost_cube,
    int       z_planes,
    float     z_near,
    float     z_far,
    int       window
)
// Plus besoin de passer ref_params, dev_cams, num_cams en paramètre :
// tout ça vient de la constant memory
{
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z;

    int half_w = window / 2;

    // --- Shared memory pour le tile de référence (identique à avant) ---
    #define TILE_W (16 + 14)
    #define TILE_H (16 + 14)
    __shared__ uint8_t tile_ref[TILE_H][TILE_W];

    int tile_origin_x = blockIdx.x * 16 - half_w;
    int tile_origin_y = blockIdx.y * 16 - half_w;
    int tile_w = 16 + window - 1;
    int tile_h = 16 + window - 1;

    // Chargement coopératif du tile
    int tid         = threadIdx.y * 16 + threadIdx.x;
    int num_threads = 16 * 16;
    int num_pixels  = tile_w * tile_h;

    for (int i = tid; i < num_pixels; i += num_threads) {
        int ty = i / tile_w;
        int tx = i % tile_w;
        int gx = tile_origin_x + tx;
        int gy = tile_origin_y + ty;

        tile_ref[ty][tx] = (gx < 0 || gx >= ref_width || gy < 0 || gy >= ref_height)
                           ? 0
                           : dev_ref_pixels[gy * ref_width + gx];
    }

    __syncthreads();

    if (x >= ref_width || y >= ref_height || zi >= z_planes) return;

    // --- Calculs géométriques — lecture depuis CONSTANT MEMORY ---
    float z = z_near * z_far / (z_near + (((float)zi / (float)z_planes) * (z_far - z_near)));

    // c_ref_params est lu depuis le cache constant → pas de transaction DRAM
    float X_ref = (c_ref_params.K_inv[0]*x + c_ref_params.K_inv[1]*y + c_ref_params.K_inv[2]) * z;
    float Y_ref = (c_ref_params.K_inv[3]*x + c_ref_params.K_inv[4]*y + c_ref_params.K_inv[5]) * z;
    float Z_ref = (c_ref_params.K_inv[6]*x + c_ref_params.K_inv[7]*y + c_ref_params.K_inv[8]) * z;

    float X = c_ref_params.R_inv[0]*X_ref + c_ref_params.R_inv[1]*Y_ref + c_ref_params.R_inv[2]*Z_ref - c_ref_params.t_inv[0];
    float Y = c_ref_params.R_inv[3]*X_ref + c_ref_params.R_inv[4]*Y_ref + c_ref_params.R_inv[5]*Z_ref - c_ref_params.t_inv[1];
    float Z = c_ref_params.R_inv[6]*X_ref + c_ref_params.R_inv[7]*Y_ref + c_ref_params.R_inv[8]*Z_ref - c_ref_params.t_inv[2];

    float min_cost = 255.0f;

    for (int c = 0; c < c_num_cams; c++) {
        // Lecture des paramètres géométriques depuis constant memory
        // (broadcast sur tout le warp = 1 seule transaction)
        float X_proj = c_cam_params[c].R[0]*X + c_cam_params[c].R[1]*Y + c_cam_params[c].R[2]*Z - c_cam_params[c].t[0];
        float Y_proj = c_cam_params[c].R[3]*X + c_cam_params[c].R[4]*Y + c_cam_params[c].R[5]*Z - c_cam_params[c].t[1];
        float Z_proj = c_cam_params[c].R[6]*X + c_cam_params[c].R[7]*Y + c_cam_params[c].R[8]*Z - c_cam_params[c].t[2];

        float x_proj = c_cam_params[c].K[0]*X_proj/Z_proj + c_cam_params[c].K[1]*Y_proj/Z_proj + c_cam_params[c].K[2];
        float y_proj = c_cam_params[c].K[3]*X_proj/Z_proj + c_cam_params[c].K[4]*Y_proj/Z_proj + c_cam_params[c].K[5];

        int cam_w = c_cam_dims[c * 2];
        int cam_h = c_cam_dims[c * 2 + 1];

        x_proj = (x_proj < 0 || x_proj >= cam_w) ? 0 : roundf(x_proj);
        y_proj = (y_proj < 0 || y_proj >= cam_h) ? 0 : roundf(y_proj);

        // Pixels de la caméra : toujours en mémoire globale (pointeurs dynamiques)
        uint8_t* cam_pixels = dev_pixel_ptrs[c];

        float cost = 0.0f;
        float cc   = 0.0f;

        for (int k = -half_w; k <= half_w; k++) {
            for (int l = -half_w; l <= half_w; l++) {
                // Référence : depuis shared memory
                uint8_t ref_pixel = tile_ref[threadIdx.y + half_w + k]
                                            [threadIdx.x + half_w + l];

                int cx = (int)x_proj + l;
                int cy = (int)y_proj + k;
                if (cx < 0 || cx >= cam_w || cy < 0 || cy >= cam_h) continue;

                uint8_t cam_pixel = cam_pixels[cy * cam_w + cx];
                cost += abs((int)ref_pixel - (int)cam_pixel);
                cc   += 1.0f;
            }
        }

        if (cc > 0.0f) {
            cost /= cc;
            if (cost < min_cost) min_cost = cost;
        }
    }

    dev_cost_cube[zi * ref_height * ref_width + y * ref_width + x] = min_cost;
}

// ============================================================================
// 2. LE WRAPPER CPU (Gère la mémoire et orchestre le lancement du Kernel)
// ============================================================================
void wrap_sweeping_plane_cuda(
    CudaCamParams ref_params, uint8_t* host_ref_pixels, int ref_width, int ref_height,
    CudaCam* host_cams, int num_cams,
    float* host_cost_cube, int z_planes, float z_near, float z_far, int window
) {
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // --- Copie des paramètres géométriques en CONSTANT MEMORY ---
    // cudaMemcpyToSymbol : syntaxe différente de cudaMemcpy !
    // 1er arg = symbole déclaré __constant__ (pas un pointeur)
    cudaMemcpyToSymbol(c_ref_params, &ref_params,
                       sizeof(CudaCamParams));

    // Extraire les paramètres des caméras secondaires dans des tableaux plats
    CudaCamParams host_cam_params[MAX_CAMS];
    int           host_cam_dims[MAX_CAMS * 2];

    for (int i = 0; i < num_cams; i++) {
        host_cam_params[i]      = host_cams[i].p;
        host_cam_dims[i * 2]    = host_cams[i].width;
        host_cam_dims[i * 2 + 1]= host_cams[i].height;
    }

    cudaMemcpyToSymbol(c_cam_params, host_cam_params,
                       num_cams * sizeof(CudaCamParams));
    cudaMemcpyToSymbol(c_cam_dims,   host_cam_dims,
                       num_cams * 2 * sizeof(int));
    cudaMemcpyToSymbol(c_num_cams,   &num_cams,
                       sizeof(int));

    // --- Allocation image de référence ---
    uint8_t* dev_ref_pixels = nullptr;
    cudaMalloc(&dev_ref_pixels, ref_width * ref_height * sizeof(uint8_t));
    cudaMemcpy(dev_ref_pixels, host_ref_pixels,
               ref_width * ref_height * sizeof(uint8_t),
               cudaMemcpyHostToDevice);

    // --- Allocation pixels des caméras secondaires ---
    // On a besoin d'un tableau de pointeurs GPU accessible depuis le kernel
    uint8_t*  dev_cam_pixels[MAX_CAMS] = {};   // pointeurs GPU, un par caméra
    uint8_t** dev_pixel_ptrs = nullptr;        // tableau de ces pointeurs, sur le GPU

    for (int i = 0; i < num_cams; i++) {
        int sz = host_cams[i].width * host_cams[i].height * sizeof(uint8_t);
        cudaMalloc(&dev_cam_pixels[i], sz);
        cudaMemcpy(dev_cam_pixels[i], host_cams[i].dev_pixels, sz,
                   cudaMemcpyHostToDevice);
    }

    // Copie du tableau de pointeurs lui-même sur le GPU
    cudaMalloc(&dev_pixel_ptrs, num_cams * sizeof(uint8_t*));
    cudaMemcpy(dev_pixel_ptrs, dev_cam_pixels,
               num_cams * sizeof(uint8_t*),
               cudaMemcpyHostToDevice);

    // --- Allocation cube de coût ---
    float* dev_cost_cube = nullptr;
    int cube_size = z_planes * ref_height * ref_width * sizeof(float);
    cudaMalloc(&dev_cost_cube, cube_size);

    // --- Lancement kernel ---
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (ref_width  + blockSize.x - 1) / blockSize.x,
        (ref_height + blockSize.y - 1) / blockSize.y,
        z_planes
    );

    cudaEventRecord(start_gpu);
    kernel_sweeping_plane_shared_const<<<gridSize, blockSize>>>(
        dev_ref_pixels, ref_width, ref_height,
        dev_pixel_ptrs,
        dev_cost_cube, z_planes, z_near, z_far, window
    );
    cudaEventRecord(stop_gpu);
    cudaDeviceSynchronize();

    cudaMemcpy(host_cost_cube, dev_cost_cube, cube_size,
               cudaMemcpyDeviceToHost);

    float gpu_ms;
    cudaEventElapsedTime(&gpu_ms, start_gpu, stop_gpu);
    std::cout << "Kernel execution time : " << gpu_ms << " ms" << std::endl;

    // --- Nettoyage ---
    cudaFree(dev_ref_pixels);
    for (int i = 0; i < num_cams; i++)
        if (dev_cam_pixels[i]) cudaFree(dev_cam_pixels[i]);
    cudaFree(dev_pixel_ptrs);
    cudaFree(dev_cost_cube);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
}



