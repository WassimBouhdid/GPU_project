#include "main.cuh"

#include <cstdio>
#include <cmath>

#define BLOCK_W 16
#define BLOCK_H 16

// ============================================================================
// 1. LE KERNEL (S'exécute en parallèle sur des milliers de cœurs du GPU)
// ============================================================================
__global__ void kernel_sweeping_plane_shared(
    CudaCamParams ref_params, uint8_t* dev_ref_pixels, int ref_width, int ref_height,
    CudaCam* dev_cams, int num_cams,
    float* dev_cost_cube, int z_planes, float z_near, float z_far, int window
) {
    // Chaque thread trouve ses coordonnées uniques (x, y, zi)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z; // L'index du plan correspond à l'index Z du bloc CUDA

    int half_w = window / 2;

    // --- Shared memory pour le tile de la référence ---
    // Taille : (BLOCK_W + window - 1) x (BLOCK_H + window - 1)
    // On utilise une taille statique max (window <= 15 -> tile <= 30x30)
    #define TILE_W (BLOCK_W + 14)   // 14 = window_max - 1
    #define TILE_H (BLOCK_H + 14)
    __shared__ uint8_t tile_ref[TILE_H][TILE_W];

    // --- Origine du tile dans l'image globale ---
    int tile_origin_x = blockIdx.x * BLOCK_W - half_w;
    int tile_origin_y = blockIdx.y * BLOCK_H - half_w;

    // Taille réelle du tile pour ce window
    int tile_w = BLOCK_W + window - 1;
    int tile_h = BLOCK_H + window - 1;

    // --- Chargement coopératif du tile de la référence ---
    // Chaque thread charge 1 ou plusieurs pixels du tile
    int tid = threadIdx.y * BLOCK_W + threadIdx.x;
    int num_threads = BLOCK_W * BLOCK_H;
    int num_pixels = tile_w * tile_h;

    for (int i = tid; i < num_pixels; i += num_threads) {
        int ty = i / tile_w;
        int tx = i % tile_w;
        int gx = tile_origin_x + tx;
        int gy = tile_origin_y + ty;

        // Gestion des bords (padding à 0)
        if (gx < 0 || gx >= ref_width || gy < 0 || gy >= ref_height)
            tile_ref[ty][tx] = 0;
        else
            tile_ref[ty][tx] = dev_ref_pixels[gy * ref_width + gx];
    }

    // Synchronisation : tout le bloc attend que le tile soit chargé
    __syncthreads();

    // --- Vérification des bornes (après le chargement coopératif) ---
    if (x >= ref_width || y >= ref_height || zi >= z_planes) return;

    // --- Calculs géométriques (identiques à avant) ---
    float z = z_near * z_far / (z_near + (((float)zi / (float)z_planes) * (z_far - z_near)));

    float X_ref = (ref_params.K_inv[0] * x + ref_params.K_inv[1] * y + ref_params.K_inv[2]) * z;
    float Y_ref = (ref_params.K_inv[3] * x + ref_params.K_inv[4] * y + ref_params.K_inv[5]) * z;
    float Z_ref = (ref_params.K_inv[6] * x + ref_params.K_inv[7] * y + ref_params.K_inv[8]) * z;

    float X = ref_params.R_inv[0]*X_ref + ref_params.R_inv[1]*Y_ref + ref_params.R_inv[2]*Z_ref - ref_params.t_inv[0];
    float Y = ref_params.R_inv[3]*X_ref + ref_params.R_inv[4]*Y_ref + ref_params.R_inv[5]*Z_ref - ref_params.t_inv[1];
    float Z = ref_params.R_inv[6]*X_ref + ref_params.R_inv[7]*Y_ref + ref_params.R_inv[8]*Z_ref - ref_params.t_inv[2];

    float min_cost = 255.0f;

    for (int c = 0; c < num_cams; c++) {
        CudaCam cam = dev_cams[c];

        float X_proj = cam.p.R[0]*X + cam.p.R[1]*Y + cam.p.R[2]*Z - cam.p.t[0];
        float Y_proj = cam.p.R[3]*X + cam.p.R[4]*Y + cam.p.R[5]*Z - cam.p.t[1];
        float Z_proj = cam.p.R[6]*X + cam.p.R[7]*Y + cam.p.R[8]*Z - cam.p.t[2];

        float x_proj = cam.p.K[0]*X_proj/Z_proj + cam.p.K[1]*Y_proj/Z_proj + cam.p.K[2];
        float y_proj = cam.p.K[3]*X_proj/Z_proj + cam.p.K[4]*Y_proj/Z_proj + cam.p.K[5];

        x_proj = (x_proj < 0 || x_proj >= cam.width)  ? 0 : roundf(x_proj);
        y_proj = (y_proj < 0 || y_proj >= cam.height) ? 0 : roundf(y_proj);

        // --- Boucle fenêtre : ref depuis SHARED, cam depuis global ---
        float cost = 0.0f;
        float cc = 0.0f;

        for (int k = -half_w; k <= half_w; k++) {
            for (int l = -half_w; l <= half_w; l++) {
                // Index dans le tile shared (toujours valide grâce au chargement coopératif)
                int tile_x = threadIdx.x + half_w + l;  // décalage depuis l'origine du bloc
                int tile_y = threadIdx.y + half_w + k;

                // Pixel référence : depuis shared memory (rapide !)
                uint8_t ref_pixel = tile_ref[tile_y][tile_x];

                // Pixel caméra secondaire : vérification + lecture mémoire globale
                int cx = (int)x_proj + l;
                int cy = (int)y_proj + k;
                if (cx < 0 || cx >= cam.width || cy < 0 || cy >= cam.height) continue;

                uint8_t cam_pixel = cam.dev_pixels[cy * cam.width + cx];

                cost += abs((int)ref_pixel - (int)cam_pixel);
                cc += 1.0f;
            }
        }

        if (cc > 0.0f) {
            cost /= cc;
            if (cost < min_cost) min_cost = cost;
        }
    }

    int cube_idx = zi * (ref_height * ref_width) + y * ref_width + x;
    dev_cost_cube[cube_idx] = min_cost;
}

// ============================================================================
// 2. LE WRAPPER CPU (Gère la mémoire et orchestre le lancement du Kernel)
// ============================================================================
void wrap_sweeping_plane_cuda(
    CudaCamParams ref_params, uint8_t* host_ref_pixels, int ref_width, int ref_height,
    CudaCam* host_cams, int num_cams,
    float* host_cost_cube, int z_planes, float z_near, float z_far, int window
) {
    // --- 1. DÉCLARATION DE TOUTES LES VARIABLES AU DÉBUT ---
    cudaError_t cudaStatus;
    cudaEvent_t start_gpu, stop_gpu;
    
    // Variables de taille et pointeurs
    int ref_img_size = ref_width * ref_height * sizeof(uint8_t);
    int cube_size = z_planes * ref_height * ref_width * sizeof(float);
    
    uint8_t* dev_ref_pixels = nullptr;
    CudaCam* local_cams = new CudaCam[num_cams];
    CudaCam* dev_cams = nullptr;
    float* dev_cost_cube = nullptr;

    // Configuration des threads
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        ((ref_width + blockSize.x - 1) / blockSize.x),
        ((ref_height + blockSize.y - 1) / blockSize.y),
        z_planes
    );

    // --- 2. LOGIQUE D'EXÉCUTION ---
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) goto Error;

    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Allocation et copie de l'image de référence
    cudaMalloc((void**)&dev_ref_pixels, ref_img_size);
    cudaMemcpy(dev_ref_pixels, host_ref_pixels, ref_img_size, cudaMemcpyHostToDevice);

    // Allocation des caméras secondaires
    for (int i = 0; i < num_cams; i++) {
        local_cams[i] = host_cams[i];
        int cam_img_size = host_cams[i].width * host_cams[i].height * sizeof(uint8_t);
        cudaMalloc((void**)&local_cams[i].dev_pixels, cam_img_size);
        cudaMemcpy(local_cams[i].dev_pixels, host_cams[i].dev_pixels, cam_img_size, cudaMemcpyHostToDevice);
    }

    cudaMalloc((void**)&dev_cams, num_cams * sizeof(CudaCam));
    cudaMemcpy(dev_cams, local_cams, num_cams * sizeof(CudaCam), cudaMemcpyHostToDevice);

    // Allocation cube de coût
    cudaMalloc((void**)&dev_cost_cube, cube_size);

    // Lancement du Kernel
    cudaEventRecord(start_gpu);
    kernel_sweeping_plane_shared<<<gridSize, blockSize>>>(
        ref_params, dev_ref_pixels, ref_width, ref_height,
        dev_cams, num_cams,
        dev_cost_cube, z_planes, z_near, z_far, window
    );
    cudaEventRecord(stop_gpu);
    
    cudaDeviceSynchronize();

    // Récupération résultat
    cudaMemcpy(host_cost_cube, dev_cost_cube, cube_size, cudaMemcpyDeviceToHost);

    // Calcul temps
    float gpu_runtime_ms;
    cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, stop_gpu);
    std::cout << "Kernel execution time : " << gpu_runtime_ms << " ms" << std::endl;

    // --- 3. NETTOYAGE ET FIN ---
Error:
    if (dev_ref_pixels) cudaFree(dev_ref_pixels);
    if (dev_cams) {
        for (int i = 0; i < num_cams; i++) {
            if (local_cams[i].dev_pixels) cudaFree(local_cams[i].dev_pixels);
        }
        cudaFree(dev_cams);
    }
    if (dev_cost_cube) cudaFree(dev_cost_cube);
    
    delete[] local_cams;
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
}



