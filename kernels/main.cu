#include "main.cuh"

#include <cstdio>
#include <cmath>

// ============================================================================
// 1. LE KERNEL (S'exécute en parallèle sur des milliers de cœurs du GPU)
// ============================================================================
__global__ void kernel_sweeping_plane(
    CudaCamParams ref_params, uint8_t* dev_ref_pixels, int ref_width, int ref_height,
    CudaCam* dev_cams, int num_cams,
    float* dev_cost_cube, int z_planes, float z_near, float z_far, int window
) {
    // Chaque thread trouve ses coordonnées uniques (x, y, zi)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z; // L'index du plan correspond à l'index Z du bloc CUDA

    // On vérifie qu'on ne déborde pas de l'image ou des plans
    if (x >= ref_width || y >= ref_height || zi >= z_planes) return;

    // --- CALCULS GÉOMÉTRIQUES INDÉPENDANTS ---
    // Calcul de la distance z pour ce plan précis
    float z = z_near * z_far / (z_near + (((float)zi / (float)z_planes) * (z_far - z_near)));

    // Passage du point 2D de l'image de réf en coordonnées 3D de la caméra de réf
    float X_ref = (ref_params.K_inv[0] * x + ref_params.K_inv[1] * y + ref_params.K_inv[2]) * z;
    float Y_ref = (ref_params.K_inv[3] * x + ref_params.K_inv[4] * y + ref_params.K_inv[5]) * z;
    float Z_ref = (ref_params.K_inv[6] * x + ref_params.K_inv[7] * y + ref_params.K_inv[8]) * z;

    // Passage des coordonnées caméra de réf aux coordonnées 3D du monde réel
    float X = ref_params.R_inv[0] * X_ref + ref_params.R_inv[1] * Y_ref + ref_params.R_inv[2] * Z_ref - ref_params.t_inv[0];
    float Y = ref_params.R_inv[3] * X_ref + ref_params.R_inv[4] * Y_ref + ref_params.R_inv[5] * Z_ref - ref_params.t_inv[1];
    float Z = ref_params.R_inv[6] * X_ref + ref_params.R_inv[7] * Y_ref + ref_params.R_inv[8] * Z_ref - ref_params.t_inv[2];

    float min_cost = 255.0f;

    // Boucle sur les autres caméras pour comparer les pixels
    for (int c = 0; c < num_cams; c++) {
        CudaCam cam = dev_cams[c];

        // Projection du point 3D du monde vers les coordonnées 3D de la caméra secondaire
        float X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
        float Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
        float Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

        // Passage de la 3D de la caméra secondaire vers ses coordonnées d'image 2D
        float x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
        float y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);

        // Sécurité pour rester dans l'image
        x_proj = (x_proj < 0 || x_proj >= cam.width) ? 0 : roundf(x_proj);
        y_proj = (y_proj < 0 || y_proj >= cam.height) ? 0 : roundf(y_proj);

        // --- CALCUL DU COÛT SUR LA FENÊTRE (WINDOW) ---
        float cost = 0.0f;
        float cc = 0.0f;

        for (int k = -window / 2; k <= window / 2; k++) {
            for (int l = -window / 2; l <= window / 2; l++) {
                if (x + l < 0 || x + l >= ref_width) continue;
                if (y + k < 0 || y + k >= ref_height) continue;
                if (x_proj + l < 0 || x_proj + l >= cam.width) continue;
                if (y_proj + k < 0 || y_proj + k >= cam.height) continue;

                // Indexation d'un tableau plat en 1D à partir de coordonnées 2D (y * largeur + x)
                int ref_pixel_idx = (y + k) * ref_width + (x + l);
                int cam_pixel_idx = ((int)y_proj + k) * cam.width + ((int)x_proj + l);

                // Différence absolue entre la luminosité des deux pixels
                cost += abs((int)dev_ref_pixels[ref_pixel_idx] - (int)cam.dev_pixels[cam_pixel_idx]);
                cc += 1.0f;
            }
        }

        if (cc > 0.0f) {
            cost /= cc;
            if (cost < min_cost) {
                min_cost = cost; // On garde le coût minimum parmi toutes les caméras
            }
        }
    }

    // Sauvegarde du résultat final dans le gros cube de coût global
    // Indexation d'un tableau 3D aplati en 1D
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

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaEvent_t start_gpu, stop_gpu; //cudaEvent are used to time the kernel
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // 1. Allocation et copie de l'image de référence sur le GPU
    uint8_t* dev_ref_pixels;
    int ref_img_size = ref_width * ref_height * sizeof(uint8_t);
    cudaMalloc((void**)&dev_ref_pixels, ref_img_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_ref_pixels, host_ref_pixels, ref_img_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // 2. Allocation et copie des images des caméras secondaires sur le GPU
    CudaCam* local_cams = new CudaCam[num_cams];
    for (int i = 0; i < num_cams; i++) {
        local_cams[i] = host_cams[i];
        
        uint8_t* dev_cam_pixels;
        int cam_img_size = host_cams[i].width * host_cams[i].height * sizeof(uint8_t);
        cudaMalloc((void**)&dev_cam_pixels, cam_img_size);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }
        // host_cams[i].dev_pixels contient temporairement le pointeur CPU de l'image brute
        cudaMemcpy(dev_cam_pixels, host_cams[i].dev_pixels, cam_img_size, cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!");
            goto Error;
        }

        local_cams[i].dev_pixels = dev_cam_pixels; // On remplace par le vrai pointeur GPU
    }

    // Copie du tableau de caméras lui-même sur le GPU
    CudaCam* dev_cams;
    cudaMalloc((void**)&dev_cams, num_cams * sizeof(CudaCam));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaMemcpy(dev_cams, local_cams, num_cams * sizeof(CudaCam), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
    // 3. Allocation de l'espace pour stocker le cube de coût final sur le GPU
    float* dev_cost_cube;
    int cube_size = z_planes * ref_height * ref_width * sizeof(float);
    cudaMalloc((void**)&dev_cost_cube, cube_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // 4. Configuration de la Grille de threads (Blos de 16x16 threads)
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        ((ref_width + blockSize.x - 1) / blockSize.x),
        ((ref_height + blockSize.y - 1) / blockSize.y),
        z_planes // L'axe Z correspond directement au nombre de plans
    );

    // Lancement du Kernel magique !

    cudaEventRecord(start_gpu);
    kernel_sweeping_plane<<<gridSize, blockSize>>>(
        ref_params, dev_ref_pixels, ref_width, ref_height,
        dev_cams, num_cams,
        dev_cost_cube, z_planes, z_near, z_far, window
    );
    cudaEventRecord(stop_gpu);
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "kernel_sweeping_plane launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_sweeping_plane!\n", cudaStatus);
        goto Error;
    }

    // 5. Récupération du cube de coût calculé depuis le GPU vers la RAM du CPU
    cudaMemcpy(host_cost_cube, dev_cost_cube, cube_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    float gpu_runtime_ms;
    cudaEventElapsedTime(&gpu_runtime_ms, start_gpu, stop_gpu);

    std::cout << "Kernel execution time : " << gpu_runtime_ms * 1000 << " us" << std::endl;
    printf("kernel execution time : %f \n", gpu_runtime_ms);


    if (start_gpu) cudaEventDestroy(start_gpu);
    if (stop_gpu) cudaEventDestroy(stop_gpu);

    // 6. Nettoyage de la mémoire GPU (Très important pour éviter les fuites)
    

Error:
    cudaFree(dev_ref_pixels);
    for (int i = 0; i < num_cams; i++) {
        cudaFree(local_cams[i].dev_pixels);
    }
    cudaFree(dev_cams);
    cudaFree(dev_cost_cube);
    delete[] local_cams;

}



