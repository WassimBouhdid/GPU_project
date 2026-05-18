#include "main.cuh"

#include <cstdio>
#include <cmath>

// Those functions are an example on how to call cuda functions from the main.cpp

__global__ void dev_test_vecAdd(int* A, int* B, int* C, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	C[i] = A[i] + B[i];
}

void wrap_test_vectorAdd() {
	printf("Vector Add:\n");

	int N = 3;
	int a[] = { 1, 2, 3 };
	int b[] = { 1, 2, 3 };
	int c[] = { 0, 0, 0 };

	int* dev_a, * dev_b, * dev_c;

	cudaMalloc((void**)&dev_a, N * sizeof(int));
	cudaMalloc((void**)&dev_b, N * sizeof(int));
	cudaMalloc((void**)&dev_c, N * sizeof(int));

	cudaMemcpy(dev_a, a, N * sizeof(int),
		cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N * sizeof(int),
		cudaMemcpyHostToDevice);

	dev_test_vecAdd <<<1, N>>> (dev_a, dev_b, dev_c, N);

	cudaMemcpy(c, dev_c, N * sizeof(int),
		cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	
	for (int i = 0; i < N; ++i) {
		printf("%i + %i = %i\n", a[i], b[i], c[i]);
	}
}

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
    double z = z_near * z_far / (z_near + (((double)zi / (double)z_planes) * (z_far - z_near)));

    // Passage du point 2D de l'image de réf en coordonnées 3D de la caméra de réf
    double X_ref = (ref_params.K_inv[0] * x + ref_params.K_inv[1] * y + ref_params.K_inv[2]) * z;
    double Y_ref = (ref_params.K_inv[3] * x + ref_params.K_inv[4] * y + ref_params.K_inv[5]) * z;
    double Z_ref = (ref_params.K_inv[6] * x + ref_params.K_inv[7] * y + ref_params.K_inv[8]) * z;

    // Passage des coordonnées caméra de réf aux coordonnées 3D du monde réel
    double X = ref_params.R_inv[0] * X_ref + ref_params.R_inv[1] * Y_ref + ref_params.R_inv[2] * Z_ref - ref_params.t_inv[0];
    double Y = ref_params.R_inv[3] * X_ref + ref_params.R_inv[4] * Y_ref + ref_params.R_inv[5] * Z_ref - ref_params.t_inv[1];
    double Z = ref_params.R_inv[6] * X_ref + ref_params.R_inv[7] * Y_ref + ref_params.R_inv[8] * Z_ref - ref_params.t_inv[2];

    float min_cost = 255.0f;

    // Boucle sur les autres caméras pour comparer les pixels
    for (int c = 0; c < num_cams; c++) {
        CudaCam cam = dev_cams[c];

        // Projection du point 3D du monde vers les coordonnées 3D de la caméra secondaire
        double X_proj = cam.p.R[0] * X + cam.p.R[1] * Y + cam.p.R[2] * Z - cam.p.t[0];
        double Y_proj = cam.p.R[3] * X + cam.p.R[4] * Y + cam.p.R[5] * Z - cam.p.t[1];
        double Z_proj = cam.p.R[6] * X + cam.p.R[7] * Y + cam.p.R[8] * Z - cam.p.t[2];

        // Passage de la 3D de la caméra secondaire vers ses coordonnées d'image 2D
        double x_proj = (cam.p.K[0] * X_proj / Z_proj + cam.p.K[1] * Y_proj / Z_proj + cam.p.K[2]);
        double y_proj = (cam.p.K[3] * X_proj / Z_proj + cam.p.K[4] * Y_proj / Z_proj + cam.p.K[5]);

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
    // 1. Allocation et copie de l'image de référence sur le GPU
    uint8_t* dev_ref_pixels;
    int ref_img_size = ref_width * ref_height * sizeof(uint8_t);
    cudaMalloc((void**)&dev_ref_pixels, ref_img_size);
    cudaMemcpy(dev_ref_pixels, host_ref_pixels, ref_img_size, cudaMemcpyHostToDevice);

    // 2. Allocation et copie des images des caméras secondaires sur le GPU
    CudaCam* local_cams = new CudaCam[num_cams];
    for (int i = 0; i < num_cams; i++) {
        local_cams[i] = host_cams[i];
        
        uint8_t* dev_cam_pixels;
        int cam_img_size = host_cams[i].width * host_cams[i].height * sizeof(uint8_t);
        cudaMalloc((void**)&dev_cam_pixels, cam_img_size);
        // host_cams[i].dev_pixels contient temporairement le pointeur CPU de l'image brute
        cudaMemcpy(dev_cam_pixels, host_cams[i].dev_pixels, cam_img_size, cudaMemcpyHostToDevice);
        
        local_cams[i].dev_pixels = dev_cam_pixels; // On remplace par le vrai pointeur GPU
    }

    // Copie du tableau de caméras lui-même sur le GPU
    CudaCam* dev_cams;
    cudaMalloc((void**)&dev_cams, num_cams * sizeof(CudaCam));
    cudaMemcpy(dev_cams, local_cams, num_cams * sizeof(CudaCam), cudaMemcpyHostToDevice);

    // 3. Allocation de l'espace pour stocker le cube de coût final sur le GPU
    float* dev_cost_cube;
    int cube_size = z_planes * ref_height * ref_width * sizeof(float);
    cudaMalloc((void**)&dev_cost_cube, cube_size);

    // 4. Configuration de la Grille de threads (Blos de 16x16 threads)
    dim3 blockSize(16, 16, 1);
    dim3 gridSize(
        (ref_width + blockSize.x - 1) / blockSize.x,
        (ref_height + blockSize.y - 1) / blockSize.y,
        z_planes // L'axe Z correspond directement au nombre de plans
    );

    // Lancement du Kernel magique !
    kernel_sweeping_plane<<<gridSize, blockSize>>>(
        ref_params, dev_ref_pixels, ref_width, ref_height,
        dev_cams, num_cams,
        dev_cost_cube, z_planes, z_near, z_far, window
    );

    // 5. Récupération du cube de coût calculé depuis le GPU vers la RAM du CPU
    cudaMemcpy(host_cost_cube, dev_cost_cube, cube_size, cudaMemcpyDeviceToHost);

    // 6. Nettoyage de la mémoire GPU (Très important pour éviter les fuites)
    cudaFree(dev_ref_pixels);
    for (int i = 0; i < num_cams; i++) {
        cudaFree(local_cams[i].dev_pixels);
    }
    cudaFree(dev_cams);
    cudaFree(dev_cost_cube);
    delete[] local_cams;
}

