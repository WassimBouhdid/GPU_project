#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

// Structure à taille fixe pour les paramètres géométriques d'une caméra
struct CudaCamParams {
    double K[9];
    double R[9];
    double t[3];
    double K_inv[9];
    double R_inv[9];
    double t_inv[3];
};

// Structure pour regrouper une caméra secondaire sur le GPU
struct CudaCam {
    int width;
    int height;
    uint8_t* dev_pixels; // Pointeur vers ses pixels sur la mémoire GPU
    CudaCamParams p;
};

// L'ancienne fonction de test (on la laisse pour l'instant)
// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();

// Notre fonction principale qui sera appelée depuis le main.cpp
void wrap_sweeping_plane_cuda(
    CudaCamParams ref_params, uint8_t* host_ref_pixels, int ref_width, int ref_height,
    CudaCam* host_cams, int num_cams,
    float* host_cost_cube, int z_planes, float z_near, float z_far, int window
);