#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <iostream>

#define MAX_CAMS 8

// Structure à taille fixe pour les paramètres géométriques d'une caméra
struct CudaCamParams {
    float K[9];
    float R[9];
    float t[3];
    float K_inv[9];
    float R_inv[9];
    float t_inv[3];
};

// Structure pour regrouper une caméra secondaire sur le GPU
struct CudaCam {
    int width;
    int height;
    uint8_t* dev_pixels; // Pointeur vers ses pixels sur la mémoire GPU
    CudaCamParams p;
};

// Notre fonction principale qui sera appelée depuis le main.cpp
void wrap_sweeping_plane_cuda(
    CudaCamParams ref_params, uint8_t* host_ref_pixels, int ref_width, int ref_height,
    CudaCam* host_cams, int num_cams,
    float* host_cost_cube, int z_planes, float z_near, float z_far, int window
);