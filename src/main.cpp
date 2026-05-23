#include "../kernels/main.cuh"
#include "cam_params.hpp"
#include "constants.hpp"
#include "graph.h"

#include <cstdio>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <string>

#define SHRT_MAX 32767

std::vector<cam> read_cams(std::string const &folder)
{
	// Init parameters
	std::vector<params<double>> cam_params_vector = get_cam_params();

	// Init cameras
	std::vector<cam> cam_array(cam_params_vector.size());
	for (int i = 0; i < cam_params_vector.size(); i++)
	{
		// Name
		std::string name = folder + "/v" + std::to_string(i) + ".png";

		// Read PNG file
		cv::Mat im_rgb = cv::imread(name);
		cv::Mat im_yuv;
		const int width = im_rgb.cols;
		const int height = im_rgb.rows;

		// Convert to YUV420
		cv::cvtColor(im_rgb, im_yuv, cv::COLOR_BGR2YUV_I420);
		const int size = width * height * 1.5; // YUV 420

		std::vector<cv::Mat> YUV;
		cv::split(im_rgb, YUV);

		// Params
		cam_array.at(i) = cam(name, width, height, size, YUV, cam_params_vector.at(i));
	}

	return cam_array;

	// cv::Mat U(height / 2, width / 2, CV_8UC1, cam_array.at(0).image.data() + (int)(width * height * 1.25));
	// cv::namedWindow("im", cv::WINDOW_NORMAL);
	// cv::imshow("im", U);
	// cv::waitKey(0);
}

std::vector<cv::Mat> sweeping_plane(cam const ref, std::vector<cam> const &cam_vector, int window = 3)
{
	// 1. Préparer le cube de coût de retour pour la suite du programme (Graph Cut)
	std::vector<cv::Mat> cost_cube(ZPlanes);
	for (int i = 0; i < cost_cube.size(); ++i)
	{
		cost_cube[i] = cv::Mat(ref.height, ref.width, CV_32FC1, 255.);
	}

	// 2. Convertir les paramètres géométriques de la caméra de référence pour le GPU
	CudaCamParams ref_params;
	for (int i = 0; i < 9; i++) {
		ref_params.K[i] = ref.p.K[i];
		ref_params.R[i] = ref.p.R[i];
		ref_params.K_inv[i] = ref.p.K_inv[i];
		ref_params.R_inv[i] = ref.p.R_inv[i];
	}
	for (int i = 0; i < 3; i++) {
		ref_params.t[i] = ref.p.t[i];
		ref_params.t_inv[i] = ref.p.t_inv[i];
	}

	// Pointeur vers les pixels bruts en niveaux de gris de la caméra de référence
	uint8_t* host_ref_pixels = ref.YUV[0].data;

	// 3. Préparer le tableau des caméras secondaires (en excluant la caméra de référence)
	int num_cams = cam_vector.size() - 1;
	std::vector<CudaCam> host_cams(num_cams);
	
	int cam_idx = 0;
	for (auto const &cam : cam_vector)
	{
		if (cam.name == ref.name)
			continue;

		host_cams[cam_idx].width = cam.width;
		host_cams[cam_idx].height = cam.height;
		host_cams[cam_idx].dev_pixels = cam.YUV[0].data; // On passe le pointeur CPU temporairement

		for (int i = 0; i < 9; i++) {
			host_cams[cam_idx].p.K[i] = cam.p.K[i];
			host_cams[cam_idx].p.R[i] = cam.p.R[i];
			host_cams[cam_idx].p.K_inv[i] = cam.p.K_inv[i];
			host_cams[cam_idx].p.R_inv[i] = cam.p.R_inv[i];
		}
		for (int i = 0; i < 3; i++) {
			host_cams[cam_idx].p.t[i] = cam.p.t[i];
			host_cams[cam_idx].p.t_inv[i] = cam.p.t_inv[i];
		}

		cam_idx++;
	}

	// 4. Allouer un tableau plat pour récupérer le cube de coût global depuis le GPU
	int flat_cube_size = ZPlanes * ref.height * ref.width;
	std::vector<float> host_flat_cost_cube(flat_cube_size, 255.0f);

	// 5. Appeler notre Wrapper CUDA !
	std::cout << "--- Lancement du Plane Sweeping sur le GPU (CUDA) ---" << std::endl;
	wrap_sweeping_plane_cuda(
		ref_params, host_ref_pixels, ref.width, ref.height,
		host_cams.data(), num_cams,
		host_flat_cost_cube.data(), ZPlanes, ZNear, ZFar, window
	);
	std::cout << "--- Calcul GPU terminé avec succès ! ---" << std::endl;

	// 6. Transférer le tableau plat 1D reçu du GPU dans le format std::vector<cv::Mat> 2D
	for (int zi = 0; zi < ZPlanes; zi++)
	{
		for (int y = 0; y < ref.height; y++)
		{
			for (int x = 0; x < ref.width; x++)
			{
				int idx = zi * (ref.height * ref.width) + y * ref.width + x;
				cost_cube[zi].at<float>(y, x) = host_flat_cost_cube[idx];
			}
		}
	}

	return cost_cube;
}

cv::Mat find_min(std::vector<cv::Mat> const &cost_cube)
{
	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	cv::Mat ret(height, width, CV_32FC1, 255.);
	cv::Mat depth(height, width, CV_8U, 255);

	for (int zi = 0; zi < zPlanes; zi++)
	{
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				if (cost_cube[zi].at<float>(y, x) < ret.at<float>(y, x))
				{
					ret.at<float>(y, x) = cost_cube[zi].at<float>(y, x);
					depth.at<u_char>(y, x) = zi;
				}
			}
		}
	}

	return depth;
}

/*The next two function are used to perform the graph cut on the results
DO NOT MODIFY THOSE FUNCTIONS - DO NOT TRY TO IMPLEMENT THEM ON THE GPU*/
void depth_estimation_by_graph_cut_sWeight_add_nodes(Graph& g, std::vector<Graph::node_id>& nodes, cv::Size destPixel, cv::Size sourcePixel, cv::Size imgSize, std::vector<double> m_aiEdgeCost, cv::Mat1w labels, int label, double cost_cur) {
	const int idxSourcePixel = sourcePixel.height * imgSize.width + sourcePixel.width;
	const int idxDestPixel = destPixel.height * imgSize.width + destPixel.width;
	const double cost_cur_temp = cost_cur;

	if (labels(sourcePixel.height, sourcePixel.width) != labels(destPixel.height, destPixel.width)) {
		//add a new node and add edge between it and the adjacent nodes
		Graph::node_id tmp_node = g.add_node();
		const double cost_temp = m_aiEdgeCost[std::abs(labels(destPixel.height, destPixel.width) - label)];
		g.set_tweights(tmp_node, 0, m_aiEdgeCost[std::abs(labels(sourcePixel.height, sourcePixel.width) - labels(destPixel.height, destPixel.width))]);
		g.add_edge(nodes[idxSourcePixel], tmp_node, cost_cur_temp, cost_cur_temp);
		g.add_edge(tmp_node, nodes[idxDestPixel], cost_temp, cost_temp);
	}
	else //only add an edge between two nodes
		g.add_edge(nodes[idxSourcePixel], nodes[idxDestPixel], cost_cur_temp, cost_cur_temp);
}

cv::Mat depth_estimation_by_graph_cut_sWeight(std::vector<cv::Mat> const& cost_cube) {
	//DO NOT TRY TO IMPLEMENT THIS FUNCTION ON THE GPU

	const int zPlanes = cost_cube.size();
	const int height = cost_cube[0].size().height;
	const int width = cost_cube[0].size().width;

	//To store the depth values assigned to each pixels, start with 0
	cv::Mat1w labels = cv::Mat::zeros(height, width, CV_16U); 
	//store the cost for a label
	std::vector<double> m_aiEdgeCost;
	double smoothing_lambda = 1.0;
	m_aiEdgeCost.resize(zPlanes);
	for (int i = 0; i < zPlanes; ++i)
		m_aiEdgeCost[i] = smoothing_lambda * i;

	for (int source = 0; source < zPlanes; ++source) {
		printf("depth layer %i \n", source);
		Graph g;
		std::vector<Graph::node_id> nodes(height * width, nullptr);

		//Putting the weights for the connection to the source and the sink for each nodes
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				//indice global du pixel
				const int pp = r * width + c;
				nodes[pp] = g.add_node();
				const ushort label = labels(r, c);
				if (label == source)
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), SHRT_MAX);
				else
					g.set_tweights(nodes[pp], cost_cube[source].at<float>(r, c), cost_cube[label].at<float>(r, c));
			}
		}

		
		for (int j = 0; j < height; j++) {
			for (int i = 0; i < width; i++) {
				const double cost_curr = m_aiEdgeCost[std::abs(labels(j, i) - source)];

				//create an edge between the adjacent nodes, may add an additional node on this edge if the previously calculated labels are different
				if (i != width - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i + 1, j), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
				if (j != height - 1) {
					depth_estimation_by_graph_cut_sWeight_add_nodes(g, nodes, cv::Size(i, j + 1), cv::Size(i, j), cv::Size(width, height), m_aiEdgeCost, labels, source, cost_curr);
				}
			}
		}
		//printf("nodes and egde set \n");

		//resolve the maximum flow/minimum cut problem
		g.maxflow();

		//update the depth labels, nodes that are still connected to the source will receive a new depth label
		for (int r = 0; r < height; ++r) {
			for (int c = 0; c < width; ++c) {
				const int pp = r * width + c;
				if (g.what_segment(nodes[pp]) != Graph::SOURCE)
					labels(r, c) = ushort(source);
			}
		}
		nodes.clear();
		
		/*
		cv::namedWindow("labels", cv::WINDOW_NORMAL);
		cv::imshow("labels", labels);
		cv::waitKey(0);
		*/

	}

	cv::Mat depth;
	labels.convertTo(depth, CV_8U, 1.0);

	return depth;
}

int main()
{
	// Read cams
	std::vector<cam> cam_vector = read_cams("data");

	// Sweeping algorithm for camera 0
	std::vector<cv::Mat> cost_cube = sweeping_plane(cam_vector.at(0), cam_vector, 5);

	// Use graph cut to generate depth map 
	// Cleaner results, long compute time
	// cv::Mat depth = depth_estimation_by_graph_cut_sWeight(cost_cube);

	// Find min cost and generate depth map
	// Faster result, low quality
	cv::Mat depth = find_min(cost_cube);


	cv::namedWindow("Depth", cv::WINDOW_NORMAL);
	cv::imshow("Depth", depth);
	cv::waitKey(0);

	cv::imwrite("./depth_map.png", depth);

	//printf("%f", depth.at<uchar>(0, 0));

	return 0;
}