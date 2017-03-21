/*
Segregation: Exercise on the conventional Schelling segregation model
Copyright (C) 2017 Fabio Correa <facorread@gmail.com>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <stdio.h>

namespace sim {
constexpr int nSigmaValues{100};

__global__ void kernel(float nNeighborsSigma[]) {
	const float sigma(blockIdx.x / static_cast<float>(nSigmaValues));
	if(!threadIdx.x)
		nNeighborsSigma[blockIdx.x] = sigma;
}

void main() {
	thrust::device_vector<float> nNeighborsSigma;
	nNeighborsSigma.resize(nSigmaValues, 0);
	kernel<<<nSigmaValues, 32>>>(thrust::raw_pointer_cast(nNeighborsSigma.data()));
	const thrust::host_vector<float> nNeighborsSigmaDevice{nNeighborsSigma};
	int sigmaI{0};
	for(const float nNeighSigma : nNeighborsSigmaDevice)
		std::cout << sigmaI++ << "\t" << nNeighSigma << '\n';
}
} // namespace sim

int main() {
		sim::main();
		// cudaDeviceReset must be called after every cudaFree, such as after the destruction
		// of all the thrust::device_vector objects, and right before exiting, for profiling
		// and tracing tools such as Nsight and Visual Profiler to show complete traces.
		if(cudaDeviceReset() != cudaSuccess) {
			std::cerr << "cudaDeviceReset failed. Please debug.\n";
			return 1;
		}
		return 0;
}
