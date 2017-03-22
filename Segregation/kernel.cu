/* This file is part of SchellingSegregation: Exercise on the conventional Schelling segregation model.
	
SchellingSegregation is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SchellingSegregation is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with SchellingSegregation.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <algorithm>
#include <cuda_runtime.h>
#include "cudaGrid.h"
#include "cudaRandom.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace cudaSim {
typedef unsigned int resultsCls;
constexpr int nSigmaValues{100};
constexpr unsigned int nTestBuckets{3};
constexpr int threadsPerBlock{256};
typedef dimensions2Cls<dimCls<unsigned char, 64>, dimCls<unsigned char, 64>> gridT;

/// Represents the status of a cell.
enum class cellStatusEnum : char {
	empty,
	color1,	// Inhabited by a color1 person.
	color2	// Inhabited by a color2 person.
};

/// Represents a cell in space.
struct __align__(4) cellCls {
	cellStatusEnum status;
	char nAffineNeighbors;
};

/// Represents variables and components to the simulation that reside in shared memory.
struct worldCls {
	float sigma;
	curandStatePhilox4_32_10_t randStates[threadsPerBlock];
	cellCls states[gridT::size()];
	randVar testDiscreteDistro[nTestBuckets];
	unsigned int testDiscreteDistroCounter[nTestBuckets];
	unsigned int totalTests;
};

__device__ bool isCloseEnough(const unsigned int a, const unsigned int b) {
	constexpr unsigned int tolerance{10000};
	if(a > b)
		return a < b + tolerance;
	return b < a + tolerance;
}

__global__ void kernel(resultsCls nNeighborsSigma[]) {
	__shared__ worldCls world;
	const unsigned long long thx2{blockIdx.x * threadIdx.x + blockIdx.x + threadIdx.x + 6};
	curand_init(5 * thx2, 2 * thx2, 3 * thx2, &world.randStates[threadIdx.x]);
	if(!threadIdx.x) {
		world.sigma = blockIdx.x / static_cast<float>(nSigmaValues);
		float testDistributionWeights[nTestBuckets]{20, 50, 30};
		buildDiscreteDistribution(nTestBuckets, testDistributionWeights, world.testDiscreteDistro);
		for(unsigned int i{0}; i < nTestBuckets; ++i)
			world.testDiscreteDistroCounter[i] = 0;
		world.totalTests = 0;
		nNeighborsSigma[blockIdx.x] = 0;
	}
	__syncthreads();
	for(unsigned int i{threadIdx.x}; i < gridT::size(); i += threadsPerBlock) {
		world.states[i].nAffineNeighbors = threadIdx.x;
		for(unsigned int j{0}; j < 0x0400; ++j) {
			const unsigned int index{discreteIndex(world.testDiscreteDistro, &world.randStates[threadIdx.x])};
			if(!(index < nTestBuckets)) // Checks out OK
				atomicInc(&nNeighborsSigma[blockIdx.x], rngMax);
			atomicInc(&world.testDiscreteDistroCounter[index], rngMax);
			atomicInc(&world.totalTests, rngMax);
		}
		// curand(&world.randStates[threadIdx.x])
	}
	/*
	atomicExch(&nNeighborsSigma[blockIdx.x], totalTests);
	if(!isCloseEnough(totalTests * 0.2f, testDiscreteDistroCounter[0])) {
		atomicInc(&nNeighborsSigma[blockIdx.x], rngMax);
	}
	*/
	/*
	if(!isCloseEnough(totalTests * 0.8f, testDiscreteDistroCounter[1]))
		atomicInc(&nNeighborsSigma[blockIdx.x], rngMax);
	if(!isCloseEnough(totalTests * 0.1f, testDiscreteDistroCounter[2]))
		atomicInc(&nNeighborsSigma[blockIdx.x], rngMax);
	*/
	__syncthreads();
//	if(!threadIdx.x)
//		nNeighborsSigma[blockIdx.x] = testDiscreteDistroCounter[0];
}
} // namespace cudaSim

namespace sim {
void main() {
	thrust::device_vector<cudaSim::resultsCls> nNeighborsSigma;
	nNeighborsSigma.resize(cudaSim::nSigmaValues, 0);
	cudaSim::kernel<<<cudaSim::nSigmaValues, cudaSim::threadsPerBlock >>>(thrust::raw_pointer_cast(nNeighborsSigma.data()));
	const thrust::host_vector<cudaSim::resultsCls> nNeighborsSigmaDevice{nNeighborsSigma};
	int sigmaI{0};
	for(const cudaSim::resultsCls& nNeighSigma : nNeighborsSigmaDevice)
		std::cout << sigmaI++ / static_cast<float>(cudaSim::nSigmaValues) << "\t" << nNeighSigma << '\n';
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
