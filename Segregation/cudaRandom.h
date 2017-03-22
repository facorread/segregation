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

/** @file
* @brief CUDA-optimized algorithms to treat linear arrays as bidimensional grids. */

#ifndef CUDARANDOM_H
#define CUDARANDOM_H

#include <curand_kernel.h>
#include <limits>

namespace cudaSim {
/// Typedef for the strongly recommended random number generator, curandStatePhilox4_32_10_t.
typedef curandStatePhilox4_32_10_t rng;
/// Typedef for the random number type.
typedef unsigned int randVar;
/// Maximum value of the random number generator.
constexpr randVar rngMax{std::numeric_limits<randVar>::max()};

/** @brief Builds a discrete distribution from an array of probability weights.
@details Intended to be executed by an individual CUDA thread. This function does not perform any kind of memory management. You are responsible to allocate memory for all parameters before invocation.
@param [in] nElements Number of elements in the array of probability weights.
@param [in,out] weights Array of probability weights. Must reside in device memory, be finite and nonnegative. This function does not enforce any of these conditions and thus the behavior is not defined. This function will overwrite the weights to become the cumulative probabilities without normalization. If you do not want to lose your weights, better make a copy before invocation.
@param [out] distro Resulting distribution, normalized to rngMax.
*/
__device__ void buildDiscreteDistribution(const unsigned int nElements, float* const weights, randVar* const distro) {
	float norm{0};
	for(unsigned int i{0}; i < nElements; ++i) {
		norm += weights[i];
		weights[i] = norm;
	}
	norm = rngMax / norm;
	for(unsigned int i{0}; i < nElements - 1; ++i)
		distro[i] = weights[i] * norm;
	distro[nElements - 1] = rngMax; // Prevent infinite loops in discreteIndex() below.
}

__device__ unsigned int discreteIndex(const randVar* discreteDistro, rng * generator) {
	const randVar randvar{curand(generator)};
	unsigned int index{0};
	while(true) {
		if(randvar <= discreteDistro[index])
			return index;
		++index;
	}
	// An infinite loop is prevented by the last line in buildDiscreteDistribution() above.
}

/* The following code has been disabled because of a bug on CUDA Toolkit 8.0 reported on 2017/02/22.

/// Constant discrete distribution, optimized for CUDA thanks to template metaprogramming. Enter the distribution weights one by one.
template <unsigned int ... probabilityWeights> class constDiscreteDistro;

/// @cond
template<unsigned int oneWeight> class constDiscreteDistro<oneWeight> {
public:
	enum : size_t {
		norm = oneWeight,
		bucket = 0,
		bucketLimit = norm * static_cast<size_t>(rngMax)
	};
	__device__ static constexpr unsigned int privateIndex(const float x) {
		return 0;
	}
};

template<unsigned int weight, unsigned int ... weights> class constDiscreteDistro<weight, weights...> {
public:
	/// Rest of the weights.
	typedef constDiscreteDistro<weights...> c1;
	/// Accumulated norm of the weights.
	enum : size_t {
		norm = c1::norm + weight,
		bucket = c1::bucket + 1,
		bucketLimit = norm * static_cast<size_t>(rngMax)
	};
	/// Returns the index corresponding to a random number.
	__device__ static unsigned int index(const unsigned int randomNumber) {
		return privateIndex(randomNumber * norm);
	};
	__device__ static unsigned int privateIndex(const size_t x) {
		return (x < bucketLimit) ?
			c1::privateIndex(x) :
			bucket;
	};
};
/// @endcond
*/
} // namespace cudaSim

#endif CUDARANDOM_H
