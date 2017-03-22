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

#ifndef CUDAGRID_H
#define CUDAGRID_H

namespace cudaSim {
/// Defines the properties of a dimension in dimension2Cls. @tparam sizeT Index and size type. Recommended types for CUDA are `unsigned char` (8 bits), `unsigned short` (16 bits) or `unsigned int` (32 bits). Do not use `decltype(desiredDimensionSize)` as it will produce an unexpected type; for example `sizeof(decltype(10)) = 4` but you probably wanted `1`. @tparam mSize Size of the world in this dimension. @tparam mWrap Whether indices wrap around in this dimension.
template<class sizeT, const sizeT mSize, const bool mWrap = true> class dimCls {
	public:
		/// Typedef for the index type.
		typedef sizeT dimSizeT;
		/// Returns the total number of objects in the given dimension (width, height).
		__host__ __device__ static constexpr dimSizeT size() { return mSize; }
		/// Returns the maximum possible index.
		__host__ __device__ static constexpr dimSizeT maxIndex() { return mSize - 1; }
		/// Returns whether the index wraps around for the given dimension.
		__host__ __device__ static constexpr bool wrap() { return mWrap; }
		/// Curates the given index.
		__host__ __device__ static void curate(dimSizeT& index) {
			if(wrap())
				index = index % size();
		};
};

/// Represents the position of an element in a 2D matrix.
template<class dimX, class dimY> struct index2Cls {
	dimX::dimSizeT xIndex;
	dimY::dimSizeT yIndex;
};

/// CUDA-optimized algorithms to treat linear arrays as bidimensional grids. For example, use `typedef dimensions2Cls<dimCls<30>, dimCls<20>> gridT` to create and index linear arrays to represent grids of 30X20 cells. @tparam dimX Properties of the grid in the X dimension. @tparam dimY Properties of the grid in the Y dimension.
template<class dimX, class dimY> class dimensions2Cls {
public:
	/// Typedef to represent the properties in the X dimension.
	typedef dimX xCls;
	/// Typedef to represent the properties in the Y dimension.
	typedef dimY yCls;
	/// Returns the required size of the container. It returns auto which can be as low as 32 bit, enough to conserve memory bandwidth on CUDA.
	__host__ __device__ static constexpr auto size() { return dimX::size() * dimY::size(); };
	/// Curates the given 2D indices.
	__host__ __device__ static void curate(index2Cls<dimX, dimY>& idx2) {
		dimX::curate(idx2.xIndex);
		dimY::curate(idx2.yIndex);
	}
	/// Returns the linear index corresponding to the given 2D indices. It returns auto which can be as low as 32 bit, enough to conserve memory bandwidth on CUDA.
	__host__ __device__ static auto index(index2Cls<dimX, dimY> idx2) {
		curate(idx2);
		return dimX::size() * idx2.yIndex + idx2.xIndex;
	}
};
} // namespace cudaSim

#endif CUDAGRID_H
