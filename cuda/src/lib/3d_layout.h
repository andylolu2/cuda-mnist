
/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Defines layout functions used by TensorRef and derived classes.

    Layout functions map logical coordinates to linear memory. They often require additional
    data to describe strides between elements.

    Layout functions must implement all members in the public interface of IdentityTensorLayout<>
    defined in cutlass/tensor_ref.h.
*/
#pragma once

#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/pitch_linear_coord.h"

namespace cutlass {
    struct Tensor3D : public Coord<3, int> {
       public:
        /// Integer-valued index
        using Index = int;

        /// Base type is a Coord of rank=3
        using Base = Coord<3, Index>;

        /// LongIndex type
        using LongIndex = typename Base::LongIndex;

       private:
        /// Batch dimension
        static int const kBatch = 0;

        /// Rows dimension
        static int const kRow = 1;

        /// Columns dimension
        static int const kColumn = 2;

       public:
        //
        // Methods
        //

        /// Default ctor
        CUTLASS_HOST_DEVICE
        Tensor3D() {}

        /// Constructs from Coord<2>
        CUTLASS_HOST_DEVICE
        Tensor3D(Coord<3, Index> const& coord) : Base(coord) {}

        /// Helper to construct from a row and column
        CUTLASS_HOST_DEVICE
        Tensor3D(Index batch, Index row, Index column) : Base(make_Coord(batch, row, column)) {}

        /// Helper to construct from a row and column, which are LongIndex based
        CUTLASS_HOST_DEVICE
        Tensor3D(LongIndex batch, LongIndex row, LongIndex column)
            : Base(make_Coord((Index)batch, (Index)row, (Index)column)) {}

        // Returns the batch of the coordinate
        CUTLASS_HOST_DEVICE
        Index const& batch() const { return this->at(kBatch); }

        // Returns the batch of the coordinate
        CUTLASS_HOST_DEVICE
        Index& batch() { return this->at(kBatch); }

        /// Returns the row of the coordinate
        CUTLASS_HOST_DEVICE
        Index const& row() const { return this->at(kRow); }

        /// Returns the row of the coordinate
        CUTLASS_HOST_DEVICE
        Index& row() { return this->at(kRow); }

        /// Returns the column of the coordinate
        CUTLASS_HOST_DEVICE
        Index const& column() const { return this->at(kColumn); }

        /// Returns the column of the coordinate
        CUTLASS_HOST_DEVICE
        Index& column() { return this->at(kColumn); }

        //
        // Coord operators
        //

        /// Element-wise addition
        CUTLASS_HOST_DEVICE
        Tensor3D operator+(Base const& b) const { return Tensor3D(Base::operator+(b)); }

        /// Element-wise subtraction
        CUTLASS_HOST_DEVICE
        Tensor3D operator-(Base const& b) const { return Tensor3D(Base::operator-(b)); }

        /// Element-wise multiplication
        CUTLASS_HOST_DEVICE
        Tensor3D operator*(Base const& b) const { return Tensor3D(Base::operator*(b)); }

        /// Element-wise division
        CUTLASS_HOST_DEVICE
        Tensor3D operator/(Base const& b) const { return Tensor3D(Base::operator/(b)); }

        /// In-place addition
        CUTLASS_HOST_DEVICE
        Tensor3D& operator+=(Base const& b) {
            Base::operator+=(b);
            return *this;
        }

        /// In-place subtraction
        CUTLASS_HOST_DEVICE
        Tensor3D& operator-=(Base const& b) {
            Base::operator-=(b);
            return *this;
        }

        /// In-place multiplication
        CUTLASS_HOST_DEVICE
        Tensor3D& operator*=(Base const& b) {
            Base::operator*=(b);
            return *this;
        }

        /// In-place division
        CUTLASS_HOST_DEVICE
        Tensor3D& operator/=(Base const& b) {
            Base::operator/=(b);
            return *this;
        }
    };

    namespace layout {

        /////////////////////////////////////////////////////////////////////////////////////////////////
        //
        // Defines data layouts of various matrix formats usable by TensorRef and other classes.
        //
        /////////////////////////////////////////////////////////////////////////////////////////////////

        /// Mapping function for row-major matrices.
        class BatchedRowMajor {
           public:
            /// Logical rank of tensor
            static int const kRank = 3;

            /// Rank of stride vector
            static int const kStrideRank = 2;

            /// Index type used for coordinates
            using Index = int32_t;

            /// Long index type used for offsets
            using LongIndex = int64_t;

            /// Logical coordinate
            using TensorCoord = Tensor3D;

            /// Stride vector
            using Stride = Coord<kStrideRank>;

            using SubLayout = cutlass::layout::RowMajor;

           private:
            //
            // Data members
            //

            /// Stride data member
            Stride stride_;

           public:
            //
            // Methods
            //
            /// Constrcutor
            CUTLASS_HOST_DEVICE
            BatchedRowMajor() {}

            /// Constructor
            CUTLASS_HOST_DEVICE
            BatchedRowMajor(
                typename Stride::Index stride_r,  // number of elements between adjacent rows
                typename Stride::Index stride_b   // number of elements between adjacent batches
            )
                : stride_(make_Coord(stride_r, stride_b)) {}

            /// Ctor
            CUTLASS_HOST_DEVICE
            BatchedRowMajor(Stride stride) : stride_(stride) {}

            /// Helper returns a layout to a tightly packed tensor
            CUTLASS_HOST_DEVICE
            static BatchedRowMajor packed(Tensor3D const& extent) {
                return BatchedRowMajor(extent.column(), extent.row() * extent.column());
            }

            /// Returns the offset of a coordinate in linear memory.
            /// Assumes coordinate has convention (row, column)
            CUTLASS_HOST_DEVICE
            LongIndex operator()(Tensor3D const& coord) const {
                return coord.column() + LongIndex(coord.row() * stride_[0]) +
                       LongIndex(coord.batch() * stride_[1]);
            }

            /// Inverse of layout function, mapping linear offset to logical coordinate
            CUTLASS_HOST_DEVICE
            Tensor3D inverse(LongIndex offset) const {
                Index batch = Index(offset / stride_[1]);
                Index row = Index((offset % stride_[1]) / stride_[0]);
                Index column = Index(offset % stride_[0]);
                return Tensor3D(batch, row, column);
            }

            /// Returns the stride of the layout
            CUTLASS_HOST_DEVICE
            Stride stride() const { return stride_; }

            /// Returns the stride of the layout
            CUTLASS_HOST_DEVICE
            Stride& stride() { return stride_; }

            /// Returns the stride of the layout
            CUTLASS_HOST_DEVICE
            typename Stride::Index stride(int idx) const { return stride_[idx]; }

            /// Returns the stride of the layout
            CUTLASS_HOST_DEVICE
            typename Stride::Index& stride(int idx) { return stride_[idx]; }

            CUTLASS_HOST_DEVICE
            typename Stride::Index stride_row() const { return stride_[0]; }

            CUTLASS_HOST_DEVICE
            typename Stride::Index& stride_row() { return stride_[0]; }

            CUTLASS_HOST_DEVICE
            typename Stride::Index stride_batch() const { return stride_[1]; }

            CUTLASS_HOST_DEVICE
            typename Stride::Index& stride_batch() { return stride_[1]; }

            /// Compute the number of contiguous elements needed to store a tensor with the given
            /// size
            CUTLASS_HOST_DEVICE
            LongIndex capacity(Tensor3D const& extent) const {
                return LongIndex(extent.batch()) * LongIndex(extent.row()) *
                       LongIndex(extent.column());
            }
        };
    }  // namespace layout
}  // namespace cutlass
