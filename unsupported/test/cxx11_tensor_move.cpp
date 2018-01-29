// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2017 Viktor Csomor <viktor.csomor@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>
#include <utility>

using Eigen::Tensor;
using Eigen::RowMajor;

void test_cxx11_tensor_move()
{
  Tensor<int,3> tensor1(2, 2, 2);
  Tensor<int,3,RowMajor> tensor2(2, 2, 2);

  for (int i = 0; i < 8; i++)
  {
    int x = i / 4;
    int y = (i % 4) / 2;
    int z = i % 2;
    tensor1(x,y,z) = i;
    tensor2(x,y,z) = 2 * i;
  }

  // Invokes the move constructor.
  Tensor<int,3> moved_tensor1 = std::move(tensor1);
  Tensor<int,3,RowMajor> moved_tensor2 = std::move(tensor2);

  VERIFY_IS_EQUAL(tensor1.size(), 0);
  VERIFY_IS_EQUAL(tensor2.size(), 0);

  VERIFY_IS_EQUAL(moved_tensor1(0,0), 0);
  VERIFY_IS_EQUAL(moved_tensor1(0,1), 1);
  VERIFY_IS_EQUAL(moved_tensor1(1,0), 2);
  VERIFY_IS_EQUAL(moved_tensor1(1,1), 3);
  VERIFY_IS_EQUAL(moved_tensor1(2,0), 4);
  VERIFY_IS_EQUAL(moved_tensor1(2,1), 5);
  VERIFY_IS_EQUAL(moved_tensor1(3,0), 6);
  VERIFY_IS_EQUAL(moved_tensor1(3,1), 7);

  VERIFY_IS_EQUAL(moved_tensor2(0,0), 0);
  VERIFY_IS_EQUAL(moved_tensor2(0,1), 2);
  VERIFY_IS_EQUAL(moved_tensor2(1,0), 4);
  VERIFY_IS_EQUAL(moved_tensor2(1,1), 6);
  VERIFY_IS_EQUAL(moved_tensor2(2,0), 8);
  VERIFY_IS_EQUAL(moved_tensor2(2,1), 10);
  VERIFY_IS_EQUAL(moved_tensor2(3,0), 12);
  VERIFY_IS_EQUAL(moved_tensor2(3,1), 14);

  Tensor<int,3> moved_tensor3(2,2,2);
  Tensor<int,3,RowMajor> moved_tensor4(2,2,2);

  moved_tensor3.setZero();
  moved_tensor4.setZero();

  // Invokes the move assignment operator.
  moved_tensor3 = std::move(moved_tensor1);
  moved_tensor4 = std::move(moved_tensor2);

  VERIFY_IS_EQUAL(moved_tensor1.size(), 8);
  VERIFY_IS_EQUAL(moved_tensor2.size(), 8);

  for (int i = 0; i < 8; i++)
  {
    int x = i / 4;
    int y = (i % 4) / 2;
    int z = i % 2;
    VERIFY_IS_EQUAL(moved_tensor1(x,y,z), 0);
    VERIFY_IS_EQUAL(moved_tensor2(x,y,z), 0);
    VERIFY_IS_EQUAL(moved_tensor3(x,y,z), i);
    VERIFY_IS_EQUAL(moved_tensor4(x,y,z), 2 * i);
  }
}
