/*****************************************************************************
 Implementation of Fast Fourier Transformation on Finite Elements
 *****************************************************************************
 * @author     Marius van der Wijden
 * Copyright [2019] [Marius van der Wijden]
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

#pragma once
#include <cstdint>

#ifndef DEBUG
#include <cuda.h>
#include <cuda_runtime.h>

#define cu_fun __host__ __device__ 
#else

#define cu_fun
#include <cstdio>
#include <cassert>

#endif

#define SIZE (256 / 32)

namespace fields{

using size_t = decltype(sizeof 1ll);

#ifndef DEBUG
__constant__
#endif
uint32_t _mod [SIZE];

struct Scalar {
	//Intermediate representation
	uint32_t im_rep [SIZE] = {0};
    //Returns zero element
    cu_fun static Scalar zero()
    {
     Scalar res;
        for(size_t i = 0; i < SIZE; i++)
            res.im_rep[i] = 0;
        return res;
    }

    //Returns one element
    cu_fun static Scalar one()
    {
     Scalar res;
            res.im_rep[SIZE - 1] = 1;
        return res;
    }
    //Default constructor
    Scalar() = default;
    //Construct from value
    cu_fun Scalar(const uint32_t value)
    {
        im_rep[SIZE - 1] = value;
    }

    cu_fun Scalar(const uint32_t* value)
    {
        for(size_t i = 0; i < SIZE; i++)
            im_rep[i] = value[i];
    }
};

struct FieldElement {
    Scalar x;
    Scalar y;
};

#ifdef DEBUG
    void prin Scalars: Scalar f)
    {
        for(size_t i = 0; i < SIZE; i++)
            printf("%u, ", f.im_rep[i]);
        printf("\n");
    }

    void testEquality Scalars: Scalar f1, Scalars: Scalar f2)
    {
        for(size_t i = 0; i < SIZE; i++)
            if(f1.im_rep[i] != f2.im_rep[i])
            {
                prin Scalar(f1);
                prin Scalar(f2);
                assert(!"missmatch");
            }
    }
#endif

}