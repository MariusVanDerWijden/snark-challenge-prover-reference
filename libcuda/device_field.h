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

    cu_fun void add(Scalar & fld1, const Scalar & fld2) const;
    cu_fun void mul(Scalar & fld1, const Scalar & fld2) const;
    cu_fun void subtract(Scalar & fld1, const Scalar & fld2) const;

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

    cu_fun Scalar operator*(const Scalar& rhs) const
    {
        Scalar s;
        for(size_t i = 0; i < SIZE; i++)
            s.im_rep[i] = this->im_rep[i];
        mul(s, rhs);
        return s;
    }

    cu_fun Scalar operator+(const Scalar& rhs) const
    {
        Scalar s;
        for(size_t i = 0; i < SIZE; i++)
            s.im_rep[i] = this->im_rep[i];
        add(s, rhs);
        return s;
    }

    cu_fun Scalar operator-(const Scalar& rhs) const
    {
        Scalar s;
        for(size_t i = 0; i < SIZE; i++)
            s.im_rep[i] = this->im_rep[i];
        subtract(s, rhs);
        return s;
    }

    cu_fun Scalar operator-() const
    {
        Scalar s;
        for(size_t i = 0; i < SIZE; i++)
            s.im_rep[i] = this->im_rep[i];
        subtract(s, *this);
        return s;
    }
#ifdef DEBUG
    void printScalars(Scalar f)
    {
        for(size_t i = 0; i < SIZE; i++)
            printf("%u, ", f.im_rep[i]);
        printf("\n");
    }

    void testEquality(Scalar f1, Scalar f2)
    {
        for(size_t i = 0; i < SIZE; i++)
            if(f1.im_rep[i] != f2.im_rep[i])
            {
                printScalar(f1);
                printScalar(f2);
                assert(!"missmatch");
            }
    }
#endif
};

cu_fun long idxOfLNZ(const Scalar& fld);
cu_fun bool hasBitAt(const Scalar& fld, long index);

struct fp2 {
    Scalar x;
    Scalar y;
    const Scalar non_residue = Scalar(13); //13 for mnt4753 and 11 for mnt6753

    fp2 () = default;

    cu_fun static fp2 zero()
    {
        fp2 res;
        res.x = Scalar::zero();
        res.y = Scalar::zero();
        return res;
    }

    cu_fun fp2(Scalar _x, Scalar _y)
    {
        x = _x;
        y = _y;
    }

    cu_fun fp2 operator*(const Scalar& rhs) const
    {
        return fp2(this->x * rhs, this->y * rhs);
    }

    cu_fun fp2 operator*(const fp2& rhs) const
    {
        const Scalar &A = rhs.x;
        const Scalar &B = rhs.y;
        const Scalar &a = this->x;
        const Scalar &b = this->y;
        const Scalar aA = a * A;
        const Scalar bB = b * B;
        return fp2(aA + non_residue * bB, ((a+b) * (A+B) - aA) - bB);
    }

    cu_fun fp2 operator-(const fp2& rhs) const
    {
        return fp2(this->x - rhs.x, this->y - rhs.y);
    }

    cu_fun fp2 operator-() const
    {
        return fp2(-this->x, -this->y);
    }

    cu_fun fp2 operator+(const fp2& rhs) const
    {
        return fp2(this->x + rhs.x, this->y + rhs.y);
    }

    cu_fun void operator=(const fp2& rhs)
    {
        this->x = rhs.x;
        this->y = rhs.y;
    }
};

struct mnt4753_G2 {
    fp2 x;
    fp2 y;
    fp2 z;

    cu_fun mnt4753_G2() {
        x = fp2::zero();
        y = fp2::zero();
        z = fp2::zero();
    }

    cu_fun mnt4753_G2(fp2 _x, fp2 _y, fp2 _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }

    cu_fun static mnt4753_G2 zero()
    {
        return mnt4753_G2(fp2::zero(), fp2::zero(), fp2::zero());
    }

    cu_fun mnt4753_G2 operator+(const mnt4753_G2& other) const
    {
        const fp2 X1Z2 = this->x * other.z;
        const fp2 Y1Z2 = this->y * other.z;
        const fp2 Z1Z2 = this->z * other.z;
        const fp2 u = other.y * this->z - Y1Z2;
        const fp2 uu = u * u;
        const fp2 v = other.x * this->z - X1Z2;
        const fp2 vv = v * v;
        const fp2 vvv = vv * v;
        const fp2 R = vv * X1Z2;
        const fp2 A = uu * Z1Z2 - (vvv + R + R);
        const fp2 X3 = v * A;
        const fp2 Y3 = u * (R-A) - vvv * Y1Z2;
        const fp2 Z3 = vvv * Z1Z2;
        return mnt4753_G2(X3, Y3, Z3);
    }

    cu_fun void operator+=(const mnt4753_G2& other)
    {
        *this = *this + other;
    }

    cu_fun mnt4753_G2 operator-() const
    {
        return mnt4753_G2(this->x, -(this->y), this->z);
    }

    cu_fun mnt4753_G2 operator-(const mnt4753_G2 &other) const
    {
        return (*this) + (-other);
    }

    cu_fun mnt4753_G2 operator*(const Scalar &other) const
    {
        mnt4753_G2 result = zero();

        bool one = false;
        for (long i = idxOfLNZ(other) - 1; i >= 0; --i)
        {
            if (one)
                result = result + result;
            if (hasBitAt(other,i))
            {
                one = true;
                result = result + *this;
            }
        }
        return result;
    }
};

}