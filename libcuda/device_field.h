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

#define SIZE (768 / 32)

namespace fields{

using size_t = decltype(sizeof 1ll);

#ifndef DEBUG
__constant__
#endif
//decimal representation of mod

uint32_t _mod [SIZE] = {115910, 764593169, 270700578, 4007841197, 3086587728, 
 1536143341, 1589111033, 1821890675, 134068517, 3902860685, 
 2580620505, 2707093405, 2971133814, 4061660573, 3087994277, 
 3411246648, 1750781161, 1987204260, 1669861489, 2596546032, 
 3818738770, 752685471, 1586521054, 610172929};

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

    cu_fun static Scalar shuffle_down(unsigned mask, Scalar val, unsigned offset)
    {
        Scalar result;
        for(size_t i = 0; i < SIZE; i++)
#if defined(__CUDA_ARCH__)
            result.im_rep[i] = __shfl_down_sync(mask, val.im_rep[i], offset);
#else
            result.im_rep[i] = val.im_rep[i];
#endif
        return result;
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

    cu_fun static fp2 shuffle_down(unsigned mask, fp2 val, unsigned offset)
    {
        fp2 result;
        result.x = Scalar::shuffle_down(mask, val.x, offset);
        result.y = Scalar::shuffle_down(mask, val.y, offset);
        return result;
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

    cu_fun static mnt4753_G2 shuffle_down(unsigned mask, mnt4753_G2 val, unsigned offset)
    {
        mnt4753_G2 result;
        result.x = fp2::shuffle_down(mask, val.x, offset);
        result.y = fp2::shuffle_down(mask, val.y, offset);
        result.z = fp2::shuffle_down(mask, val.z, offset);
        return result;
    }
};

}

//Modular representation

//mnt4753 mod:
//41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888253786114353726529584385201591605722013126468931404347949840543007986327743462853720628051692141265303114721689601
//mnt6753
//41898490967918953402344214791240637128170709919953949071783502921025352812571106773058893763790338921418070971888458477323173057491593855069696241854796396165721416325350064441470418137846398469611935719059908164220784476160001
//lg2(prime) = 752.8 -> 90 bytes to store -> 24 * 32bit = 768 / 32

//Binary representation
//00000000000000011100010011000110
//00101101100100101100010000010001
//00010000001000101001000000100010
//11101110111000101100110110101101
//10110111111110011001011101010000
//01011011100011111010111111101101
//01011110101101111110100011111001
//01101100100101111101100001110011
//00000111111111011011100100100101
//11101000101000001110110110001101
//10011001110100010010010011011001
//10100001010110101111011110011101
//10110001000101111110011101110110
//11110010000110000000010110011101
//10111000000011110000110110100101
//11001011010100110111111000111000
//01101000010110101100110011101001
//01110110011100100101010010100100
//01100011100010000001000001110001
//10011010110001000010010111110000
//11100011100111010101010001010010
//00101100110111010001000110011111
//01011110100100000110001111011110
//00100100010111101000000000000001

//decimal representation
//= {115910, 764593169, 270700578, 4007841197, 3086587728, 
// 1536143341, 1589111033, 1821890675, 134068517, 3902860685, 
// 2580620505, 2707093405, 2971133814, 4061660573, 3087994277, 
// 3411246648, 1750781161, 1987204260, 1669861489, 2596546032, 
// 3818738770, 752685471, 1586521054, 610172929};
