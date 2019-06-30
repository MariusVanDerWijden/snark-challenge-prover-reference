#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;


using namespace std;
using namespace cuFIXNUM;

template <typename fixnum>
struct fp2 {
    fixnum x;
    fixnum y;
    const fixnum non_residue; //13 for mnt4753 and 11 for mnt6753

    fp2 () = default;

    fixnum static fp2 zero()
    {
        fp2 res;
        res.x = fixnum(0);
        res.y = fixnum(0);
        return res;
    }
    
    fp2 operator*(const fixnum& rhs) const
    {
        return fp2(this->x * rhs, this->y * rhs);
    }

    fp2 operator*(const fp2& rhs) const
    {
        const fixnum &A = rhs.x;
        const fixnum &B = rhs.y;
        const fixnum &a = this->x;
        const fixnum &b = this->y;
        const fixnum aA = a * A;
        const fixnum bB = b * B;
        return fp2(aA + non_residue * bB, ((a+b) * (A+B) - aA) - bB);
    }

    fp2 operator-(const fp2& rhs) const
    {
        return fp2(this->x - rhs.x, this->y - rhs.y);
    }

    fp2 operator-() const
    {
        return fp2(-this->x, -this->y);
    }

    fp2 operator+(const fp2& rhs) const
    {
        return fp2(this->x + rhs.x, this->y + rhs.y);
    }

    void operator=(const fp2& rhs)
    {
        this->x = rhs.x;
        this->y = rhs.y;
    }
};

template <typename fixnum>
struct fieldElement {
    fp2<fixnum> x;
    fp2<fixnum> y;
    fp2<fixnum> z;

    fixnum long idxOfLNZ(const fixnum& fld);

    fixnum bool hasBitAt(const fixnum& fld, long index);

    fixnum fieldElement() {
        x = fp2::zero();
        y = fp2::zero();
        z = fp2::zero();
    }

    fixnum fieldElement(fp2 _x, fp2 _y, fp2 _z)
    {
        x = _x;
        y = _y;
        z = _z;
    }

    fixnum static fieldElement zero()
    {
        return fieldElement(fp2::zero(), fp2::zero(), fp2::zero());
    }

    fixnum fieldElement operator+(const fieldElement& other) const
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
        return fieldElement(X3, Y3, Z3);
    }

    fixnum void operator+=(const fieldElement& other)
    {
        *this = *this + other;
    }

    fixnum fieldElement operator-() const
    {
        return fieldElement(this->x, -(this->y), this->z);
    }

    fixnum fieldElement operator-(const fieldElement &other) const
    {
        return (*this) + (-other);
    }

    fixnum fieldElement operator*(const fixnum &other) const
    {
        fieldElement result = zero();

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
