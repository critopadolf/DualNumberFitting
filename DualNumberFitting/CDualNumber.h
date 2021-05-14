#pragma once
/*

  Source:
  https://blog.demofox.org/2017/02/20/multivariable-dual-numbers-automatic-differentiation/

  Can take full or partial derivatives by passing a dual number through a function and reading the m_dual values

  The author mentions using dual numbers for back propogation, so this project is an implementation of that idea, using his code for Dual Number math

*/
#include <stdio.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

class CDualNumber
{
public:

    CDualNumber();
    CDualNumber(int numVars, float f = 0.0f);
    CDualNumber(int numVars, float f, size_t variableIndex);
    // storage for real and dual values
    int NUMVARIABLES;
    float                           m_real;
    std::vector<float> m_dual;
};

#define PI 3.14159265359f

#define EPSILON 0.001f  // for numeric derivatives calculation


//----------------------------------------------------------------------
// Math Operations
//----------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& os, const CDualNumber& v)
{
    os << "[ " << v.m_real << " ";
    for (int i = 0; i < v.m_dual.size(); ++i) {
        os << v.m_dual[i];
        if (i != v.m_dual.size() - 1)
            os << ", ";
    }
    os << "]\n";
    return os;
}


inline CDualNumber operator + (const CDualNumber& a, const CDualNumber& b)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = a.m_real + b.m_real;
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = a.m_dual[i] + b.m_dual[i];
    return ret;
}


inline CDualNumber operator - (const CDualNumber& a, const CDualNumber& b)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = a.m_real - b.m_real;
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = a.m_dual[i] - b.m_dual[i];
    return ret;
}


inline CDualNumber operator * (const CDualNumber& a, const CDualNumber& b)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = a.m_real * b.m_real;
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = a.m_real * b.m_dual[i] + a.m_dual[i] * b.m_real;
    return ret;
}



inline CDualNumber operator / (const CDualNumber& a, const CDualNumber& b)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = a.m_real / b.m_real;
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = (a.m_dual[i] * b.m_real - a.m_real * b.m_dual[i]) / (b.m_real * b.m_real);
    return ret;
}


inline CDualNumber sqrt(const CDualNumber& a)
{
    CDualNumber ret(a.NUMVARIABLES);
    float sqrtReal = sqrt(a.m_real);
    ret.m_real = sqrtReal;
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = 0.5f * a.m_dual[i] / sqrtReal;
    return ret;
}


inline CDualNumber pow(const CDualNumber& a, float y)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = pow(a.m_real, y);
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = y * a.m_dual[i] * pow(a.m_real, y - 1.0f);
    return ret;
}


inline CDualNumber sin(const CDualNumber& a)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = sin(a.m_real);
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = a.m_dual[i] * cos(a.m_real);
    return ret;
}


inline CDualNumber cos(const CDualNumber& a)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = cos(a.m_real);
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = -a.m_dual[i] * sin(a.m_real);
    return ret;
}


inline CDualNumber tan(const CDualNumber& a)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = tan(a.m_real);
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = a.m_dual[i] / (cos(a.m_real) * cos(a.m_real));
    return ret;
}


inline CDualNumber atan(const CDualNumber& a)
{
    CDualNumber ret(a.NUMVARIABLES);
    ret.m_real = tan(a.m_real);
    for (size_t i = 0; i < a.NUMVARIABLES; ++i)
        ret.m_dual[i] = a.m_dual[i] / (1.0f + a.m_real * a.m_real);
    return ret;
}

