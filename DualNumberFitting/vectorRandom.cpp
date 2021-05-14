#pragma once
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <vector>
typedef std::vector<float>(*funcInp)(std::vector<float>);
//fill a vector with random elements (bool, int, float)
template <typename T>
inline void fillVector(T* vec, float RangeStart, float RangeEnd)
{

    //using T0 = typename std::decay<decltype(*vec.begin())>::type;
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    typename T::value_type type;
    if (typeid(type) == typeid(bool))
    {
        std::uniform_int_distribution<int> dist{ 0, 1 };
        auto gen = [&dist, &mersenne_engine]() {
            return dist(mersenne_engine);
        };
        std::generate(begin(*vec), end(*vec), gen);
    }
    else if (typeid(type) == typeid(int))
    {
        std::uniform_int_distribution<int> dist{ (int)RangeStart, (int)RangeEnd };
        auto gen = [&dist, &mersenne_engine]() {
            return dist(mersenne_engine);
        };
        std::generate(begin(*vec), end(*vec), gen);
    }
    else if (typeid(type) == typeid(float))
    {
        std::uniform_real_distribution<float> dist{ RangeStart, RangeEnd };
        auto gen = [&dist, &mersenne_engine]() {
            return dist(mersenne_engine);
        };
        std::generate(begin(*vec), end(*vec), gen);
    }

}
//fill a 2D vector with random elements (bool, int, float)
template <typename T>
inline void fill2DVector(T* vec, int len, float RangeStart, float RangeEnd)
{
    auto gen = [&RangeStart, &RangeEnd, &len]() {
        typename T::value_type v(len);
        fillVector(&v, RangeStart, RangeEnd);
        return v;
    };
    std::generate(begin(*vec), end(*vec), gen);
}
// C++ template to print vector container elements 
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != v.size() - 1)
            os << ", ";
    }
    os << "]";
    return os;
}
//pick a random index from the input vector, excluding zeros
template <typename T>
inline int pickRandomOne(T* vec)
{
    int len = (*vec).size();
    std::vector<int> o(len);
    int x = 0;
    int y = 0;
    for (typename T::value_type v : *vec)
    {
        if (v)
        {
            o[y] = x;
            y++;
        }
        x++;
    }
    y--;
    if (y < 0)
        return -1;
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::uniform_int_distribution<int> dist{ 0, y };
    return o[dist(mersenne_engine)];
}
//pick random index from vector
template <typename T>
inline int pickRand(T* vec)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    std::uniform_int_distribution<int> dist{ 0, (int)((*vec).size() - 1) };
    return dist(mersenne_engine);
}
//create random var
template <typename T>
inline T my_rand(T RangeStart, T RangeEnd)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    //typename T::value_type type;
    if (typeid(T) == typeid(bool))
    {
        std::uniform_int_distribution<int> dist{ 0, 1 };
        return dist(mersenne_engine);
    }
    else if (typeid(T) == typeid(int))
    {
        std::uniform_int_distribution<int> dist{ (int)RangeStart, (int)RangeEnd };
        return dist(mersenne_engine);
    }
    else if (typeid(T) == typeid(float))
    {
        //float tmp = RangeEnd - RangeStart;
        std::uniform_real_distribution<float> dist{ (float)RangeStart, (float)RangeEnd };
        return dist(mersenne_engine);
    }
    else
    {
        return 0;
    }
}
//mutate an int using a binomial distribution
inline void mutateInt(int* i, int RangeStart, int RangeEnd)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    int range = RangeEnd - RangeStart;
    std::binomial_distribution<> dist{ range,0.5 };
    int di = dist(mersenne_engine);
    //std::cout << di << "\n";
    int k = *i + (di - (int)(0.5 * range));
    if (k <= RangeEnd && k >= RangeStart)
    {
        *i = k;
    }

}
//mutate elements of an int vector if my_rand < p (for each element)
inline void mutateIntVec(std::vector<int>* vec, int RangeStart, int RangeEnd, float p)
{
    std::for_each((*vec).begin(), (*vec).end(), [&RangeStart, &RangeEnd, &p](int& n) {
        float r = my_rand<float>(0.0, 1.0);
        if (r < p)
        {
            mutateInt(&n, RangeStart, RangeEnd);
        }
        });
}
//invert elements of a bool vector if my_rand < p (for each element)
inline void mutateBoolVec(std::vector<bool>* vec, float p)
{
    for (auto i = vec->begin(); i != vec->end(); ++i)
    {
        float r = my_rand<float>(0.0, 1.0);
        if (r < p)
        {
            *i = not *i;
        }
    }

}
//mutate float by ref using a normal distribution
inline void mutateFloat(float* i, float RangeStart, float RangeEnd, float std_dev)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    float range = RangeEnd - RangeStart;
    std::normal_distribution<> dist{ 0, std_dev };
    float di = dist(mersenne_engine);
    //std::cout << di << "\n";
    float k = *i + di;
    if (k <= RangeEnd && k >= RangeStart)
    {
        *i = k;
    }
}
//mutate elements of a float vector
inline void mutateFloatVector(std::vector<float>* vec, float RangeStart, float RangeEnd, float p, float std_dev)
{
    std::for_each((*vec).begin(), (*vec).end(), [&RangeStart, &RangeEnd, &p, &std_dev](float& n) {
        float r = my_rand<float>(0.0, 1.0);
        if (r < p)
        {
            mutateFloat(&n, RangeStart, RangeEnd, std_dev);
        }
        });
}
template <typename T>
inline void mutate(T* i, float RangeStart, float RangeEnd, float std_dev)
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine{ rnd_device() };
    float range = RangeEnd - RangeStart;
    std::normal_distribution<> dist{ 0, std_dev };
    float di = dist(mersenne_engine);
    //std::cout << di << "\n";
    T k = *i + (T)di;
    if (k <= RangeEnd && k >= RangeStart)
    {
        *i = k;
    }
}
//mutate int or float vec with given std deviation, cast to type
template <typename T>
inline void mutateVec(T* vec, float RangeStart, float RangeEnd, float p, float std_dev)
{
    std::for_each((*vec).begin(), (*vec).end(), [&RangeStart, &RangeEnd, &p, &std_dev](typename T::value_type& n) {
        float r = my_rand<float>(0.0, 1.0);
        if (r < p)
        {
            mutate(&n, RangeStart, RangeEnd, std_dev);
        }
        });
}
template <typename T>
inline T vCrossOver(T* v1, T* v2)
{
    int x = 0;
    T vo;
    for (auto i = v1->begin(); i != v1->end(); ++i)
    {
        float r = my_rand(0, 1);
        vo.push_back(((*v1)[x] * (r < 0.5)) + ((*v2)[x] * (r >= 0.5)));
        x++;
    }
    return vo;
}
//fill an x vector with random points from the given range
//fill the y vector with the function values from that x vector
inline void fill2DVectorWithFunc(std::vector<std::vector<float>>* vx, std::vector<std::vector<float>>* vy, funcInp f, int n, float RangeStart, float RangeEnd)
{
    fill2DVector(vx, n, RangeStart, RangeEnd);
    for (int x = 0; x < vx->size(); x++)
    {
        (*vy)[x] = f((*vx)[x]);
    }
}
//takes in a matrix and returns a vector where 0's mean that the given column index is entirely zeroes
template <typename T>
inline std::vector<bool>  findZeroColumns(T* vx)
{
    std::vector<bool> zeroCols((*vx)[0].size(), 0);
    for (int i = 0; i < (*vx).size(); i++)
    {
        for (int j = 0; j < (*vx)[i].size(); j++)
        {
            zeroCols[j] = zeroCols[j] || (*vx)[i][j];
        }
    }

    return zeroCols;
}
