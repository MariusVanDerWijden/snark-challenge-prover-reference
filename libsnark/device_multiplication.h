#pragma once 
#include <vector>

#include <stdio.h>
#include <cassert>

#ifdef __cplusplus

template<typename T, typename FieldT>
T cuda_multi_exp_inner(
    typename std::vector<T>::const_iterator vec_start,
    typename std::vector<T>::const_iterator vec_end,
    typename std::vector<FieldT>::const_iterator scalar_start,
    typename std::vector<FieldT>::const_iterator scalar_end);

#endif