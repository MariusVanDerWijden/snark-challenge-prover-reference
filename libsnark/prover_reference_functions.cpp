#include <cassert>
#include <cstdio>
#include <fstream>
#include <libff/algebra/curves/mnt753/mnt4753/mnt4753_pp.hpp>
#include <libff/algebra/curves/mnt753/mnt6753/mnt6753_pp.hpp>
#include <libff/algebra/scalar_multiplication/multiexp.hpp>
#include <libff/common/profiling.hpp>
#include <libff/common/rng.hpp>
#include <libff/common/utils.hpp>
#include <libsnark/knowledge_commitment/kc_multiexp.hpp>
#include <libsnark/knowledge_commitment/knowledge_commitment.hpp>
#include <libsnark/reductions/r1cs_to_qap/r1cs_to_qap.hpp>
#include <libsnark/serialization.hpp>
#include <omp.h>
#include "printf.h"

#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>

#include <libfqfft/evaluation_domain/domains/basic_radix2_domain.hpp>

#include "prover_reference_include/prover_reference_functions.hpp"

#include <libsnark/device_multiplication.h>

using namespace libff;
using namespace libsnark;

template<typename T, typename FieldT, multi_exp_method Method>
T local_multi_exp_with_mixed_addition(typename std::vector<T>::const_iterator vec_start,
                                typename std::vector<T>::const_iterator vec_end,
                                typename std::vector<FieldT>::const_iterator scalar_start,
                                typename std::vector<FieldT>::const_iterator scalar_end,
                                const size_t chunks)
{
    assert(std::distance(vec_start, vec_end) == std::distance(scalar_start, scalar_end));
    enter_block("Process scalar vector");
    auto value_it = vec_start;
    auto scalar_it = scalar_start;

    const FieldT zero = FieldT::zero();
    const FieldT one = FieldT::one();
    std::vector<FieldT> p;
    std::vector<T> g;

    T acc = T::zero();

    size_t num_skip = 0;
    size_t num_add = 0;
    size_t num_other = 0;

    for (; scalar_it != scalar_end; ++scalar_it, ++value_it)
    {
        if (*scalar_it == zero)
        {
            // do nothing
            ++num_skip;
        }
        else if (*scalar_it == one)
        {
#ifdef USE_MIXED_ADDITION
            acc = acc.mixed_add(*value_it);
#else
            acc = acc + (*value_it);
#endif
            ++num_add;
        }
        else
        {
            p.emplace_back(*scalar_it);
            g.emplace_back(*value_it);
            ++num_other;
        }
    }
    print_indent(); printf("* Elements of w skipped: %zu (%0.2f%%)\n", num_skip, 100.*num_skip/(num_skip+num_add+num_other));
    print_indent(); printf("* Elements of w processed with special addition: %zu (%0.2f%%)\n", num_add, 100.*num_add/(num_skip+num_add+num_other));
    print_indent(); printf("* Elements of w remaining: %zu (%0.2f%%)\n", num_other, 100.*num_other/(num_skip+num_add+num_other));

    
    auto tmp = acc + cuda_multi_exp_inner<T, FieldT>(g.begin(), g.end(), p.begin(), p.end());

    leave_block("Process scalar vector");

    return tmp;
}

const multi_exp_method method = multi_exp_method_BDLO12;

template <typename G, typename Fr>
G multiexp(typename std::vector<Fr>::const_iterator scalar_start,
           typename std::vector<G>::const_iterator g_start, size_t length) {
#ifdef MULTICORE
  const size_t chunks = length; // to override, set OMP_NUM_THREADS env var or call
                             // omp_set_num_threads()
#else
  const size_t chunks = 1;
#endif

  return local_multi_exp_with_mixed_addition<G, Fr, method>(
      g_start, g_start + length, scalar_start, scalar_start + length, chunks);
}

class mnt4753_libsnark::groth16_input {
public:
  std::shared_ptr<std::vector<Fr<mnt4753_pp>>> w;
  std::shared_ptr<std::vector<Fr<mnt4753_pp>>> ca, cb, cc;
  Fr<mnt4753_pp> r;

  groth16_input(const char *path, size_t d, size_t m) {
    w = std::make_shared<std::vector<libff::Fr<mnt4753_pp>>>(
        std::vector<libff::Fr<mnt4753_pp>>());
    ca = std::make_shared<std::vector<libff::Fr<mnt4753_pp>>>(
        std::vector<libff::Fr<mnt4753_pp>>());
    cb = std::make_shared<std::vector<libff::Fr<mnt4753_pp>>>(
        std::vector<libff::Fr<mnt4753_pp>>());
    cc = std::make_shared<std::vector<libff::Fr<mnt4753_pp>>>(
        std::vector<libff::Fr<mnt4753_pp>>());
    FILE *inputs = fopen(path, "r");

    for (size_t i = 0; i < m + 1; ++i) {
      w->emplace_back(read_fr<mnt4753_pp>(inputs));
    }

    for (size_t i = 0; i < d + 1; ++i) {
      ca->emplace_back(read_fr<mnt4753_pp>(inputs));
    }
    for (size_t i = 0; i < d + 1; ++i) {
      cb->emplace_back(read_fr<mnt4753_pp>(inputs));
    }
    for (size_t i = 0; i < d + 1; ++i) {
      cc->emplace_back(read_fr<mnt4753_pp>(inputs));
    }

    r = read_fr<mnt4753_pp>(inputs);

    fclose(inputs);
  }
};

class mnt4753_libsnark::groth16_params {
public:
  size_t d;
  size_t m;
  std::shared_ptr<std::vector<libff::G1<mnt4753_pp>>> A, B1, L, H;
  std::shared_ptr<std::vector<libff::G2<mnt4753_pp>>> B2;

  groth16_params(const char *path) {
    FILE *params = fopen(path, "r");
    d = read_size_t(params);
    m = read_size_t(params);
    A = std::make_shared<std::vector<libff::G1<mnt4753_pp>>>(
        std::vector<libff::G1<mnt4753_pp>>());
    B1 = std::make_shared<std::vector<libff::G1<mnt4753_pp>>>(
        std::vector<libff::G1<mnt4753_pp>>());
    L = std::make_shared<std::vector<libff::G1<mnt4753_pp>>>(
        std::vector<libff::G1<mnt4753_pp>>());
    H = std::make_shared<std::vector<libff::G1<mnt4753_pp>>>(
        std::vector<libff::G1<mnt4753_pp>>());
    B2 = std::make_shared<std::vector<libff::G2<mnt4753_pp>>>(
        std::vector<libff::G2<mnt4753_pp>>());
    for (size_t i = 0; i <= m; ++i) {
      A->emplace_back(read_g1<mnt4753_pp>(params));
    }
    for (size_t i = 0; i <= m; ++i) {
      B1->emplace_back(read_g1<mnt4753_pp>(params));
    }
    for (size_t i = 0; i <= m; ++i) {
      B2->emplace_back(read_g2<mnt4753_pp>(params));
    }
    for (size_t i = 0; i < m - 1; ++i) {
      L->emplace_back(read_g1<mnt4753_pp>(params));
    }
    for (size_t i = 0; i < d; ++i) {
      H->emplace_back(read_g1<mnt4753_pp>(params));
    }
    fclose(params);
  }
};

struct mnt4753_libsnark::evaluation_domain {
  std::shared_ptr<libfqfft::evaluation_domain<Fr<mnt4753_pp>>> data;
};

struct mnt4753_libsnark::field {
  Fr<mnt4753_pp> data;
};

struct mnt4753_libsnark::G1 {
  libff::G1<mnt4753_pp> data;
};

struct mnt4753_libsnark::G2 {
  libff::G2<mnt4753_pp> data;
};

struct mnt4753_libsnark::vector_Fr {
  std::shared_ptr<std::vector<Fr<mnt4753_pp>>> data;
  size_t offset;
};

struct mnt4753_libsnark::vector_G1 {
  std::shared_ptr<std::vector<libff::G1<mnt4753_pp>>> data;
};
struct mnt4753_libsnark::vector_G2 {
  std::shared_ptr<std::vector<libff::G2<mnt4753_pp>>> data;
};

void mnt4753_libsnark::init_public_params() {
  mnt4753_pp::init_public_params();
}

void mnt4753_libsnark::print_G1(mnt4753_libsnark::G1 *a) { a->data.print(); }

void mnt4753_libsnark::print_G2(mnt4753_libsnark::G2 *a) { a->data.print(); }

mnt4753_libsnark::evaluation_domain *
mnt4753_libsnark::get_evaluation_domain(size_t d) {
  return new evaluation_domain{
      .data = libfqfft::get_evaluation_domain<Fr<mnt4753_pp>>(d)};
}

mnt4753_libsnark::G1 *mnt4753_libsnark::G1_add(mnt4753_libsnark::G1 *a,
                                               mnt4753_libsnark::G1 *b) {
  return new mnt4753_libsnark::G1{.data = a->data + b->data};
}

mnt4753_libsnark::G1 *mnt4753_libsnark::G1_scale(field *a, G1 *b) {
  return new G1{.data = a->data * b->data};
}

void mnt4753_libsnark::vector_Fr_muleq(mnt4753_libsnark::vector_Fr *a,
                                       mnt4753_libsnark::vector_Fr *b,
                                       size_t size) {
  size_t a_off = a->offset, b_off = b->offset;
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++) {
    a->data->at(i + a_off) = a->data->at(i + a_off) * b->data->at(i + b_off);
  }
}

void mnt4753_libsnark::vector_Fr_subeq(mnt4753_libsnark::vector_Fr *a,
                                       mnt4753_libsnark::vector_Fr *b,
                                       size_t size) {
  size_t a_off = a->offset, b_off = b->offset;
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++) {
    a->data->at(i + a_off) = a->data->at(i + a_off) - b->data->at(i + b_off);
  }
}

mnt4753_libsnark::vector_Fr *
mnt4753_libsnark::vector_Fr_offset(mnt4753_libsnark::vector_Fr *a,
                                   size_t offset) {
  return new vector_Fr{.data = a->data, .offset = offset};
}

void mnt4753_libsnark::vector_Fr_copy_into(mnt4753_libsnark::vector_Fr *src,
                                           mnt4753_libsnark::vector_Fr *dst,
                                           size_t length) {
  std::cerr << "length is " << length << ", offset is " << src->offset
            << ", size of src is " << src->data->size() << ", size of dst is "
            << dst->data->size() << std::endl;
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < length; i++) {
    // std::cerr << "doing iteration " << i << std::endl;
    dst->data->at(i) = src->data->at(i);
  }
  // std::copy(src->data->begin(), src->data->end(), dst->data->begin() );
}

mnt4753_libsnark::vector_Fr *mnt4753_libsnark::vector_Fr_zeros(size_t length) {
  std::vector<Fr<mnt4753_pp>> data(length, Fr<mnt4753_pp>::zero());
  return new mnt4753_libsnark::vector_Fr{
      .data = std::make_shared<std::vector<Fr<mnt4753_pp>>>(data)};
}

void mnt4753_libsnark::domain_iFFT(mnt4753_libsnark::evaluation_domain *domain,
                                   mnt4753_libsnark::vector_Fr *a) {
  std::vector<Fr<mnt4753_pp>> &data = *a->data;
  domain->data->iFFT(data);
}
void mnt4753_libsnark::domain_cosetFFT(
    mnt4753_libsnark::evaluation_domain *domain,
    mnt4753_libsnark::vector_Fr *a) {
  domain->data->cosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}
void mnt4753_libsnark::domain_icosetFFT(
    mnt4753_libsnark::evaluation_domain *domain,
    mnt4753_libsnark::vector_Fr *a) {
  domain->data->icosetFFT(*a->data, Fr<mnt4753_pp>::multiplicative_generator);
}
void mnt4753_libsnark::domain_divide_by_Z_on_coset(
    mnt4753_libsnark::evaluation_domain *domain,
    mnt4753_libsnark::vector_Fr *a) {
  domain->data->divide_by_Z_on_coset(*a->data);
}
size_t
mnt4753_libsnark::domain_get_m(mnt4753_libsnark::evaluation_domain *domain) {
  return domain->data->m;
}

mnt4753_libsnark::G1 *
mnt4753_libsnark::multiexp_G1(mnt4753_libsnark::vector_Fr *scalar_start,
                              mnt4753_libsnark::vector_G1 *g_start,
                              size_t length) {
                                printf("###########");
  return new mnt4753_libsnark::G1{
      multiexp<libff::G1<mnt4753_pp>, Fr<mnt4753_pp>>(
          scalar_start->data->begin() + scalar_start->offset,
          g_start->data->begin(), length)};
}
mnt4753_libsnark::G2 *
mnt4753_libsnark::multiexp_G2(mnt4753_libsnark::vector_Fr *scalar_start,
                              mnt4753_libsnark::vector_G2 *g_start,
                              size_t length) {
  return new mnt4753_libsnark::G2{
      multiexp<libff::G2<mnt4753_pp>, Fr<mnt4753_pp>>(
          scalar_start->data->begin() + scalar_start->offset,
          g_start->data->begin(), length)};
}

mnt4753_libsnark::groth16_input *
mnt4753_libsnark::read_input(const char *path,
                             mnt4753_libsnark::groth16_params *params) {
  return new mnt4753_libsnark::groth16_input(path, params->d, params->m);
}

mnt4753_libsnark::vector_Fr *
mnt4753_libsnark::input_w(mnt4753_libsnark::groth16_input *input) {
  return new mnt4753_libsnark::vector_Fr{.data = input->w, .offset = 0};
}
mnt4753_libsnark::vector_Fr *
mnt4753_libsnark::input_ca(mnt4753_libsnark::groth16_input *input) {
  return new mnt4753_libsnark::vector_Fr{.data = input->ca, .offset = 0};
}
mnt4753_libsnark::vector_Fr *mnt4753_libsnark::input_cb(groth16_input *input) {
  return new mnt4753_libsnark::vector_Fr{.data = input->cb, .offset = 0};
}
mnt4753_libsnark::vector_Fr *mnt4753_libsnark::input_cc(groth16_input *input) {
  return new vector_Fr{.data = input->cc, .offset = 0};
}
mnt4753_libsnark::field *mnt4753_libsnark::input_r(groth16_input *input) {
  return new mnt4753_libsnark::field{.data = input->r};
}

mnt4753_libsnark::groth16_params *
mnt4753_libsnark::read_params(const char *path) {
  return new mnt4753_libsnark::groth16_params(path);
}

size_t mnt4753_libsnark::params_d(mnt4753_libsnark::groth16_params *params) {
  return params->d;
}
size_t mnt4753_libsnark::params_m(mnt4753_libsnark::groth16_params *params) {
  return params->m;
}
mnt4753_libsnark::vector_G1 *
mnt4753_libsnark::params_A(mnt4753_libsnark::groth16_params *params) {
  return new mnt4753_libsnark::vector_G1{.data = params->A};
}
mnt4753_libsnark::vector_G1 *
mnt4753_libsnark::params_B1(mnt4753_libsnark::groth16_params *params) {
  return new mnt4753_libsnark::vector_G1{.data = params->B1};
}
mnt4753_libsnark::vector_G1 *
mnt4753_libsnark::params_L(mnt4753_libsnark::groth16_params *params) {
  return new mnt4753_libsnark::vector_G1{.data = params->L};
}
mnt4753_libsnark::vector_G1 *
mnt4753_libsnark::params_H(mnt4753_libsnark::groth16_params *params) {
  return new mnt4753_libsnark::vector_G1{.data = params->H};
}
mnt4753_libsnark::vector_G2 *
mnt4753_libsnark::params_B2(mnt4753_libsnark::groth16_params *params) {
  return new mnt4753_libsnark::vector_G2{.data = params->B2};
}

void mnt4753_libsnark::delete_G1(mnt4753_libsnark::G1 *a) { delete a; }
void mnt4753_libsnark::delete_G2(mnt4753_libsnark::G1 *a) { delete a; }
void mnt4753_libsnark::delete_vector_Fr(mnt4753_libsnark::vector_Fr *a) {
  delete a;
}
void mnt4753_libsnark::delete_vector_G1(mnt4753_libsnark::vector_G1 *a) {
  delete a;
}
void mnt4753_libsnark::delete_vector_G2(mnt4753_libsnark::vector_G2 *a) {
  delete a;
}
void mnt4753_libsnark::delete_groth16_input(
    mnt4753_libsnark::groth16_input *a) {
  delete a;
}
void mnt4753_libsnark::delete_groth16_params(
    mnt4753_libsnark::groth16_params *a) {
  delete a;
}
void mnt4753_libsnark::delete_evaluation_domain(
    mnt4753_libsnark::evaluation_domain *a) {
  delete a;
}

void mnt4753_libsnark::groth16_output_write(mnt4753_libsnark::G1 *A,
                                            mnt4753_libsnark::G2 *B,
                                            mnt4753_libsnark::G1 *C,
                                            const char *output_path) {
  FILE *out = fopen(output_path, "w");
  write_g1<mnt4753_pp>(out, A->data);
  write_g2<mnt4753_pp>(out, B->data);
  write_g1<mnt4753_pp>(out, C->data);
  fclose(out);
}
class mnt6753_libsnark::groth16_input {
public:
  std::shared_ptr<std::vector<Fr<mnt6753_pp>>> w;
  std::shared_ptr<std::vector<Fr<mnt6753_pp>>> ca, cb, cc;
  Fr<mnt6753_pp> r;

  groth16_input(const char *path, size_t d, size_t m) {
    w = std::make_shared<std::vector<libff::Fr<mnt6753_pp>>>(
        std::vector<libff::Fr<mnt6753_pp>>());
    ca = std::make_shared<std::vector<libff::Fr<mnt6753_pp>>>(
        std::vector<libff::Fr<mnt6753_pp>>());
    cb = std::make_shared<std::vector<libff::Fr<mnt6753_pp>>>(
        std::vector<libff::Fr<mnt6753_pp>>());
    cc = std::make_shared<std::vector<libff::Fr<mnt6753_pp>>>(
        std::vector<libff::Fr<mnt6753_pp>>());
    FILE *inputs = fopen(path, "r");

    for (size_t i = 0; i < m + 1; ++i) {
      w->emplace_back(read_fr<mnt6753_pp>(inputs));
    }

    for (size_t i = 0; i < d + 1; ++i) {
      ca->emplace_back(read_fr<mnt6753_pp>(inputs));
    }
    for (size_t i = 0; i < d + 1; ++i) {
      cb->emplace_back(read_fr<mnt6753_pp>(inputs));
    }
    for (size_t i = 0; i < d + 1; ++i) {
      cc->emplace_back(read_fr<mnt6753_pp>(inputs));
    }

    r = read_fr<mnt6753_pp>(inputs);

    fclose(inputs);
  }
};

class mnt6753_libsnark::groth16_params {
public:
  size_t d;
  size_t m;
  std::shared_ptr<std::vector<libff::G1<mnt6753_pp>>> A, B1, L, H;
  std::shared_ptr<std::vector<libff::G2<mnt6753_pp>>> B2;

  groth16_params(const char *path) {
    FILE *params = fopen(path, "r");
    d = read_size_t(params);
    m = read_size_t(params);
    A = std::make_shared<std::vector<libff::G1<mnt6753_pp>>>(
        std::vector<libff::G1<mnt6753_pp>>());
    B1 = std::make_shared<std::vector<libff::G1<mnt6753_pp>>>(
        std::vector<libff::G1<mnt6753_pp>>());
    L = std::make_shared<std::vector<libff::G1<mnt6753_pp>>>(
        std::vector<libff::G1<mnt6753_pp>>());
    H = std::make_shared<std::vector<libff::G1<mnt6753_pp>>>(
        std::vector<libff::G1<mnt6753_pp>>());
    B2 = std::make_shared<std::vector<libff::G2<mnt6753_pp>>>(
        std::vector<libff::G2<mnt6753_pp>>());
    for (size_t i = 0; i <= m; ++i) {
      A->emplace_back(read_g1<mnt6753_pp>(params));
    }
    for (size_t i = 0; i <= m; ++i) {
      B1->emplace_back(read_g1<mnt6753_pp>(params));
    }
    for (size_t i = 0; i <= m; ++i) {
      B2->emplace_back(read_g2<mnt6753_pp>(params));
    }
    for (size_t i = 0; i < m - 1; ++i) {
      L->emplace_back(read_g1<mnt6753_pp>(params));
    }
    for (size_t i = 0; i < d; ++i) {
      H->emplace_back(read_g1<mnt6753_pp>(params));
    }
    fclose(params);
  }
};

struct mnt6753_libsnark::evaluation_domain {
  std::shared_ptr<libfqfft::evaluation_domain<Fr<mnt6753_pp>>> data;
};

struct mnt6753_libsnark::field {
  Fr<mnt6753_pp> data;
};

struct mnt6753_libsnark::G1 {
  libff::G1<mnt6753_pp> data;
};

struct mnt6753_libsnark::G2 {
  libff::G2<mnt6753_pp> data;
};

struct mnt6753_libsnark::vector_Fr {
  std::shared_ptr<std::vector<Fr<mnt6753_pp>>> data;
  size_t offset;
};

struct mnt6753_libsnark::vector_G1 {
  std::shared_ptr<std::vector<libff::G1<mnt6753_pp>>> data;
};
struct mnt6753_libsnark::vector_G2 {
  std::shared_ptr<std::vector<libff::G2<mnt6753_pp>>> data;
};

void mnt6753_libsnark::init_public_params() {
  mnt6753_pp::init_public_params();
}

void mnt6753_libsnark::print_G1(mnt6753_libsnark::G1 *a) { a->data.print(); }

void mnt6753_libsnark::print_G2(mnt6753_libsnark::G2 *a) { a->data.print(); }

mnt6753_libsnark::evaluation_domain *
mnt6753_libsnark::get_evaluation_domain(size_t d) {
  return new evaluation_domain{
      .data = libfqfft::get_evaluation_domain<Fr<mnt6753_pp>>(d)};
}

mnt6753_libsnark::G1 *mnt6753_libsnark::G1_add(mnt6753_libsnark::G1 *a,
                                               mnt6753_libsnark::G1 *b) {
  return new mnt6753_libsnark::G1{.data = a->data + b->data};
}

mnt6753_libsnark::G1 *mnt6753_libsnark::G1_scale(field *a, G1 *b) {
  return new G1{.data = a->data * b->data};
}

void mnt6753_libsnark::vector_Fr_muleq(mnt6753_libsnark::vector_Fr *a,
                                       mnt6753_libsnark::vector_Fr *b,
                                       size_t size) {
  size_t a_off = a->offset, b_off = b->offset;
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++) {
    a->data->at(i + a_off) = a->data->at(i + a_off) * b->data->at(i + b_off);
  }
}

void mnt6753_libsnark::vector_Fr_subeq(mnt6753_libsnark::vector_Fr *a,
                                       mnt6753_libsnark::vector_Fr *b,
                                       size_t size) {
  size_t a_off = a->offset, b_off = b->offset;
#ifdef MULTICORE
#pragma omp parallel for
#endif
  for (size_t i = 0; i < size; i++) {
    a->data->at(i + a_off) = a->data->at(i + a_off) - b->data->at(i + b_off);
  }
}

mnt6753_libsnark::vector_Fr *
mnt6753_libsnark::vector_Fr_offset(mnt6753_libsnark::vector_Fr *a,
                                   size_t offset) {
  return new vector_Fr{.data = a->data, .offset = offset};
}

void mnt6753_libsnark::vector_Fr_copy_into(mnt6753_libsnark::vector_Fr *src,
                                           mnt6753_libsnark::vector_Fr *dst,
                                           size_t length) {
  std::copy(src->data->begin() + src->offset,
            src->data->begin() + src->offset + length, dst->data->begin());
}

mnt6753_libsnark::vector_Fr *mnt6753_libsnark::vector_Fr_zeros(size_t length) {
  return new mnt6753_libsnark::vector_Fr{
      .data = std::make_shared<std::vector<Fr<mnt6753_pp>>>(
          length, Fr<mnt6753_pp>::zero())};
}

void mnt6753_libsnark::domain_iFFT(mnt6753_libsnark::evaluation_domain *domain,
                                   mnt6753_libsnark::vector_Fr *a) {
  std::vector<Fr<mnt6753_pp>> &data = *a->data;
  domain->data->iFFT(data);
}
void mnt6753_libsnark::domain_cosetFFT(
    mnt6753_libsnark::evaluation_domain *domain,
    mnt6753_libsnark::vector_Fr *a) {
  domain->data->cosetFFT(*a->data, Fr<mnt6753_pp>::multiplicative_generator);
}
void mnt6753_libsnark::domain_icosetFFT(
    mnt6753_libsnark::evaluation_domain *domain,
    mnt6753_libsnark::vector_Fr *a) {
  domain->data->icosetFFT(*a->data, Fr<mnt6753_pp>::multiplicative_generator);
}
void mnt6753_libsnark::domain_divide_by_Z_on_coset(
    mnt6753_libsnark::evaluation_domain *domain,
    mnt6753_libsnark::vector_Fr *a) {
  domain->data->divide_by_Z_on_coset(*a->data);
}
size_t
mnt6753_libsnark::domain_get_m(mnt6753_libsnark::evaluation_domain *domain) {
  return domain->data->m;
}

mnt6753_libsnark::G1 *
mnt6753_libsnark::multiexp_G1(mnt6753_libsnark::vector_Fr *scalar_start,
                              mnt6753_libsnark::vector_G1 *g_start,
                              size_t length) {

  return new mnt6753_libsnark::G1{
      multiexp<libff::G1<mnt6753_pp>, Fr<mnt6753_pp>>(
          scalar_start->data->begin() + scalar_start->offset,
          g_start->data->begin(), length)};
}
mnt6753_libsnark::G2 *
mnt6753_libsnark::multiexp_G2(mnt6753_libsnark::vector_Fr *scalar_start,
                              mnt6753_libsnark::vector_G2 *g_start,
                              size_t length) {
  return new mnt6753_libsnark::G2{
      multiexp<libff::G2<mnt6753_pp>, Fr<mnt6753_pp>>(
          scalar_start->data->begin() + scalar_start->offset,
          g_start->data->begin(), length)};
}

mnt6753_libsnark::groth16_input *
mnt6753_libsnark::read_input(const char *path,
                             mnt6753_libsnark::groth16_params *params) {
  return new mnt6753_libsnark::groth16_input(path, params->d, params->m);
}

mnt6753_libsnark::vector_Fr *
mnt6753_libsnark::input_w(mnt6753_libsnark::groth16_input *input) {
  return new mnt6753_libsnark::vector_Fr{.data = input->w, .offset = 0};
}
mnt6753_libsnark::vector_Fr *
mnt6753_libsnark::input_ca(mnt6753_libsnark::groth16_input *input) {
  return new mnt6753_libsnark::vector_Fr{.data = input->ca, .offset = 0};
}
mnt6753_libsnark::vector_Fr *mnt6753_libsnark::input_cb(groth16_input *input) {
  return new mnt6753_libsnark::vector_Fr{.data = input->cb, .offset = 0};
}
mnt6753_libsnark::vector_Fr *mnt6753_libsnark::input_cc(groth16_input *input) {
  return new vector_Fr{.data = input->cc, .offset = 0};
}
mnt6753_libsnark::field *mnt6753_libsnark::input_r(groth16_input *input) {
  return new mnt6753_libsnark::field{.data = input->r};
}

mnt6753_libsnark::groth16_params *
mnt6753_libsnark::read_params(const char *path) {
  return new mnt6753_libsnark::groth16_params(path);
}

size_t mnt6753_libsnark::params_d(mnt6753_libsnark::groth16_params *params) {
  return params->d;
}
size_t mnt6753_libsnark::params_m(mnt6753_libsnark::groth16_params *params) {
  return params->m;
}
mnt6753_libsnark::vector_G1 *
mnt6753_libsnark::params_A(mnt6753_libsnark::groth16_params *params) {
  return new mnt6753_libsnark::vector_G1{.data = params->A};
}
mnt6753_libsnark::vector_G1 *
mnt6753_libsnark::params_B1(mnt6753_libsnark::groth16_params *params) {
  return new mnt6753_libsnark::vector_G1{.data = params->B1};
}
mnt6753_libsnark::vector_G1 *
mnt6753_libsnark::params_L(mnt6753_libsnark::groth16_params *params) {
  return new mnt6753_libsnark::vector_G1{.data = params->L};
}
mnt6753_libsnark::vector_G1 *
mnt6753_libsnark::params_H(mnt6753_libsnark::groth16_params *params) {
  return new mnt6753_libsnark::vector_G1{.data = params->H};
}
mnt6753_libsnark::vector_G2 *
mnt6753_libsnark::params_B2(mnt6753_libsnark::groth16_params *params) {
  return new mnt6753_libsnark::vector_G2{.data = params->B2};
}

void mnt6753_libsnark::delete_G1(mnt6753_libsnark::G1 *a) { delete a; }
void mnt6753_libsnark::delete_G2(mnt6753_libsnark::G1 *a) { delete a; }
void mnt6753_libsnark::delete_vector_Fr(mnt6753_libsnark::vector_Fr *a) {
  delete a;
}
void mnt6753_libsnark::delete_vector_G1(mnt6753_libsnark::vector_G1 *a) {
  delete a;
}
void mnt6753_libsnark::delete_vector_G2(mnt6753_libsnark::vector_G2 *a) {
  delete a;
}
void mnt6753_libsnark::delete_groth16_input(
    mnt6753_libsnark::groth16_input *a) {
  delete a;
}
void mnt6753_libsnark::delete_groth16_params(
    mnt6753_libsnark::groth16_params *a) {
  delete a;
}
void mnt6753_libsnark::delete_evaluation_domain(
    mnt6753_libsnark::evaluation_domain *a) {
  delete a;
}

void mnt6753_libsnark::groth16_output_write(mnt6753_libsnark::G1 *A,
                                            mnt6753_libsnark::G2 *B,
                                            mnt6753_libsnark::G1 *C,
                                            const char *output_path) {
  FILE *out = fopen(output_path, "w");
  write_g1<mnt6753_pp>(out, A->data);
  write_g2<mnt6753_pp>(out, B->data);
  write_g1<mnt6753_pp>(out, C->data);
  fclose(out);
}
