/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "holtwinters/HoltWinters.cuh"

namespace ML {

using namespace MLCommon;

struct HoltWintersInputs {
  int batch_size;
  int frequency;
  ML::SeasonalType seasonal;
  int start_periods;
};

template <typename T>
class HoltWintersTest : public ::testing::TestWithParam<HoltWintersInputs> {
 public:
  void basicTest() {
    params = ::testing::TestWithParam<HoltWintersInputs>::GetParam();
    batch_size = params.batch_size;
    frequency = params.frequency;
    ML::SeasonalType seasonal = params.seasonal;
    start_periods = params.start_periods;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    alpha_ptr = std::vector<T>(batch_size);
    beta_ptr = std::vector<T>(batch_size);
    gamma_ptr = std::vector<T>(batch_size);
    SSE_error_ptr = std::vector<T>(batch_size);
    forecast_ptr = std::vector<T>(batch_size * h);

    std::vector<T> dataset_h = {
      112,    118,    132,    129,    121,    135,    148,    148,    136,
      119,    104,    118,    115,    126,    141,    135,    125,    149,
      170,    170,    158,    133,    114,    140,    145,    150,    178,
      163,    172,    178,    199,    199,    184,    162,    146,    166,
      171,    180,    193,    181,    183,    218,    230,    242,    209,
      191,    172,    194,    196,    196,    236,    235,    229,    243,
      264,    272,    237,    211,    180,    201,    204,    188,    235,
      227,    234,    264,    302,    293,    259,    229,    203,    229,
      242,    233,    267,    269,    270,    315,    364,    347,    312,
      274,    237,    278,    284,    277,    317,    313,    318,    374,
      413,    405,    355,    306,    271,    306,    315,    301,    356,
      348,    355,    422,    465,    467,    404,    347,    305,    336,
      340,    318,    362,    348,    363,    435,    491,    505,    404,
      359,    310,    337,    315.42, 316.31, 316.50, 317.56, 318.13, 318.00,
      316.39, 314.65, 313.68, 313.18, 314.66, 315.43, 316.27, 316.81, 317.42,
      318.87, 319.87, 319.43, 318.01, 315.74, 314.00, 313.68, 314.84, 316.03,
      316.73, 317.54, 318.38, 319.31, 320.42, 319.61, 318.42, 316.63, 314.83,
      315.16, 315.94, 316.85, 317.78, 318.40, 319.53, 320.42, 320.85, 320.45,
      319.45, 317.25, 316.11, 315.27, 316.53, 317.53, 318.58, 318.92, 319.70,
      321.22, 322.08, 321.31, 319.58, 317.61, 316.05, 315.83, 316.91, 318.20,
      319.41, 320.07, 320.74, 321.40, 322.06, 321.73, 320.27, 318.54, 316.54,
      316.71, 317.53, 318.55, 319.27, 320.28, 320.73, 321.97, 322.00, 321.71,
      321.05, 318.71, 317.66, 317.14, 318.70, 319.25, 320.46, 321.43, 322.23,
      323.54, 323.91, 323.59, 322.24, 320.20, 318.48, 317.94, 319.63, 320.87,
      322.17, 322.34, 322.88, 324.25, 324.83, 323.93, 322.38, 320.76, 319.10,
      319.24, 320.56, 321.80, 322.40, 322.99, 323.73, 324.86, 325.40, 325.20,
      323.98, 321.95, 320.18, 320.09, 321.16, 322.74, 26.663, 23.598, 26.931,
      24.740, 25.806, 24.364, 24.477, 23.901, 23.175, 23.227, 21.672, 21.870,
      21.439, 21.089, 23.709, 21.669, 21.752, 20.761, 23.479, 23.824, 23.105,
      23.110, 21.759, 22.073, 21.937, 20.035, 23.590, 21.672, 22.222, 22.123,
      23.950, 23.504, 22.238, 23.142, 21.059, 21.573, 21.548, 20.000, 22.424,
      20.615, 21.761, 22.874, 24.104, 23.748, 23.262, 22.907, 21.519, 22.025,
      22.604, 20.894, 24.677, 23.673, 25.320, 23.583, 24.671, 24.454, 24.122,
      24.252, 22.084, 22.991, 23.287, 23.049, 25.076, 24.037, 24.430, 24.667,
      26.451, 25.618, 25.014, 25.110, 22.964, 23.981, 23.798, 22.270, 24.775,
      22.646, 23.988, 24.737, 26.276, 25.816, 25.210, 25.199, 23.162, 24.707,
      24.364, 22.644, 25.565, 24.062, 25.431, 24.635, 27.009, 26.606, 26.268,
      26.462, 25.246, 25.180, 24.657, 23.304, 26.982, 26.199, 27.210, 26.122,
      26.706, 26.878, 26.152, 26.379, 24.712, 25.688, 24.990, 24.239, 26.721,
      23.475, 24.767, 26.219, 28.361, 28.599, 27.914, 27.784, 25.693, 26.881};

    allocate(data, batch_size * n);
    updateDevice(data, dataset_h.data(), batch_size * n, stream);

    cumlHandle handle;
    handle.setStream(stream);

    ML::HoltWintersFitPredict(handle, n, batch_size, frequency, h,
                              start_periods, seasonal, data, alpha_ptr.data(),
                              beta_ptr.data(), gamma_ptr.data(),
                              SSE_error_ptr.data(), forecast_ptr.data());

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override { CUDA_CHECK(cudaFree(data)); }

 public:
  HoltWintersInputs params;
  T *data;
  int n = 120, h = 50;
  int batch_size, frequency, start_periods;
  std::vector<T> alpha_ptr, beta_ptr, gamma_ptr, SSE_error_ptr, forecast_ptr;
};

const std::vector<HoltWintersInputs> inputsf1 = {
  {3, 12, ML::SeasonalType::ADDITIVE, 2}};
const std::vector<HoltWintersInputs> inputsf2 = {
  {3, 12, ML::SeasonalType::MULTIPLICATIVE, 2}};

typedef HoltWintersTest<float> HoltWintersTestAF;
TEST_P(HoltWintersTestAF, Fit) {
  myPrintHostVector("alpha", alpha_ptr.data(), batch_size);
  myPrintHostVector("beta", beta_ptr.data(), batch_size);
  myPrintHostVector("gamma", gamma_ptr.data(), batch_size);
  myPrintHostVector("forecast", forecast_ptr.data(), batch_size * h);
  myPrintHostVector("error", SSE_error_ptr.data(), batch_size);
  ASSERT_TRUE(true == true);
}

typedef HoltWintersTest<float> HoltWintersTestMF;
TEST_P(HoltWintersTestMF, Fit) {
  myPrintHostVector("alpha", alpha_ptr.data(), batch_size);
  myPrintHostVector("beta", beta_ptr.data(), batch_size);
  myPrintHostVector("gamma", gamma_ptr.data(), batch_size);
  myPrintHostVector("forecast", forecast_ptr.data(), batch_size * h);
  myPrintHostVector("error", SSE_error_ptr.data(), batch_size);
  ASSERT_TRUE(true == true);
}

INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestAF,
                        ::testing::ValuesIn(inputsf1));
INSTANTIATE_TEST_CASE_P(HoltWintersTests, HoltWintersTestMF,
                        ::testing::ValuesIn(inputsf2));

}  // namespace ML