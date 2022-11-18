# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from cuml.testing.strategies import (
    standard_regression_datasets,
    regression_datasets,
    split_datasets,
    standard_datasets,
)
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import floating_dtypes


@given(standard_datasets())
def test_standard_datasets_default(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)


@given(
    standard_datasets(
        dtypes=floating_dtypes(sizes=(32,)),
        n_samples=st.integers(10, 20),
        n_features=st.integers(30, 40),
    )
)
def test_standard_datasets(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert 10 <= X.shape[0] <= 20
    assert 30 <= X.shape[1] <= 40
    assert 10 <= y.shape[0] <= 20
    assert y.shape[1] == 1


@given(split_datasets(standard_datasets()))
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_split_datasets(split_dataset):
    X_train, X_test, y_train, y_test = split_dataset

    assert X_train.ndim == X_test.ndim == 2
    assert X_train.shape[1] == X_test.shape[1]
    assert 2 <= (len(X_train) + len(X_test)) <= 200

    assert y_train.ndim == y_test.ndim
    assert y_train.ndim in (0, 1, 2)
    assert (y_train.ndim == 0) or (2 <= (len(y_train) + len(y_test)) <= 200)


@given(standard_regression_datasets())
def test_standard_regression_datasets_default(dataset):
    X, y = dataset
    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)
    assert X.dtype == y.dtype


@given(
    standard_regression_datasets(
        dtypes=floating_dtypes(sizes=64),
        n_samples=st.integers(min_value=2, max_value=200),
        n_features=st.integers(min_value=1, max_value=200),
        n_informative=st.just(10),
        random_state=0,
    )
)
def test_standard_regression_datasets(dataset):

    from sklearn.datasets import make_regression

    X, y = dataset
    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 1 and y.shape[0] <= 200) or y.ndim == 0
    assert X.dtype == y.dtype

    X_cmp, y_cmp = make_regression(
        n_samples=X.shape[0], n_features=X.shape[1], random_state=0
    )

    assert X.dtype.type == X_cmp.dtype.type
    assert X.ndim == X_cmp.ndim
    assert X.shape == X_cmp.shape
    assert y.dtype.type == y_cmp.dtype.type
    assert y.ndim == y_cmp.ndim
    assert y.shape == y_cmp.shape
    assert (X == X_cmp).all()
    assert (y == y_cmp).all()


@given(regression_datasets())
def test_regression_datasets(dataset):
    X, y = dataset

    assert X.ndim == 2
    assert X.shape[0] <= 200
    assert X.shape[1] <= 200
    assert (y.ndim == 0) or (y.ndim in (1, 2) and y.shape[0] <= 200)


@given(split_datasets(regression_datasets()))
@settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_split_regression_datasets(split_dataset):
    X_train, X_test, y_train, y_test = split_dataset

    assert X_train.ndim == X_test.ndim == 2
    assert y_train.ndim == y_test.ndim
    assert y_train.ndim in (0, 1, 2)
    assert 2 <= (len(X_train) + len(X_test)) <= 200
