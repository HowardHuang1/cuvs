# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# Set the list of Cython files to build
set(cython_sources kmeans.pyx)
set(linked_libraries cuvs::cuvs cuvs::c_api)

# Build all of the Cython targets
rapids_cython_create_modules(
  CXX
  SOURCE_FILES "${cython_sources}"
  LINKED_LIBRARIES "${linked_libraries}" ASSOCIATED_TARGETS cuvs MODULE_PREFIX cluster_kmeans_
)

# Install pure Python module (Dask multi-GPU KMeans)
# Use same layout as Cython modules (relative to CMAKE_INSTALL_PREFIX)
install(FILES dask_kmeans.py
  DESTINATION cuvs/cluster/kmeans
  COMPONENT python)
