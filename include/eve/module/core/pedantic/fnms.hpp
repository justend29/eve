//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core/regular/fnms.hpp>
#include <eve/module/core/pedantic/impl/fnms.hpp>

#if defined(EVE_INCLUDE_X86_HEADER)
#  include <eve/module/core/pedantic/impl/simd/x86/fnms.hpp>
#endif
