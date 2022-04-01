//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core/regular/derivative.hpp>
#include <eve/module/core/constant/zero.hpp>
#include <eve/module/core/regular/derivative.hpp>

namespace eve::detail
{

  template<floating_real_value T>
  EVE_FORCEINLINE constexpr T trunc_(EVE_SUPPORTS(cpu_)
                                   , diff_type<1> const &
                                   , T x) noexcept
  {
    return zero(as(x));
  }

  // -----------------------------------------------------------------------------------------------
  // Masked case
  template<conditional_expr C, floating_real_value U>
  EVE_FORCEINLINE auto trunc_(EVE_SUPPORTS(cpu_), C const &cond, diff_type<1> const &
                             , U const &t) noexcept
  {
    return mask_op( cond, eve::diff(eve::trunc), t);
  }

}
