//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core/regular/all.hpp>
#include <eve/detail/implementation.hpp>
#include <eve/traits/as_logical.hpp>
#include <eve/concept/value.hpp>
#include <eve/detail/apply_over.hpp>
#include <eve/concept/compatible.hpp>
#include <eve/module/core/constant/true.hpp>

namespace eve::detail
{
  template<real_value T, real_value U>
  EVE_FORCEINLINE constexpr auto is_ordered_(EVE_SUPPORTS(cpu_)
                                            , T const &a
                                            , U const &b) noexcept
  requires compatible_values<T, U>
  {
    return arithmetic_call(is_ordered, a, b);
  }

  template<real_value T>
  EVE_FORCEINLINE constexpr as_logical_t<T> is_ordered_(EVE_SUPPORTS(cpu_)
                                                       , T const &a
                                                       , T const &b) noexcept
  {
    if constexpr(has_native_abi_v<T>)
    {
      if constexpr(integral_value<T>) return true_(eve::as(a));
      else                            return (a == a) && (b == b);
    }
    else                              return apply_over(is_ordered, a, b);
  }

  // -----------------------------------------------------------------------------------------------
  // logical masked case
  template<conditional_expr C, real_value U, real_value V>
  EVE_FORCEINLINE auto is_ordered_(EVE_SUPPORTS(cpu_), C const &cond, U const &u, V const &v) noexcept
  {
    return logical_mask_op(cond, is_ordered, u, v);
  }
}
