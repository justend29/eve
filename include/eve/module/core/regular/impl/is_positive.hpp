//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/detail/implementation.hpp>
#include <eve/module/core/regular/is_gez.hpp>
#include <eve/traits/as_logical.hpp>
#include <eve/concept/value.hpp>
#include <eve/detail/apply_over.hpp>
#include <eve/traits/as_integer.hpp>
#include <eve/module/core/constant/true.hpp>

namespace eve::detail
{
  template<real_value T>
  EVE_FORCEINLINE constexpr auto is_positive_(EVE_SUPPORTS(cpu_)
                                             , T const &v) noexcept
  {
    if constexpr(unsigned_value<T>)        return true_(eve::as(v));
    else if constexpr(has_native_abi_v<T>)
    {
     if constexpr(signed_integral_value<T>) return is_gez(v);
      else if constexpr(simd_value<T>)
      {
        using elt_t =  element_type_t<T>;
        using swi_t = as_wide_t<eve::as_integer_t<elt_t, signed>, cardinal_t<T>>;
        using lwi_t = as_logical_t<as_wide_t<elt_t, cardinal_t<T>>>;
        return bit_cast(is_gez(bit_cast(v,as<swi_t>())), as<lwi_t>());
      }
      else
      {
        using si_t = eve::as_integer_t<T, signed>;
        return bit_cast(is_gez(bit_cast(v,as<si_t>{})), as<as_logical_t<T>>{});
      }
    }
    else return apply_over(is_positive, v);
  }
}
