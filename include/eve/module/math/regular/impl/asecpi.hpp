//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math/regular/asec.hpp>
#include <eve/module/math/regular/radinpi.hpp>

namespace eve::detail
{
  template<floating_real_value T, decorator D>
  EVE_FORCEINLINE constexpr auto
  asecpi_(EVE_SUPPORTS(cpu_), D const &decorator, T const &a) noexcept
      requires(is_one_of<D>(types<regular_type, raw_type> {}))
  {
    if constexpr( has_native_abi_v<T> )
    {
      return radinpi(decorator(eve::asec)(a));
    }
    else
    {
      return apply_over(decorator(eve::asecpi), a);
    }
  }

  template<floating_real_value T>
  EVE_FORCEINLINE constexpr auto asecpi_(EVE_SUPPORTS(cpu_), T const &a) noexcept
  {
    return asecpi(regular_type(), a);
  }
}
