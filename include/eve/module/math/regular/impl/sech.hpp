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
#include <eve/module/math/constant/log_2.hpp>
#include <eve/module/math/constant/maxlog.hpp>
#include <eve/module/math/regular/exp.hpp>

namespace eve::detail
{

  template<floating_real_value T>
  EVE_FORCEINLINE constexpr auto sech_(EVE_SUPPORTS(cpu_)
                                     , T a0) noexcept
  {
    //////////////////////////////////////////////////////////////////////////////
    // if x = abs(a0) according x < Threshold e =  exp(x) or exp(x/2) is
    // respectively computed
    // *  in the first case sech (e+rec(e))/2
    // *  in the second     sech is (e/2)*e (avoiding undue overflow)
    // Threshold is maxlog - Log_2
    //////////////////////////////////////////////////////////////////////////////
    if constexpr(has_native_abi_v<T>)
    {
      T x = eve::abs(a0);
      auto test1 = (x > maxlog(eve::as<T>())-log_2(eve::as<T>()));
      auto fac = if_else(test1, half(eve::as<T>()),eve::one);
      T tmp1 = exp(-x*fac);
      T tmp = T(2)*tmp1;
      if constexpr(scalar_value<T>)
      {
        return test1 ? tmp1*tmp : tmp/fma(tmp1, tmp1, T(1));
      }
      else
      {
        return if_else(test1, tmp1*tmp, tmp/fma(tmp1, tmp1, T(1)));
      }
    }
    else
    {
      return apply_over(sech, a0);
    }

  }
}
