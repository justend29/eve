//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/module/math/constant/pi.hpp>

int main()
{
  auto lmax = eve::pi(eve::as<EVE_VALUE>());
  auto lmin = -lmax;

  auto arg0 = eve::bench::random_<EVE_VALUE>(lmin, lmax);
  auto std__cos = [](auto x){return std::cos(x);};

  eve::bench::experiment xp;
  run<EVE_VALUE>(EVE_NAME(scalar std::cos)  , xp, std__cos                  , arg0);
  run<EVE_VALUE>(EVE_NAME(full_circle(cos)) , xp, eve::full_circle(eve::cos), arg0);
  run<EVE_TYPE >(EVE_NAME(full_circle(cos)) , xp, eve::full_circle(eve::cos), arg0);
  run<EVE_VALUE>(EVE_NAME(eve::cos)      , xp, eve::cos           , arg0);
  run<EVE_TYPE >(EVE_NAME(eve::cos)      , xp, eve::cos           , arg0);
}
