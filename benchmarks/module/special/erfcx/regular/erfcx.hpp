//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#include <eve/module/core.hpp>

#include <eve/module/special.hpp>
int main()
{
  auto lmin = EVE_VALUE(-1000);
  auto lmax = EVE_VALUE(1000);

  auto arg0 = eve::bench::random_<EVE_VALUE>(lmin,lmax);

  eve::bench::experiment xp;
  run<EVE_VALUE>(EVE_NAME(erfcx) , xp, eve::erfcx , arg0);
  run<EVE_TYPE> (EVE_NAME(erfcx) , xp, eve::erfcx , arg0);

}
