//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#include <eve/module/core.hpp>
#include <cmath>

int main()
{
  auto lmin = eve::valmin(eve::as<EVE_VALUE>());
  auto lmax = eve::valmax(eve::as<EVE_VALUE>());

  auto const std__is_gtz = [](EVE_VALUE x) -> eve::logical<EVE_VALUE>  { return x > 0; };

  auto arg0 = eve::bench::random_<EVE_VALUE>(lmin,lmax);

  eve::bench::experiment xp;
  run<EVE_VALUE> (EVE_NAME(std__is_gtz) , xp, std__is_gtz, arg0);
  run<EVE_VALUE> (EVE_NAME(is_gtz) , xp, eve::is_gtz, arg0);
  run<EVE_TYPE>  (EVE_NAME(is_gtz) , xp, eve::is_gtz, arg0);
}
