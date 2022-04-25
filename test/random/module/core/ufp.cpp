//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#include <eve/module/core.hpp>
#include "producers.hpp"
#include "generator.hpp"
#include <cmath>
#include <bit>

EVE_TEST_TYPES("Random check for eve::ufp", eve::test::simd::all_types)
<typename T>(eve::as<T>)
{
  using e_t = eve::element_type_t<T>;
  auto vmin = eve::valmin(eve::as<e_t>());
  auto vmax = eve::valmax(eve::as<e_t>());
   if constexpr(eve::floating_value<e_t>)
  {
    auto std_ufp = [](auto e) -> e_t { return ((e <= 1) ? 0 : std::exp2l(std::floor(std::log2l(e)))); };
    EVE_ULP_RANGE_CHECK( T, eve::uniform_prng<e_t>(vmin, vmax),  std_ufp, eve::ufp );
  }
   else  if constexpr(eve::signed_value<e_t>)
  {
    auto std_ufp = [](auto e) -> e_t { return (e <1 ? 0 : std::ufp(eve::uint_(e))); };
    EVE_ULP_RANGE_CHECK( T, eve::uniform_prng<e_t>(vmin, vmax),  std_ufp, eve::ufp );
  }
  else
  {
    auto std_ufp = [](auto e) -> e_t { return std::ufp(e); };
    eve::uniform_prng<e_t> p(vmin, vmax);
    EVE_ULP_RANGE_CHECK(T, p, std_ufp, eve::ufp);
  }
};
