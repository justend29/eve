//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"
#include <eve/module/complex.hpp>

auto vmin =  -1000.0;
auto vmax =  1000.0;

EVE_TEST( "Check behavior of sqr_abs on scalar"
        , eve::test::scalar::ieee_reals
        , eve::test::generate(eve::test::randoms(vmin, vmax)
                             , eve::test::randoms(vmin, vmax))
        )
  <typename T>(T const& a0, T const& a1 )
{
  using e_t = typename T::value_type;
  for(auto e : a0)
  {
    for(auto f : a1)
    {
      TTS_EQUAL( eve::sqr_abs(eve::complex<e_t>(e, f)), eve::sum_of_prod(e, e, f, f) );
    }
  }
};

EVE_TEST( "Check behavior of sqr_abs on wide"
        , eve::test::simd::ieee_reals
        , eve::test::generate(eve::test::randoms(vmin, vmax)
                             , eve::test::randoms(vmin, vmax))
        )
  <typename T>(T const& a0, T const& a1 )
{
  using e_t = typename T::value_type;
  using z_t = eve::wide<eve::complex<e_t>, typename T::cardinal_type>;

  TTS_EQUAL( eve::sqr_abs(z_t{a0,a1}), eve::sum_of_prod(a0, a0, a1, a1) );
};
