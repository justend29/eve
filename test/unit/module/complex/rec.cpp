//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"
#include "measures.hpp"
#include <eve/module/core.hpp>
#include <eve/module/complex.hpp>

auto vmin =  -1000.0;
auto vmax =  1000.0;

EVE_TEST( "Check behavior of rec on scalar"
        , eve::test::scalar::ieee_reals
        , eve::test::generate(eve::test::randoms(vmin, vmax)
                             , eve::test::randoms(vmin, vmax))
        )
<typename T>(T const& a0,  T const & a1)
{
  using e_t = typename T::value_type;
  using c_t = eve::complex<e_t>;
  for(auto e : a0)
  {
    TTS_EQUAL( eve::rec(e), e_t(1)/e );
    for(auto f : a1)
    {
      auto z1= c_t(e, f);
      TTS_ULP_EQUAL( eve::rec(z1), c_t(1)/z1, 2.0);
    }
  }
};


EVE_TEST( "Check behavior of rec on wide"
        , eve::test::simd::ieee_reals
        , eve::test::generate(eve::test::randoms(vmin, vmax)
                             , eve::test::randoms(vmin, vmax))
        )
<typename T>(T const& a0, T const & a1 )
{
  using e_t = typename T::value_type;
  using z_t = eve::wide<eve::complex<e_t>, typename T::cardinal_type>;

  TTS_ULP_EQUAL( eve::rec(z_t{a0,a1}), (z_t(1)/(z_t{a0,a1})), 2.0);
};
