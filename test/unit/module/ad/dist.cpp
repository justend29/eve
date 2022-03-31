/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"
#include <algorithm>
#include <eve/module/core.hpp>
#include <eve/module/ad.hpp>

//==================================================================================================
// Tests for eve::dist
//==================================================================================================
EVE_TEST( "Check behavior of eve::dist(eve::wide)"
        , eve::test::simd::ieee_reals
        , eve::test::generate ( eve::test::randoms(-10, +10)
                              , eve::test::randoms(-10, +10)
                              , eve::test::logicals(0,3)
                              )
        )
<typename T, typename M>(T const& a0, T const& a1, M const& mask)
{
  using eve::detail::map;
  using eve::var;
  using eve::val;
  using eve::der;
  using eve::diff_1st;
  using eve::diff_2nd;

  auto vda0 = var(a0);
  auto vda1 = var(a1);
  TTS_EQUAL(val(eve::dist(vda0, a1))  , eve::dist(a0, a1));
  TTS_EQUAL(val(eve::dist(a0, vda1))  , eve::dist(a0, a1));
  TTS_EQUAL(der(eve::dist(vda0, a1))  , diff_1st(eve::dist)(a0, a1));
  TTS_EQUAL(der(eve::dist(a0, vda1))  , diff_2nd(eve::dist)(a0, a1));

  TTS_EQUAL(val(eve::dist[mask](vda0, a1))  , eve::dist[mask](a0, a1));
  TTS_EQUAL(val(eve::dist[mask](a0, vda1))  , eve::dist[mask](a0, a1));
  TTS_EQUAL(der(eve::dist[mask](vda0, a1))  , eve::if_else(mask, diff_1st(eve::dist)(a0, a1), eve::one));
  TTS_EQUAL(der(eve::dist[mask](a0, vda1))  , eve::if_else(mask, diff_2nd(eve::dist)(a0, a1), eve::zero));
};
