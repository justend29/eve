/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"

#include <eve/module/core.hpp>
//==================================================================================================
// Types tests
//==================================================================================================
TTS_CASE_TPL( "Check return types of two_split"
              , eve::test::simd::ieee_reals
              )
<typename T>(tts::type<T>)
{
  using v_t =eve::element_type_t<T>;
  using wtyp = kumi::tuple<T, T>;
  using  typ = kumi::tuple<v_t, v_t>;
  TTS_EXPR_IS(eve::two_split(T()),wtyp);
  TTS_EXPR_IS(eve::two_split(v_t()), typ);
};


//==================================================================================================
//== two_split  tests
//==================================================================================================
TTS_CASE_TPL( "Check behavior of two_split on all types full range"
              , eve::test::simd::ieee_reals
              )
<typename T>(tts::type<T>)
{
  {
    auto [p0, p1] = eve::two_split(T(1));
    TTS_EQUAL(p0, T(1));
    TTS_EQUAL(p1, T(0));
  }

  if (std::is_same_v<eve::element_type_t<T>, double>)
  {
    auto [p0, p1] = eve::two_split(T(1.0e-30));
    TTS_EQUAL (p0, T(0x1.4484cp-100));
    TTS_EQUAL (p1, T(-0x1.143d6p-128));
  }
  else
  {
    auto [p0, p1] = eve::two_split(T(1.0e-30));
    TTS_EQUAL (p0, T(0x1.448p-100f));
    TTS_EQUAL (p1, T(0x1.3p-114f));
  }
};
