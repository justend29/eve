//==================================================================================================
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
TTS_CASE_TPL( "Check return types of eve::logical_not(simd)"
              , eve::test::simd::all_types
              )
<typename T>(tts::type<T>)
{
  using eve::logical;
  using v_t = eve::element_type_t<T>;
  TTS_EXPR_IS( eve::logical_not(T())                    , logical<T>   );
  TTS_EXPR_IS( eve::logical_not(v_t())                  , logical<v_t> );
};

//==================================================================================================
// Tests for eve::logical_not
//==================================================================================================

TTS_CASE_WITH( "Check behavior of eve::logical_not(simd)"
        , eve::test::simd::all_types
        , tts::generate (tts::logicals(0, 3))
        )
<typename T>(T const& a0)
{
  using eve::detail::map;
  using v_t = eve::element_type_t<T>;

  TTS_EQUAL(eve::logical_not(a0), map([](auto e) -> v_t { return  !e; }, a0));
  TTS_EQUAL(eve::logical_not(true),  false);
  TTS_EQUAL(eve::logical_not(false),  true);
};
