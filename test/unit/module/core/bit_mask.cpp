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
TTS_CASE_TPL( "Check return types of bit_mask"
        , eve::test::simd::all_types
        )
<typename T>(tts::type<T>)
{
  using v_t = eve::element_type_t<T>;

  //regular
  TTS_EXPR_IS( eve::bit_mask(T()  ) , T);
  TTS_EXPR_IS( eve::bit_mask(v_t()) , v_t);
  TTS_EXPR_IS( eve::bit_mask(eve::logical<T>()), T);
};

//==================================================================================================
//  bit_mask tests
//==================================================================================================
TTS_CASE_WITH( "Check behavior of bit_mask(simd) on all types"
        , eve::test::simd::all_types
        , tts::generate ( tts::randoms(eve::valmin, eve::valmax)
                              ,  tts::logicals(0, 3))
        )
<typename T, typename U>(T const& a0, U const & t)
{
  using v_t = eve::element_type_t<T>;
  using eve::bit_mask;
  using eve::detail::map;
  TTS_IEEE_EQUAL( bit_mask(a0)
                , map ( [](auto v) -> v_t
                      {
                        return v ? eve::allbits(eve::as(v)) : eve::zero(eve::as(v));
                      }, a0
                    )
                );
  TTS_IEEE_EQUAL( eve::bit_mask[t](a0), eve::if_else(t, eve::bit_mask(a0), a0));
};
