//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#include <tts/tests/range.hpp>
#include <eve/module/math.hpp>
#include <eve/module/core.hpp>
#include "measures.hpp"
#include "producers.hpp"
#include <cmath>

TTS_CASE_TPL("wide random check on csc", EVE_TYPE)
{
  using v_t = eve::element_type_t<T>;

  auto std_csc = tts::vectorize<T>( [](auto e) { return 1/std::sin(double(e)); } );

  if constexpr(eve::platform::supports_denormals)
  {
    eve::exhaustive_producer<T> p(-eve::pio_4(eve::as<v_t>()), eve::pio_4(eve::as<v_t>()));
    TTS_RANGE_CHECK(p, std_csc, eve::quarter_circle(eve::csc));
  }
  else
  {
    eve::exhaustive_producer<T>  p(eve::smallestposval(eve::as<v_t>()), eve::pio_4(eve::as<v_t>()));
    TTS_RANGE_CHECK(p, std_csc, eve::quarter_circle(eve::csc));
  }
}
