//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/wide.hpp>
#include <eve/arch/top_bits.hpp>
#include <eve/module/core.hpp>
#include <eve/module/complex.hpp>


namespace eve
{
  template<typename T, typename N>
  inline bool compare_equal(wide<eve::complex<T>, N> const &l, wide<eve::complex<T>, N> const &r)
  {
    return eve::all(is_equal(l, r));
  }
  template<typename T>
  inline bool compare_equal(complex<T> const &l, eve::complex<T> const &r)
  {
    return is_equal(l, r);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//==  complex
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace tts
{
  template<typename T, typename N>
  inline double ulp_distance(eve::wide<eve::complex<T>, N> const &l, eve::wide<eve::complex<T>, N> const &r)
  {
    auto diff = l-r;
    auto [rl, il] = l;
    auto [rr, ir] = r;
    auto d = eve::if_else(eve::almost(eve::is_real)(diff)
                         , eve::ulpdist(rl, rr)
                         , eve::if_else(eve::almost(eve::is_imag)(diff)
                                       , eve::ulpdist(il, ir)
                                       , eve::ulpdist(l, r)
                                       )
                         );
    return eve::maximum(d);
  }

  template<typename T, typename N>
  inline double relative_distance(eve::wide<eve::complex<T>, N> const &l, eve::wide<eve::complex<T>, N> const &r)
  {
    return eve::maximum(eve::reldist(l, r));
  }

  template<typename T, typename N>
  inline double absolute_distance(eve::wide<eve::complex<T>, N> const &l, eve::wide<eve::complex<T>, N> const &r)
  {
     return eve::maximum(eve::dist(l, r));
  }

  /////////////
  //==  scalar
  /////////////

  template<typename T>
  inline double ulp_distance(eve::complex<T> const &l, eve::complex<T> const &r)
  {
   return eve::ulpdist(l, r);
  }

  template<typename T>
  inline double relative_distance(eve::complex<T> const &l, eve::complex<T> const &r)
  {
    return eve::reldist(l, r);
  }


  template<typename T>
  inline double absolute_distance(eve::complex<T> const &l, eve::complex<T> const &r)
  {
    auto diff = l-r;
    auto [rl, il] = l;
    auto [rr, ir] = r;
    auto d = eve::if_else(eve::almost(eve::is_real)(diff)
                         , eve::ulpdist(rl, rr)
                         , eve::if_else(eve::almost(eve::is_imag)(diff)
                                       , eve::ulpdist(il, ir)
                                       , eve::ulpdist(l, r)));
    return d;
  }

}
