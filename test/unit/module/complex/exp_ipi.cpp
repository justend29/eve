//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"
#include "measures.hpp"
#include <eve/module/complex.hpp>
#include <eve/module/math.hpp>
#include <eve/module/core.hpp>
#include <complex>

EVE_TEST( "Check behavior of exp_ipi on scalar"
        , eve::test::scalar::ieee_reals
        , eve::test::generate( eve::test::randoms(-10, 10)
                             , eve::test::randoms(-10, 10))
        )
  <typename T>(T const& a0, T const& a1 )
{
  using e_t = typename T::value_type;
  using c_t = std::complex<e_t>;
  using z_t = eve::complex<e_t>;
  using eve::as;
  auto pis = eve::pi(as<e_t>());
  for(auto e : a0)
  {
    for(auto f : a1)
    {
      TTS_ULP_EQUAL(eve::exp_ipi(eve::complex<e_t>(e, f)),  z_t(std::exp(c_t(-pis*f, pis*e))), 300.0);
      TTS_ULP_EQUAL(eve::exp_ipi(eve::complex<e_t>(e, f)),  z_t(std::exp(c_t(-pis*f, pis*e))), 300.0);
    }
  }
};

EVE_TEST( "Check behavior of exp_ipi on wide"
        , eve::test::simd::ieee_reals
        , eve::test::generate(eve::test::randoms(-10, 10)
                             , eve::test::randoms(-10, 10))
        )
  <typename T>(T const& a0, T const& a1 )
{
  using e_t = typename T::value_type;
  using ce_t = eve::complex<e_t>;
  using z_t = eve::wide<eve::complex<e_t>, typename T::cardinal_type>;
  using c_t = std::complex<e_t>;
  auto pis = eve::pi(eve::as<e_t>());
  auto std_ch = [pis](auto x, auto y){return std::exp(c_t(-pis*y, pis*x)); };
  auto init_with_std = [std_ch](auto a0,  auto a1){
    z_t b;
    for(int i = 0; i !=  eve::cardinal_v<T>; ++i)
    {
      ce_t z(std_ch(a0.get(i), a1.get(i)));
      b.set(i, z);
    }
    return b;
  };
  TTS_ULP_EQUAL(eve::exp_ipi(z_t{a0,a1}), init_with_std(a0, a1), 300.0);
};
