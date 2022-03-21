//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#include "test.hpp"
#include <eve/module/complex.hpp>
#include <complex>

EVE_TEST_TYPES( "Check complex::operator+", eve::test::scalar::ieee_reals)
<typename T>(eve::as<T>)
{
  using w_t = eve::wide<T>;
  std::complex<T>             c_s1(T{1}, T{5}), c_s2(T{2}, T{1}),  c_m(T{-3}, T{11});
  eve::complex<T>             z_s1(c_s1), z_s2(c_s2), z_m(c_m);
  TTS_EQUAL(std::real(c_s1*c_s2), std::real(c_m));
  TTS_EQUAL(std::imag(c_s1*c_s2), std::imag(c_m));

  TTS_EQUAL(eve::real(z_s1*z_s2), eve::real(z_m));
  TTS_EQUAL(eve::imag(z_s1*z_s2), eve::imag(z_m));

  auto fill_r = [](auto e, auto) { return T(1+e); };
  auto fill_i = [](auto e, auto) { return T(1)/T(1+3*e); };
  auto fill_r1 = [](auto e, auto) { return T(1-e/2); };
  auto fill_i1 = [](auto e, auto) { return T(-1+3.5*e); };

  using w_t = eve::wide<T>;
  using cw_t = eve::wide<eve::complex<T>>;
  w_t wr(fill_r);
  w_t wi(fill_i);
  w_t wr1(fill_r1);
  w_t wi1(fill_i1);

  eve::wide<eve::complex<T>> z(wr, wi);
  eve::wide<eve::complex<T>> zbar(wr, -wi);
  eve::wide<eve::complex<T>> z1(wr1, wi1);
  auto std_mul =  [](eve::complex<T> a,  eve::complex<T> b) -> eve::complex<T> {
    return std::complex<T>(eve::real(a), eve::imag(a))*std::complex<T>(eve::real(b), eve::imag(b));
  };
  auto my_mul = [std_mul](cw_t a,  cw_t b) -> cw_t { return map(std_mul, a, b); };

  TTS_ULP_EQUAL(z*zbar, my_mul(z, zbar), 0.5);
  TTS_ULP_EQUAL( z*zbar, my_mul(z, zbar), 0.5);
  TTS_ULP_EQUAL(z*z1,   my_mul(z, z1), 0.5);
};
