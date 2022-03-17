//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/detail/overload.hpp>

namespace eve
{
  //================================================================================================
  //! @addtogroup complex
  //! @{
  //! @var imag
  //!
  //! @brief Callable object computing imaginary part of values.
  //!
  //! **Required header:** `#include <eve/module/complex.hpp>`
  //!
  //! #### Members Functions
  //!
  //! | Member       | Effect                                                     |
  //! |:-------------|:-----------------------------------------------------------|
  //! | `operator()` | the  computation of imaginary part                         |
  //!
  //! ---
  //!
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
  //!  auto operator()(value auto x) const noexcept;
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //!
  //! **Parameters**
  //!
  //!`x`:   [value](@ref eve::value).
  //!
  //! **Return value**
  //! 0 if `x` is real or the imaginary part of `x` if x is an instance of eve::complex.
  //!
  //! #### Example
  //!
  //! @godbolt{doc/complex/imag.cpp}
  //!
  //!  @}
  //================================================================================================

  namespace tag { struct real_; }
  template<> struct supports_conditional<tag::real_> : std::false_type {};

  EVE_MAKE_CALLABLE(real_, real);

  namespace detail
  {
    template<floating_real_value V> EVE_FORCEINLINE V real_(EVE_SUPPORTS(cpu_), V) noexcept
    {
      return V(0);
    }
  }
}
