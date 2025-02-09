//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core.hpp>

namespace eve
{
  //================================================================================================
  //! @addtogroup math
  //! @{
  //! @var minlog2denormal
  //!
  //! @brief Callable object computing the least value for which eve::exp2
  //! returns a non zero result
  //!
  //! **Required header:** `#include <eve/module/math.hpp>`
  //!
  //!
  //! | Member       | Effect                                                     |
  //! |:-------------|:-----------------------------------------------------------|
  //! | `operator()` | Computes the aforementioned constant                               |
  //!
  //! ---
  //!
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
  //!  tempate < floating_value T > T operator()( as<T> const & t) const noexcept;
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //!
  //! **Parameters**
  //!
  //!`t`:   [Type wrapper](@ref eve::as) instance embedding the type of the constant.
  //!
  //! **Return value**
  //!
  //! the call `eve::minlog2denormal(as<T>())` is semantically equivalent to:
  //!   - T(-150.0f) if eve::element_type_t<T> is float
  //!   - T(-1075.0) if eve::element_type_t<T> is double
  //!
  //! This is the greatest value for which `eve::pedantic(eve::exp2)` is  zero
  //!
  //! ---
  //!
  //! #### Example
  //!
  //! @godbolt{doc/math/minlog2denormal.cpp}
  //!
  //! @}
  //================================================================================================
  EVE_MAKE_CALLABLE(minlog2denormal_, minlog2denormal);

  namespace detail
  {
    template<floating_value T>
    EVE_FORCEINLINE constexpr auto minlog2denormal_(EVE_SUPPORTS(cpu_), as<T> const &) noexcept
    {
      using t_t           = element_type_t<T>;

      if constexpr(std::is_same_v<t_t, float>)  return T(-150);
      else if constexpr(std::is_same_v<t_t, double>) return T(-1075);
    }

    template<typename T, typename D>
    EVE_FORCEINLINE constexpr auto minlog2denormal_(EVE_SUPPORTS(cpu_), D const &, as<T> const &) noexcept
    requires(is_one_of<D>(types<upward_type, downward_type> {}))
    {
      return minlog2denormal(as<T>());
    }
  }
}
