//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core/decorator/roundings.hpp>
#include <eve/detail/implementation.hpp>
#include <eve/as.hpp>

namespace eve
{
  //================================================================================================
  //! @addtogroup core
  //! @{
  //! @var one
  //!
  //! @brief Callable object computing the 1 value.
  //!
  //! **Required header:** `#include <eve/module/core.hpp>`
  //!
  //! | Member       | Effect                                                     |
  //! |:-------------|:-----------------------------------------------------------|
  //! | `operator()` | Computes the one constant                               |
  //!
  //! ---
  //!
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
  //!  template < value T > T operator()( as<T> const & t) const noexcept;
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //!
  //! **Parameters**
  //!
  //!`t`:   [Type wrapper](@ref eve::as) instance embedding the type of the constant.
  //!
  //! **Return value**
  //!
  //! the call `eve::one(as<T>())` is semantically equivalent to `T(1)`
  //!
  //! ---
  //!
  //! #### Example
  //!
  //! @godbolt{doc/core/one.cpp}
  //!
  //! @}
  //================================================================================================
  EVE_MAKE_CALLABLE(one_, one);
  namespace detail
  {
    template<typename T>
    EVE_FORCEINLINE constexpr auto one_(EVE_SUPPORTS(cpu_), as<T> const &) noexcept
    {
      return T(1);
    }
    
    template<typename T, typename D>
    EVE_FORCEINLINE constexpr auto one_(EVE_SUPPORTS(cpu_), D const &, as<T> const &) noexcept
    requires(is_one_of<D>(types<upward_type, downward_type> {}))
    {
      return one(as<T>());
    }
  }
}
