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
  //! @var log_2
  //!
  //! @brief Callable object computing the constant \f$\log 2\f$.
  //!
  //! **Required header:** `#include <eve/module/math.hpp>`
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
  //! the call `eve::log_2(as<T>())` is semantically equivalent to  `eve::log(T(2.0)`.
  //!
  //! ---
  //!
  //! #### Example
  //!
  //! @godbolt{doc/math/log_2.cpp}
  //!
  //! @}
  //================================================================================================
  EVE_MAKE_CALLABLE(log_2_, log_2);

  namespace detail
  {
    template<floating_value T>
    EVE_FORCEINLINE constexpr auto log_2_(EVE_SUPPORTS(cpu_), as<T> const &) noexcept
    {
      using t_t           = element_type_t<T>;

      if constexpr(std::is_same_v<t_t, float>) return Constant<T,  0X3F317218U>();
      else if constexpr(std::is_same_v<t_t, double>) return Constant<T, 0X3FE62E42FEFA39EFULL>();
    }

    template<typename T, typename D>
    EVE_FORCEINLINE constexpr auto log_2_(EVE_SUPPORTS(cpu_), D const &, as<T> const &) noexcept
    requires(is_one_of<D>(types<upward_type, downward_type> {}))
    {
      using t_t           = element_type_t<T>;
      if constexpr(std::is_same_v<t_t, float>)
      {
        if constexpr(std::is_same_v<D, upward_type>)
          return eve::log_2(as<T>());
        else
          return Constant<T, 0X3F317217U>();
      }
      else
      {
        if constexpr(std::is_same_v<D, downward_type>)
          return eve::log_2(as<T>());
        else
          return Constant<T, 0X3FE62E42FEFA39F0ULL>();
      }
    }
  }
}
