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
  //! @addtogroup core
  //! @{
  //! @var inv_2pi
  //!
  //! @brief Callable object computing the inv_2pi constant value : \f$\frac{1}{2\pi}\f$.
  //!
  //! **Required header:** `#include <eve/module/math.hpp>`
  //!
  //! | Member       | Effect                                                     |
  //! |:-------------|:-----------------------------------------------------------|
  //! | `operator()` | Computes the inv_2pi constant                              |
  //!
  //! ---
  //!
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{.cpp}
  //!  template < floating_value T > T operator()( as<T> const & t) const noexcept;
  //!  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //!
  //! **Parameters**
  //!
  //!`t`:   [Type wrapper](@ref eve::as) instance embedding the type of the constant.
  //!
  //! **Return value**
  //!
  //! the inv_2pi constant in the chosen type.
  //!
  //! ---
  //!
  //! #### Example
  //!
  //! @godbolt{doc/math/inv_2pi.cpp}
  //!
  //! @}
  //================================================================================================
  EVE_MAKE_CALLABLE(inv_2pi_, inv_2pi);

  namespace detail
  {
    template<floating_real_value T>
    EVE_FORCEINLINE auto inv_2pi_(EVE_SUPPORTS(cpu_), eve::as<T> const & ) noexcept
    {
      using t_t =  element_type_t<T>;
      if constexpr(std::is_same_v<t_t, float>)       return T(0x1.45f306p-3);
      else if constexpr(std::is_same_v<t_t, double>) return T(0x1.45f306dc9c883p-3);
    }

    template<floating_real_value T, typename D>
    EVE_FORCEINLINE constexpr auto inv_2pi_(EVE_SUPPORTS(cpu_), D const &, as<T> const &) noexcept
    requires(is_one_of<D>(types<upward_type, downward_type> {}))
    {
      using t_t =  element_type_t<T>;
      if constexpr(std::is_same_v<D, upward_type>)
      {
        if constexpr(std::is_same_v<t_t, float>)  return T(0x1.45f308p-3);
        else if constexpr(std::is_same_v<t_t, double>) return T(0x1.45f306dc9c883p-3);
      }
      else if constexpr(std::is_same_v<D, downward_type>)
      {
        if constexpr(std::is_same_v<t_t, float>)  return T(0x1.45f306p-3);
        else if constexpr(std::is_same_v<t_t, double>) return T(0x1.45f306dc9c882p-3);
      }
    }
  }
}
