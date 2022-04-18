//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/module/core/decorator/roundings.hpp>
#include <eve/module/core/constant/constant.hpp>
#include <eve/concept/value.hpp>
#include <eve/detail/implementation.hpp>
#include <eve/detail/meta.hpp>
#include <eve/as.hpp>
#include <type_traits>

namespace eve
{
  //================================================================================================
  //! @addtogroup core
  //! @{
  //! @var zeta_three
  //!
  //! @brief Callable object computing the zeta_three constant value.
  //!
  //! **Required header:** `#include <eve/module/core.hpp>`
  //!
  //! | Member       | Effect                                                     |
  //! |:-------------|:-----------------------------------------------------------|
  //! | `operator()` | Computes the zeta_three constant                              |
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
  //! the zeta_three constant in the chosen type.
  //!
  //! ---
  //!
  //! #### Example
  //!
  //! @godbolt{doc/core/zeta_three.cpp}
  //!
  //! @}
  //================================================================================================
  EVE_MAKE_CALLABLE(zeta_three_, zeta_three);

  namespace detail
  {
    template<floating_real_value T>
    EVE_FORCEINLINE auto zeta_three_(EVE_SUPPORTS(cpu_), eve::as<T> const & ) noexcept
    {
      using t_t =  element_type_t<T>;
      if constexpr(std::is_same_v<t_t, float>)       return 0x1.33bap+0;
      else if constexpr(std::is_same_v<t_t, double>) return 0x1.33ba004f00621p+0;
    }

    template<floating_real_value T, typename D>
    EVE_FORCEINLINE constexpr auto zeta_three_(EVE_SUPPORTS(cpu_), D const &, as<T> const &) noexcept
    requires(is_one_of<D>(types<upward_type, downward_type> {}))
    {
      using t_t =  element_type_t<T>;
      if constexpr(std::is_same_v<D, upward_type>)
      {
        if constexpr(std::is_same_v<t_t, float>)  return 0x1.33ba02p+0;
        else if constexpr(std::is_same_v<t_t, double>) return 0x1.33ba004f00622p+0;
      }
      else if constexpr(std::is_same_v<D, downward_type>)
      {
        if constexpr(std::is_same_v<t_t, float>)  return 0x1.33bap+0;
        else if constexpr(std::is_same_v<t_t, double>) return 0x1.33ba004f00621p+0;
      }
    }
  }
}
