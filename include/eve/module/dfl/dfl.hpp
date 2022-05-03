//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
**/
//==================================================================================================
#pragma once

#include <eve/concept/vectorizable.hpp>
#include <eve/detail/abi.hpp>
#include <eve/module/core.hpp>
#include <eve/module/math.hpp>
#include <eve/module/complex/regular/complex.hpp>
#include <eve/module/complex/regular/detail/arithmetic.hpp>
#include <eve/module/complex/regular/detail/math.hpp>
#include <eve/module/complex/regular/detail/predicates.hpp>
#include <eve/product_type.hpp>
#include <ostream>

namespace eve
{
  template<typename T, typename U>
  struct make_as_wide : std::conditional<!simd_value<T> && simd_value<U>,as_wide_t<T>,T>
  {};

  template<typename T, typename U>
  struct make_as_wide<T,eve::detail::callable_object<U>>
  {
    using type = T;
  };

  template<typename T, typename U>
  struct make_as_wide<eve::detail::callable_object<T>, U>
  {
    using type = U;
  };

  template<typename T, typename U>
  using make_as_wide_t = typename make_as_wide<T,U>::type;

  //================================================================================================
  //! @addtogroup simd_types
  //! @{
  //================================================================================================
  //! @brief SIMD-compatible representation of dfl numbers
  //!
  //! **Required header:** `#include <eve/module/dfl.hpp>`
  //!
  //! eve::dfl is structure representing dfl number and mean to be used in conjunction with
  //! eve::wide.
  //!
  //! @tparam Type  Underlying floating point type
  //================================================================================================
  template<floating_scalar_value Type>
  struct dfl : struct_support<dfl<Type>, Type, Type>
  {
    using eve_disable_ordering  = void;
    using parent                = struct_support<dfl<Type>, Type, Type>;

    /// Underlying type
    using value_type = Type;

    /// Default constructor
    explicit  dfl(Type r = 0)     noexcept : parent{r,0} {}
              dfl(Type r, Type i) noexcept : parent{r,i} {}

    /// Stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, like<dfl> auto const& z)
    {
      return os <<"{" << std::showpos << high(z) << ",  " << std::showpos << low(z) "}" << std::noshowpos;
    }

    //==============================================================================================
    //  Real/Imag Access
    //==============================================================================================
    EVE_FORCEINLINE friend decltype(auto) tagged_dispatch(eve::tag::high_, like<dfl> auto&& z)
    {
      return get<0>(EVE_FWD(z));
    }

    EVE_FORCEINLINE friend decltype(auto) tagged_dispatch(eve::tag::low_, like<dfl> auto && z)
    {
      return get<1>(EVE_FWD(z));
    }

    //==============================================================================================
    // Functions support
    //==============================================================================================
    template<typename Tag, like<dfl> Z>
    EVE_FORCEINLINE friend  auto  tagged_dispatch(Tag const& tag, Z const& z) noexcept
                            ->    decltype(detail::dfl_unary_dispatch(tag, z))
    {
      return detail::dfl_unary_dispatch(tag, z);
    }

    template<typename Tag, decorator D, like<dfl> Z>
    EVE_FORCEINLINE friend  auto  tagged_dispatch(Tag const& tag, D const& d, Z const& z) noexcept
                            ->    decltype(detail::dfl_unary_dispatch(tag, d, z))
    {
      return detail::dfl_unary_dispatch(tag, d, z);
    }

    template<typename Tag, typename Z1, typename Z2>
    requires(like<Z1,dfl> || like<Z2,dfl>)
    EVE_FORCEINLINE friend  auto  tagged_dispatch ( Tag const& tag
                                                  , Z1 const& z1, Z2 const& z2
                                                  ) noexcept
                            ->    decltype(detail::dfl_binary_dispatch(tag, z1, z2))
    {
      return detail::dfl_binary_dispatch(tag, z1, z2);
    }

    //==============================================================================================
    // Operator +
    //==============================================================================================
    EVE_FORCEINLINE friend auto operator+(like<dfl> auto const& z) noexcept { return z; }

    template<like<dfl> Z1, like<dfl> Z2>
    EVE_FORCEINLINE friend auto& operator+= (Z1& a, Z2 const& b) noexcept
    {

      auto [s1, s2] = two_add(high(a), high(b));
      auto [t1, t2] = two_add(low(a),  low(b) );
      s2 += s1;
      [s1, s2] = quick_two_add(s1, s2, s2);
      s2 += t2;
      [s1, s2] = quick_two_add(s1, s2);
      /* renormalize */
      return self(s1, s2);

      real(self) += real(o);
      imag(self) += imag(o);
      return self;
    }

    EVE_FORCEINLINE friend auto& operator+=(like<dfl> auto& self, callable_i_ const&) noexcept
    {
      imag(self)++;
      return self;
    }

    template<typename Z>
    EVE_FORCEINLINE friend auto& operator+=(like<dfl> auto& self, Z o) noexcept
    requires(like<Z,Type> || std::convertible_to<Z,Type>)
    {
      real(self) += o;
      return self;
    }

    template<like<dfl> Z1, real_value Z2>
    EVE_FORCEINLINE friend  auto operator+(Z1 const& lhs, Z2 const& rhs) noexcept
    requires(requires(make_as_wide_t<Z1,Z2> t) { t += rhs; })
    {
      make_as_wide_t<Z1,Z2> that{lhs};
      return that += rhs;
    }

    template<real_value Z1, like<dfl> Z2>
    EVE_FORCEINLINE friend  auto operator+(Z1 const& lhs, Z2 const& rhs) noexcept
    requires(requires(make_as_wide_t<Z2,Z1> t) { t += lhs; })
    {
      return rhs + lhs;
    }

    //==============================================================================================
    // Operator -
    //==============================================================================================
    template<like<dfl> Z> EVE_FORCEINLINE friend auto operator-(Z const& z) noexcept
    {
      return Z{-real(z), -imag(z)};
    }

//     std::string to_string(int precision,
//                           int width,
//                           std::ios_base::fmtflags
//                           fmt,
//                           bool showpos,
//                           bool uppercase,
//                           char fill) const;


  };
}
