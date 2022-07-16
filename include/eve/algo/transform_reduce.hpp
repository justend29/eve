//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

#include <eve/algo/array_utils.hpp>
#include <eve/algo/for_each_iteration.hpp>
#include <eve/algo/preprocess_range.hpp>
#include <eve/algo/traits.hpp>
#include <eve/algo/views/convert.hpp>
#include <eve/function/convert.hpp>
#include <eve/module/core.hpp>
#include <eve/traits.hpp>

#include <utility>

namespace eve::algo
{
//================================================================================================
//! @addtogroup algorithms
//! @{
//!  @var transform_reduce
//!
//!  @brief SIMD version of std::transform_reduce for a single range
//!     * default unrolling is 4
//!     * alignes
//!
//!  Due to the nature of how simd algorithms work we need a zero for the reduce operation,
//!  so instead of plus you pass {add, zero} where zero is identity for the add operation.
//!  Instead of zero it can be benefitial to pass eve's constants like `eve::zero`, `eve::one`
//!  because we sometimes have a better implementation.
//!
//!  Unfortunatly passing eve's constants instead of init is not supported at the moment.
//!
//!  NOTE: the interface differs from the standard because we felt this better matches our usecase:
//!        do unary transformation and then accumulate that.
//!
//!  NOTE: we do not provide binary range interface because we felt it would confuse things.
//!        Use zip to get that effect.
//!
//!  NOTE: transform_reduce is needed in the world where views::map exist because views::map is more
//!        flexible and therefor needs more from the operation. In transform_reduce we don't need
//!        the map_op to be a template, for example.
//!
//!  NOTE: compiler can autovectoize reductions, especially with special options.
//!        Maybe you don't need a library implementation.
//!
//!  Interface:
//!
//!   transform_reduce(range, map_op, pair{op, zero}, init) -> decltype(init);
//!   reduce(range, init) -> decltype(init);  - default operation is {eve::add, eve::zero}
//!
//!  **Required header:** `#include <eve/algo/transform_reduce.hpp>`
//!
//! #### Example
//!
//! @godbolt{doc/algo/transform_reduce.cpp}
//!
//! }@
//================================================================================================

template<typename TraitsSupport> struct transform_reduce_ : TraitsSupport
{
  using traits_type = typename TraitsSupport::traits_type;

  template<typename MapOp, typename AddOp, typename Zero, typename SumWide> struct delegate
  {
    MapOp map_op;
    AddOp add_op;
    Zero  zero;

    std::array<SumWide, get_unrolling<traits_type>()> sums;

    delegate(MapOp map_op, AddOp add_op, Zero zero, SumWide init)
        : map_op(map_op)
        , add_op(add_op)
        , zero(zero)
    {
      sums.fill(as_value(zero, as<SumWide> {}));
      sums[0] = add_op(sums[0], init);
    }

    EVE_FORCEINLINE bool step(auto it, eve::relative_conditional_expr auto ignore, auto idx)
    {
      auto loaded = eve::load[ignore](it);
      auto mapped = map_op(loaded);
      auto cvt    = eve::convert(mapped, eve::as<element_type_t<SumWide>> {});
      sums[idx()] = add_op(sums[idx()], if_else(ignore, cvt, zero));
      return false;
    }

    template<typename I, std::size_t size>
    EVE_FORCEINLINE bool unrolled_step(std::array<I, size> arr)
    {
      eve::detail::for_<0, 1, size>([&](auto idx) { step(arr[idx()], ignore_none, idx); });
      return false;
    }

    EVE_FORCEINLINE auto finish()
    {
      auto sum = array_reduce(sums, add_op);
      return eve::reduce(sum, add_op);
    }
  };

  template<typename Rng, typename MapOp, typename AddOp, typename Zero, typename U>
  EVE_FORCEINLINE U
  operator()(Rng&& rng, MapOp map_op, std::pair<AddOp, Zero> add_zero, U init) const
  {
    algo::traits consider_sum_type {consider_types<U>};
    auto         processed =
        preprocess_range(algo::default_to(TraitsSupport::get_traits(), consider_sum_type), rng);

    if( processed.begin() == processed.end() ) return init;

    using I = decltype(processed.begin());

    using sum_wide_t        = wide<U, iterator_cardinal_t<I>>;
    sum_wide_t init_as_wide = eve::as_value(add_zero.second, as<sum_wide_t> {});
    init_as_wide.set(0, init);

    delegate<MapOp, AddOp, Zero, sum_wide_t> d {
        map_op, add_zero.first, add_zero.second, init_as_wide};

    algo::for_each_iteration(processed.traits(), processed.begin(), processed.end())(d);
    return d.finish();
  }

  template<typename Rng, typename MapOp, typename U>
  EVE_FORCEINLINE U operator()(Rng&& rng, MapOp map_op, U init) const
  {
    return operator()(EVE_FWD(rng), map_op, std::pair {eve::add, eve::zero}, init);
  }
};

inline constexpr auto transform_reduce =
    function_with_traits<transform_reduce_>[default_simple_algo_traits];
}
