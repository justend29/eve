//==================================================================================================
/*
  EVE - Expressive Vector Engine
  Copyright : EVE Contributors & Maintainers
  SPDX-License-Identifier: MIT
*/
//==================================================================================================
#pragma once

namespace eve::detail
{
  template <typename N>
  EVE_FORCEINLINE
  wide<std::uint8_t, N> byte_16_runtime_shuffle_(EVE_SUPPORTS(vmx_)
                                               , wide<std::uint8_t, N> what
                                               , wide<std::uint8_t, N> pattern) noexcept
    requires ( N() <= 16 )
  {
    return vec_perm(what.storage(), what.storage(), pattern.storage());
  }
}
