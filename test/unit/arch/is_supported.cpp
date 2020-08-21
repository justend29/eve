//==================================================================================================
/**
  EVE - Expressive Vector Engine
  Copyright 2018 Joel FALCOU

  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
  SPDX-License-Identifier: MIT
**/
//==================================================================================================

#include <eve/arch/tags.hpp>
#include <eve/arch/spec.hpp>
#include <iostream>
#include <iomanip>

int main()
{
  std::cout << "Static detections:\n";
  std::cout << "========================\n";
  std::cout << "Current API: " << eve::current_api << "\n";
  std::cout << "========================\n";
  std::cout << "X86-like SIMD extensions\n";
  std::cout << "SSE2   : " << std::boolalpha << (eve::current_api >= eve::sse2  ) << "\n";
  std::cout << "SSE3   : " << std::boolalpha << (eve::current_api >= eve::sse3  ) << "\n";
  std::cout << "SSSE3  : " << std::boolalpha << (eve::current_api >= eve::ssse3 ) << "\n";
  std::cout << "SSE4.1 : " << std::boolalpha << (eve::current_api >= eve::sse4_1) << "\n";
  std::cout << "SSE4.2 : " << std::boolalpha << (eve::current_api >= eve::sse4_2) << "\n";
  std::cout << "AVX    : " << std::boolalpha << (eve::current_api >= eve::avx   ) << "\n";
  std::cout << "FMA3   : " << std::boolalpha << eve::supports_fma3                << "\n";
  std::cout << "XOP    : " << std::boolalpha << eve::supports_fma4                << "\n";
  std::cout << "FMA4   : " << std::boolalpha << eve::supports_xop                 << "\n";
  std::cout << "AVX2   : " << std::boolalpha << (eve::current_api >= eve::avx2  ) << "\n";
  std::cout << "\n";
  std::cout << "========================\n";
  std::cout << "PPC SIMD extensions\n";
  std::cout << "VSX   : " << std::boolalpha << (eve::current_api >= eve::vsx   ) << "\n";
  std::cout << "VMX   : " << std::boolalpha << (eve::current_api >= eve::vmx   ) << "\n";
  std::cout << "\n";
  std::cout << "========================\n";
  std::cout << "ARM SIMD extensions\n";
  std::cout << "NEON   : " << std::boolalpha << (eve::current_api >= eve::neon   ) << "\n";
  std::cout << "AARCH64: " << std::boolalpha << eve::supports_aarch64              << "\n";
  std::cout << "\n";

  std::cout << "Dynamic detections:\n";
  std::cout << "========================\n";
  std::cout << "X86-like SIMD extensions\n";
  std::cout << "SSE2   : " << std::boolalpha << eve::is_supported(eve::sse2) << "\n";
  std::cout << "SSE3   : " << std::boolalpha << eve::is_supported(eve::sse3) << "\n";
  std::cout << "SSSE3  : " << std::boolalpha << eve::is_supported(eve::ssse3) << "\n";
  std::cout << "SSE4.1 : " << std::boolalpha << eve::is_supported(eve::sse4_1) << "\n";
  std::cout << "SSE4.2 : " << std::boolalpha << eve::is_supported(eve::sse4_2) << "\n";
  std::cout << "AVX    : " << std::boolalpha << eve::is_supported(eve::avx) << "\n";
  std::cout << "FMA3   : " << std::boolalpha << eve::is_supported(eve::fma3) << "\n";
  std::cout << "XOP    : " << std::boolalpha << eve::is_supported(eve::xop) << "\n";
  std::cout << "FMA4   : " << std::boolalpha << eve::is_supported(eve::fma4) << "\n";
  std::cout << "AVX2   : " << std::boolalpha << eve::is_supported(eve::avx2) << "\n";
  std::cout << "\n";
  std::cout << "========================\n";
  std::cout << "PPC SIMD extensions\n";
  std::cout << "VMX   : " << std::boolalpha << eve::is_supported(eve::vmx) << "\n";
  std::cout << "VSX   : " << std::boolalpha << eve::is_supported(eve::vsx) << "\n";
  std::cout << "\n";
  std::cout << "========================\n";
  std::cout << "ARM SIMD extensions\n";
  std::cout << "NEON  : " << std::boolalpha << eve::is_supported(eve::neon) << "\n";
  std::cout << "ASIMD : " << std::boolalpha << eve::is_supported(eve::aarch64) << "\n";

  return 0;
}
