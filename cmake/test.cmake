##==================================================================================================
##  EVE - Expressive Vector Engine
##  Copyright 2018 Joel FALCOU
##
##  Licensed under the MIT License <http://opensource.org/licenses/MIT>.
##  SPDX-License-Identifier: MIT
##==================================================================================================

include(download)
include(add_parent_target)

## Basic type roots
set(real_types      double float                      )
set(int_types       int64 int32 int16 int8            )
set(uint_types      uint64 uint32 uint16 uint8        )
set(integral_types  "${int_types};${uint_types}"      )
set(all_types       "${real_types};${integral_types}" )

## Centralize all required setup for unit tests
function(add_unit_test root)
  if( MSVC )
    set( options /std:c++latest -W3 -EHsc)
  else()
    set( options -std=c++17 -Wall -Wno-missing-braces )
  endif()

  foreach(file ${ARGN})
    string(REPLACE ".cpp" ".unit" base ${file})
    string(REPLACE "/"    "." base ${base})
    string(REPLACE "\\"   "." base ${base})
    set(test "${root}.${base}")

    add_executable( ${test}  ${file})
    target_compile_options  ( ${test} PUBLIC ${options} )

    set_property( TARGET ${test}
                  PROPERTY RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                )

    if (CMAKE_CROSSCOMPILING_CMD)
      add_test( NAME ${test}
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                COMMAND ${CMAKE_CROSSCOMPILING_CMD} $<TARGET_FILE:${test}>
              )
    else()
      add_test( NAME ${test}
                WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/test"
                COMMAND $<TARGET_FILE:${test}>
              )
    endif()

    set_target_properties ( ${test} PROPERTIES
                            EXCLUDE_FROM_DEFAULT_BUILD TRUE
                            EXCLUDE_FROM_ALL TRUE
                            ${MAKE_UNIT_TARGET_PROPERTIES}
                          )

    add_dependencies(unit ${test})

    add_parent_target(${test})
  endforeach()
endfunction()

##==================================================================================================
function (list_tests root unit)
  set(sources )

  foreach(e ${ARGN})
    list(APPEND sources "${root}/${e}.cpp")
  endforeach(e)

  add_unit_test( ${unit} "${sources}" )
endfunction()

## Disable testing/doc for tts
set(TTS_BUILD_TEST OFF CACHE INTERNAL "OFF")
set(TTS_BUILD_DOC  OFF CACHE INTERNAL "OFF")

download_project( PROJ                tts
                  GIT_REPOSITORY      git@lri-git:esquisse/tts
                  GIT_TAG             master
                  "UPDATE_DISCONNECTED 1"
                  QUIET
                )

add_subdirectory(${tts_SOURCE_DIR} ${tts_BINARY_DIR})
include_directories("${tts_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/test")

## Setup our tests
add_custom_target(tests)
add_custom_target(unit)
add_dependencies(tests unit)

add_subdirectory(${PROJECT_SOURCE_DIR}/test/)
