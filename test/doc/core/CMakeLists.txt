##==================================================================================================
##  EVE - Expressive Vector Engine
##  Copyright : EVE Contributors & Maintainers
##  SPDX-License-Identifier: MIT
##==================================================================================================
##==================================================================================================

make_unit( "doc.core" abs.cpp                  )
make_unit( "doc.core" absmax.cpp               )
make_unit( "doc.core" absmin.cpp               )
make_unit( "doc.core" add.cpp                  )
make_unit( "doc.core" agm.cpp                  )
make_unit( "doc.core" average.cpp              )
make_unit( "doc.core" binarize.cpp             )
make_unit( "doc.core" binarize_not.cpp         )
make_unit( "doc.core" bit_and.cpp              )
make_unit( "doc.core" bit_andnot.cpp           )
make_unit( "doc.core" bit_cast.cpp             )
make_unit( "doc.core" bit_ceil.cpp             )
make_unit( "doc.core" bit_floor.cpp            )
make_unit( "doc.core" bitincrement.cpp         )
make_unit( "doc.core" bit_mask.cpp             )
make_unit( "doc.core" bit_notand.cpp           )
make_unit( "doc.core" bit_not.cpp              )
make_unit( "doc.core" bit_notor.cpp            )
make_unit( "doc.core" bitofsign.cpp            )
make_unit( "doc.core" bit_or.cpp               )
make_unit( "doc.core" bit_ornot.cpp            )
make_unit( "doc.core" bit_select.cpp           )
###make_unit( "doc.core" bit_shl.cpp                       )
make_unit( "doc.core" bit_shr.cpp              )
make_unit( "doc.core" bit_width.cpp            )
make_unit( "doc.core" bit_xor.cpp              )
make_unit( "doc.core" broadcast.cpp            )
make_unit( "doc.core" ceil.cpp                 )
make_unit( "doc.core" clamp.cpp                )
make_unit( "doc.core" combine.cpp              )
make_unit( "doc.core" constant.cpp             )
make_unit( "doc.core" convert.cpp              )
make_unit( "doc.core" copysign.cpp             )
make_unit( "doc.core" countl_one.cpp           )
make_unit( "doc.core" countl_zero.cpp          )
make_unit( "doc.core" countr_one.cpp           )
make_unit( "doc.core" countr_zero.cpp          )
make_unit( "doc.core" count_true.cpp           )
make_unit( "doc.core" dec.cpp                  )
make_unit( "doc.core" deco.cpp                 )
make_unit( "doc.core" dist.cpp                 )
make_unit( "doc.core" div.cpp                  )
make_unit( "doc.core" exponent.cpp             )
make_unit( "doc.core" exponentmask.cpp         )
make_unit( "doc.core" false.cpp                )
make_unit( "doc.core" fam.cpp                  )
make_unit( "doc.core" fanm.cpp                 )
make_unit( "doc.core" fdim.cpp                 )
make_unit( "doc.core" firstbitset.cpp          )
make_unit( "doc.core" firstbitunset.cpp        )
make_unit( "doc.core" floor.cpp                )
make_unit( "doc.core" fma.cpp                  )
make_unit( "doc.core" fms.cpp                  )
make_unit( "doc.core" fnma.cpp                 )
make_unit( "doc.core" fnms.cpp                 )
make_unit( "doc.core" frac.cpp                 )
make_unit( "doc.core" fracscale.cpp            )
make_unit( "doc.core" frexp.cpp                )
make_unit( "doc.core" fsm.cpp                  )
make_unit( "doc.core" fsnm.cpp                 )
make_unit( "doc.core" half.cpp                 )
make_unit( "doc.core" hi.cpp                   )
make_unit( "doc.core" ieee_constant.cpp        )
make_unit( "doc.core" if_add.cpp               )
make_unit( "doc.core" if_allbits_else.cpp      )
make_unit( "doc.core" if_else_allbits.cpp      )
make_unit( "doc.core" if_else.cpp              )
make_unit( "doc.core" if_else_mone.cpp         )
make_unit( "doc.core" if_else_one.cpp          )
make_unit( "doc.core" if_else_zero.cpp         )
make_unit( "doc.core" if_mone_else.cpp         )
make_unit( "doc.core" if_mul.cpp               )
make_unit( "doc.core" ifnot_else.cpp           )
make_unit( "doc.core" if_one_else.cpp          )
make_unit( "doc.core" ifrexp.cpp               )
make_unit( "doc.core" if_sub.cpp               )
make_unit( "doc.core" if_zero_else.cpp         )
make_unit( "doc.core" inc.cpp                  )
make_unit( "doc.core" inf.cpp                  )
make_unit( "doc.core" is_denormal.cpp          )
make_unit( "doc.core" is_eqz.cpp               )
make_unit( "doc.core" is_even.cpp              )
make_unit( "doc.core" is_finite.cpp            )
make_unit( "doc.core" is_flint.cpp             )
make_unit( "doc.core" is_gez.cpp               )
make_unit( "doc.core" is_gtz.cpp               )
make_unit( "doc.core" is_imag.cpp              )
make_unit( "doc.core" is_less.cpp              )
make_unit( "doc.core" is_less_equal.cpp        )
make_unit( "doc.core" is_lessgreater.cpp       )
make_unit( "doc.core" is_lez.cpp               )
make_unit( "doc.core" is_ltz.cpp               )
make_unit( "doc.core" is_nan.cpp               )
make_unit( "doc.core" is_negative.cpp          )
make_unit( "doc.core" is_ngez.cpp              )
make_unit( "doc.core" is_ngtz.cpp              )
make_unit( "doc.core" is_nlez.cpp              )
make_unit( "doc.core" is_nltz.cpp              )
make_unit( "doc.core" is_normal.cpp            )
make_unit( "doc.core" is_not_denormal.cpp      )
make_unit( "doc.core" is_not_finite.cpp        )
make_unit( "doc.core" is_not_flint.cpp         )
make_unit( "doc.core" is_not_greater.cpp       )
make_unit( "doc.core" is_not_greater_equal.cpp )
make_unit( "doc.core" is_not_imag.cpp          )
make_unit( "doc.core" is_not_infinite.cpp      )
make_unit( "doc.core" is_not_less.cpp          )
make_unit( "doc.core" is_not_less_equal.cpp    )
make_unit( "doc.core" is_not_nan.cpp           )
make_unit( "doc.core" is_not_real.cpp          )
make_unit( "doc.core" is_odd.cpp               )
make_unit( "doc.core" is_ordered.cpp           )
make_unit( "doc.core" is_positive.cpp          )
make_unit( "doc.core" is_pow2.cpp              )
make_unit( "doc.core" is_real.cpp              )
make_unit( "doc.core" is_unordered.cpp         )
make_unit( "doc.core" ldexp.cpp                )
make_unit( "doc.core" limitexponent.cpp        )
make_unit( "doc.core" lo.cpp                   )
make_unit( "doc.core" logeps.cpp               )
make_unit( "doc.core" logical_andnot.cpp       )
make_unit( "doc.core" logical_notand.cpp       )
make_unit( "doc.core" logical_notor.cpp        )
make_unit( "doc.core" logical_ornot.cpp        )
make_unit( "doc.core" logical_xor.cpp          )
make_unit( "doc.core" lohi.cpp                 )
make_unit( "doc.core" manhattan.cpp            )
make_unit( "doc.core" mantissa.cpp             )
make_unit( "doc.core" mantissamask.cpp         )
make_unit( "doc.core" mask.cpp                 )
make_unit( "doc.core" maxabs.cpp               )
make_unit( "doc.core" max.cpp                  )
make_unit( "doc.core" maxexponent.cpp          )
make_unit( "doc.core" maxexponentm1.cpp        )
make_unit( "doc.core" maxflint.cpp             )
make_unit( "doc.core" maximum.cpp              )
make_unit( "doc.core" maxmag.cpp               )
make_unit( "doc.core" mhalf.cpp                )
make_unit( "doc.core" minabs.cpp               )
make_unit( "doc.core" min.cpp                  )
make_unit( "doc.core" mindenormal.cpp          )
make_unit( "doc.core" minexponent.cpp          )
make_unit( "doc.core" minf.cpp                 )
make_unit( "doc.core" minimum.cpp              )
make_unit( "doc.core" minmag.cpp               )
make_unit( "doc.core" minus.cpp                )
make_unit( "doc.core" modf.cpp                 )
make_unit( "doc.core" mone.cpp                 )
make_unit( "doc.core" mul.cpp                  )
make_unit( "doc.core" mzero.cpp                )
make_unit( "doc.core" nan.cpp                  )
make_unit( "doc.core" nbmantissabits.cpp       )
make_unit( "doc.core" nearest.cpp              )
make_unit( "doc.core" negabsmax.cpp            )
make_unit( "doc.core" negabsmin.cpp            )
make_unit( "doc.core" negate.cpp               )
make_unit( "doc.core" negatenz.cpp             )
make_unit( "doc.core" negmaxabs.cpp            )
make_unit( "doc.core" negminabs.cpp            )
make_unit( "doc.core" nextafter.cpp            )
make_unit( "doc.core" next.cpp                 )
make_unit( "doc.core" none.cpp                 )
make_unit( "doc.core" one.cpp                  )
make_unit( "doc.core" oneminus.cpp             )
make_unit( "doc.core" oneosqrteps.cpp          )
make_unit( "doc.core" plus.cpp                 )
make_unit( "doc.core" popcount.cpp             )
make_unit( "doc.core" prev.cpp                 )
make_unit( "doc.core" rat.cpp                  )
make_unit( "doc.core" rec.cpp                  )
make_unit( "doc.core" reduce.cpp               )
make_unit( "doc.core" rem.cpp                  )
make_unit( "doc.core" remdiv.cpp               )
make_unit( "doc.core" replace_ignored.cpp      )
make_unit( "doc.core" round.cpp                )
make_unit( "doc.core" roundscale.cpp           )
make_unit( "doc.core" rshl.cpp                 )
make_unit( "doc.core" rshr.cpp                 )
make_unit( "doc.core" rsqrt.cpp                )
make_unit( "doc.core" saturate.cpp             )
make_unit( "doc.core" shl.cpp                  )
make_unit( "doc.core" shr.cpp                  )
make_unit( "doc.core" sign.cpp                 )
make_unit( "doc.core" sign_alternate.cpp       )
make_unit( "doc.core" significants.cpp         )
make_unit( "doc.core" signmask.cpp             )
make_unit( "doc.core" signnz.cpp               )
make_unit( "doc.core" smallestposval.cpp       )
make_unit( "doc.core" sqr_abs.cpp              )
make_unit( "doc.core" sqr.cpp                  )
make_unit( "doc.core" sqrt.cpp                 )
make_unit( "doc.core" sqrteps.cpp              )
make_unit( "doc.core" sqrtvalmax.cpp           )
make_unit( "doc.core" sub.cpp                  )
make_unit( "doc.core" swap_if.cpp              )
make_unit( "doc.core" three_fma.cpp            )
make_unit( "doc.core" true.cpp                 )
make_unit( "doc.core" trunc.cpp                )
make_unit( "doc.core" two_add.cpp              )
make_unit( "doc.core" two_prod.cpp             )
make_unit( "doc.core" twotonmb.cpp             )
make_unit( "doc.core" valmax.cpp               )
make_unit( "doc.core" valmin.cpp               )
make_unit( "doc.core" zero.cpp                 )
make_unit( "doc.core" zip.cpp                  )