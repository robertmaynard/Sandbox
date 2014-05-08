//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#if !defined(BOOST_PP_IS_ITERATING)

# ifndef __dax_cont_internal_Bindings_h
# define __dax_cont_internal_Bindings_h

#include "ParameterPackCxx03.h"
#include "GetNthType.h"
#include "Tags.h"

#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>


namespace dax { namespace cont {
  namespace arg
    {
    class Field {};
    }
  namespace sig
    {
    class Tag {};
    class In: public Tag {};
    class Out: public Tag {};
    }
}}



namespace dax { namespace cont { namespace internal {

namespace detail {

template <typename R> struct GetConceptAndTagsImpl
{
  typedef R Concept;
  typedef dax::internal::Tags<sig::Tag()> Tags;
};

} // namespace detail

#  define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, "Bindings.h"))
#  include BOOST_PP_ITERATE()

}}} // namespace dax::cont::internal

# endif //__dax_cont_internal_Bindings_h

#else // defined(BOOST_PP_IS_ITERATING)

namespace detail {
using namespace dax::cont::internal::detail;

template <typename R _dax_pp_comma _dax_pp_typename___T>
struct GetConceptAndTagsImpl< R (*)(_dax_pp_T___) >
{
  typedef R Concept;
  typedef typename boost::decay<sig::Tag(_dax_pp_T___)>::type FunctionType;
  typedef dax::internal::Tags< FunctionType > Tags;
};

} // namespace detail

#endif // defined(BOOST_PP_IS_ITERATING)
