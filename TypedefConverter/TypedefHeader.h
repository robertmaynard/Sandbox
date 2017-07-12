//============================================================================
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2014 Sandia Corporation.
//  Copyright 2014 UT-Battelle, LLC.
//  Copyright 2014 Los Alamos National Security.
//
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//  Under the terms of Contract DE-AC52-06NA25396 with Los Alamos National
//  Laboratory (LANL), the U.S. Government retains certain rights in
//  this software.
//============================================================================

struct ThrustIteratorTag
{
  typedef ThrustIteratorFromArrayPortalTag Type;
};

template <typename PortalType>
struct IteratorTraits
{
  typedef vtkm::cont::ArrayPortalToIterators<PortalType> PortalToIteratorType;
  typedef typename detail::ThrustIteratorTag<typename PortalToIteratorType::IteratorType>::Type Tag;
  typedef typename IteratorChooser<PortalType, Tag>::Type IteratorType;
};

template <typename ValueType_,
          typename PortalTypeFirst_,
          typename PortalTypeSecond_,
          typename PortalTypeThird_>
class ArrayPortalCartesianProduct
{
public:
  typedef ValueType_ ValueType;
  typedef ValueType_ IteratorType;
  typedef PortalTypeFirst_ PortalTypeFirst;
  typedef PortalTypeSecond_ PortalTypeSecond;
  typedef PortalTypeThird_ PortalTypeThird;
};

template <typename T,
          typename FirstHandleType,
          typename SecondHandleType,
          typename ThirdHandleType,
          typename Device>
class ArrayTransfer<T,
                    StorageTagCartesianProduct<FirstHandleType, SecondHandleType, ThirdHandleType>,
                    Device>
{
  typedef StorageTagCartesianProduct<FirstHandleType, SecondHandleType, ThirdHandleType> StorageTag;
  typedef vtkm::cont::internal::Storage<T, StorageTag> StorageType;

public:
  typedef T ValueType;

  typedef typename StorageType::PortalType PortalControl;
  typedef typename StorageType::PortalConstType PortalConstControl;

  typedef vtkm::exec::internal::ArrayPortalCartesianProduct<
    ValueType,
    typename FirstHandleType::template ExecutionTypes<Device>::Portal,
    typename SecondHandleType::template ExecutionTypes<Device>::Portal,
    typename ThirdHandleType::template ExecutionTypes<Device>::Portal>
    PortalExecution;

  typedef vtkm::exec::internal::ArrayPortalCartesianProduct<
    ValueType,
    typename FirstHandleType::template ExecutionTypes<Device>::PortalConst,
    typename SecondHandleType::template ExecutionTypes<Device>::PortalConst,
    typename ThirdHandleType::template ExecutionTypes<Device>::PortalConst>
    PortalConstExecution;

};
