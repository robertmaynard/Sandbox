#define VTKM_DEVICE_ADAPTER VTKM_DEVICE_ADAPTER_CUDA
#define BOOST_SP_DISABLE_THREADS

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include <vtkm/Types.h>
#include <vtkm/cont/ArrayHandle.h>
#include <vtkm/worklet/DispatcherMapField.h>

class Example {
public:
  __host__ __device__ Example() : m_value( 1 ) {}
  __host__ __device__ explicit Example(int v) : m_value( v ) {}
  __host__ __device__ Example( const Example& other ) : m_value( other.m_value ) {}
  __host__ __device__ int value() const { return m_value; }
private:
  int m_value;
};

class CPUStackAllocated : public vtkm::worklet::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn<>, FieldOut<>);
  typedef void ExecutionSignature(_1, _2);

   __host__
  CPUStackAllocated(const Example& e)
  {
    for(std::size_t i = 0; i < 100; ++i)
    {
      this->m_Examples[i] = e;
    }
  }

  __host__ __device__
  CPUStackAllocated(const CPUStackAllocated& other)
  {
    for(std::size_t i = 0; i < 100; ++i)
    {
      this->m_Examples[i] = other.m_Examples[i];
    }
  }

  template<typename T>
  VTKM_EXEC_EXPORT
  void operator()(const T &in, T& out) const
  {
    if(in.value() < 100)
      {
      out = this->m_Examples[in.value()];
      }
    else
      {
      out = in;
      }
  }

  Example m_Examples[100];
};

class CPUHeapAllocated : public vtkm::worklet::WorkletMapField
{
public:
  typedef void ControlSignature(FieldIn<>, FieldOut<>);
  typedef void ExecutionSignature(_1, _2);

  __host__
  CPUHeapAllocated():
    m_Examples(NULL),
    m_Length(0)
  {
  }

   __host__
  CPUHeapAllocated(std::size_t len)
  {
    this->m_Examples = new Example[len];
    for(std::size_t i = 0; i < len; ++i)
    {
      this->m_Examples[i] = Example(i);
    }
    this->m_Length = len;
  }

  __host__ __device__
  CPUHeapAllocated(const CPUHeapAllocated& other):
    m_Examples(other.m_Examples),
    m_Length(other.m_Length)
  {
  }


  ~CPUHeapAllocated()
  {
    delete[] this->m_Examples;
  }

  template<typename T>
  VTKM_EXEC_EXPORT
  void operator()(const T &in, T& out) const
  {
    if(m_Length <  100)
      {
      out = this->m_Examples[in.value()];
      }
    else
      {
      out = in;
      }
  }

  Example* m_Examples;
  std::size_t m_Length;
};


void test_copy_made_on_stack()
{
  Example examp(100);
  CPUStackAllocated worklet(examp);

  std::vector< Example > vexamples(150);
  for(std::size_t i=0; i < 150; ++i)
    { vexamples[i] = Example(i); }

  vtkm::cont::ArrayHandle< Example > inputHandle = vtkm::cont::make_ArrayHandle(vexamples);
  vtkm::cont::ArrayHandle< Example > outputHandle;

  vtkm::worklet::DispatcherMapField<CPUStackAllocated> dispatcher(worklet);
  dispatcher.Invoke(inputHandle, outputHandle);
}

void test_copy_made_on_heap()
{
  CPUHeapAllocated worklet;

  std::vector< Example > vexamples(150);
  for(std::size_t i=0; i < 150; ++i)
    { vexamples[i] = Example(i); }

  vtkm::cont::ArrayHandle< Example > inputHandle = vtkm::cont::make_ArrayHandle(vexamples);
  vtkm::cont::ArrayHandle< Example > outputHandle;

  vtkm::worklet::DispatcherMapField<CPUHeapAllocated> dispatcher(worklet);

  dispatcher.Invoke(inputHandle, outputHandle);
}


int main(int, char**) {
  std::cout << "test_copy_made_on_stack" << std::endl;
  test_copy_made_on_stack();

  std::cout << "test_copy_made_on_heap" << std::endl;
  std::cout << "this throw an exception" << std::endl;
  test_copy_made_on_heap();
  return 0;
}