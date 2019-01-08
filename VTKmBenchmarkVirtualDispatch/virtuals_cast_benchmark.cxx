
#include <random>

#include <vtkm/cont/ArrayHandleVirtual.h>
#include <vtkm/cont/ArrayHandleVirtualCoordinates.h>
#include <vtkm/cont/VariantArrayHandle.h>
#include <vtkm/worklet/Invoker.h>
#include <vtkm/worklet/WorkletMapField.h>
#include <vtkm/worklet/WorkletMapTopology.h>

#include <benchmark/benchmark.h>

class Square : public vtkm::worklet::WorkletMapField {
public:
  using MathTypes = vtkm::ListTagBase<vtkm::Vec<vtkm::Float32, 3>,
                                      vtkm::Vec<vtkm::Float64, 3>>;

  using ControlSignature = void(FieldIn<MathTypes>, FieldOut<MathTypes>);
  using ExecutionSignature = void(_1, _2);

  template <typename T, typename U>
  VTKM_EXEC void operator()(T input, U &output) const {
    output = static_cast<U>(input * input);
  }
};

std::mt19937 rng;
constexpr const int ARRAY_SIZE = 2;

//-----------------------------------------------
vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> make_input() {
  using T = vtkm::Vec<vtkm::FloatDefault, 3>;
  T *buffer = new T[ARRAY_SIZE];

  std::uniform_real_distribution<vtkm::FloatDefault> range;

  for (std::size_t i = 0; i < ARRAY_SIZE; ++i) {
    buffer[i] = T(range(rng), range(rng), range(rng));
  }

  auto user_free_function = [](void *ptr) { delete[] static_cast<T *>(ptr); };
  vtkm::cont::internal::Storage<T, vtkm::cont::StorageTagBasic> storage(
      buffer, ARRAY_SIZE, user_free_function);
  return vtkm::cont::ArrayHandle<T>(std::move(storage));
}

vtkm::cont::ArrayHandleVirtual<vtkm::Vec<vtkm::FloatDefault, 3>>
make_virt_input() {
  std::uniform_int_distribution<int> range(0, 2);
  auto result = range(rng);
  if (result == 0) {
    return vtkm::cont::ArrayHandleVirtualCoordinates(make_input());
  } else if (result == 1) {
    vtkm::cont::ArrayHandleUniformPointCoordinates pc(256, 256, 256);
    return vtkm::cont::ArrayHandleVirtualCoordinates(pc);
  } else {
    using RectilinearCoordsArrayType = vtkm::cont::ArrayHandleCartesianProduct<
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
        vtkm::cont::ArrayHandle<vtkm::FloatDefault>>;
    return vtkm::cont::ArrayHandleVirtualCoordinates(
        RectilinearCoordsArrayType{});
  }
}

//-----------------------------------------------
static void OneArgDispatch(benchmark::State &state) {
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> input =
      make_input();
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> output;
  for (auto _ : state) {
    vtkm::worklet::Invoker invoke;
    invoke(Square{}, input, output);
    benchmark::DoNotOptimize(output);
  }
}

//-----------------------------------------------
static void TwoArgDispatch(benchmark::State &state) {
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> input =
      make_input();
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> output;
  for (auto _ : state) {
    vtkm::worklet::Invoker invoke;
    invoke(Square{}, input, output);
    benchmark::DoNotOptimize(output);
  }
}

//-----------------------------------------------
static void OneArgVariantDispatch(benchmark::State &state) {

  using MathTypes = vtkm::ListTagBase<vtkm::Vec<vtkm::Float32, 3>,
                                      vtkm::Vec<vtkm::Float64, 3>>;

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> input =
      make_input();
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> output;
  vtkm::cont::VariantArrayHandleBase<MathTypes> dinput(input);
  for (auto _ : state) {
    vtkm::worklet::Invoker invoke;
    invoke(Square{}, dinput, output);
    benchmark::DoNotOptimize(output);
  }
}

//-----------------------------------------------
static void TwoArgVariantDispatch(benchmark::State &state) {

  using MathTypes = vtkm::ListTagBase<vtkm::Vec<vtkm::Float32, 3>,
                                      vtkm::Vec<vtkm::Float64, 3>>;

  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> input =
      make_input();
  vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>> output;
  vtkm::cont::VariantArrayHandleBase<MathTypes> dinput(input);
  vtkm::cont::VariantArrayHandleBase<MathTypes> doutput(output);
  for (auto _ : state) {
    vtkm::worklet::Invoker invoke;
    invoke(Square{}, dinput, doutput);
    benchmark::DoNotOptimize(output);
  }
}

//-----------------------------------------------
static void IsType(benchmark::State &state) {

  using AHVec = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>>;
  using AHUniform = vtkm::cont::ArrayHandleUniformPointCoordinates;
  using AHCartesian = vtkm::cont::ArrayHandleCartesianProduct<
      vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
      vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
      vtkm::cont::ArrayHandle<vtkm::FloatDefault>>;

  int f = 0;
  auto input = make_virt_input();
  for (auto _ : state) {
    if (vtkm::cont::IsType<AHVec>(input)) {
      f = 1;
      benchmark::DoNotOptimize(f);
    } else if (vtkm::cont::IsType<AHUniform>(input)) {
      f = 2;
      benchmark::DoNotOptimize(f);
    } else if (vtkm::cont::IsType<AHCartesian>(input)) {
      f = 3;
      benchmark::DoNotOptimize(f);
    }
  }
}

//-----------------------------------------------
static void IsTypeVariant(benchmark::State &state) {
  using AHVec = vtkm::cont::ArrayHandle<vtkm::Vec<vtkm::FloatDefault, 3>>;
  using AHUniform = vtkm::cont::ArrayHandleUniformPointCoordinates;
  using AHCartesian = vtkm::cont::ArrayHandleCartesianProduct<
      vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
      vtkm::cont::ArrayHandle<vtkm::FloatDefault>,
      vtkm::cont::ArrayHandle<vtkm::FloatDefault>>;

  int f = 0;
  vtkm::cont::VariantArrayHandle input(make_virt_input());
  for (auto _ : state) {
    if (vtkm::cont::IsType<AHVec>(input)) {
      f = 1;
      benchmark::DoNotOptimize(f);
    } else if (vtkm::cont::IsType<AHUniform>(input)) {
      f = 2;
      benchmark::DoNotOptimize(f);
    } else if (vtkm::cont::IsType<AHCartesian>(input)) {
      f = 3;
      benchmark::DoNotOptimize(f);
    }
  }
}

// Register the function as a benchmark
BENCHMARK(OneArgDispatch);
BENCHMARK(TwoArgDispatch);
BENCHMARK(OneArgVariantDispatch);
BENCHMARK(TwoArgVariantDispatch);
BENCHMARK(IsType);
BENCHMARK(IsTypeVariant);
BENCHMARK_MAIN();
