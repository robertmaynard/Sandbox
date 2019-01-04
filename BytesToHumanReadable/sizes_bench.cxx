#include <array>
#include <cstdlib>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <benchmark/benchmark.h>

static void slowHumanSize(benchmark::State &state) {
  for (auto _ : state) {
    std::int64_t bytes = state.range(0);
    int prec = 2;
    std::string suffix = "bytes";

    // Might truncate, but it really doesn't matter unless the precision arg
    // is obscenely huge.
    double bytesf = static_cast<double>(bytes);

    if (bytesf >= 1024.) {
      bytesf /= 1024.;
      benchmark::DoNotOptimize(bytesf);
      suffix = "KiB";
    }

    if (bytesf >= 1024.) {
      bytesf /= 1024.;
      benchmark::DoNotOptimize(bytesf);
      suffix = "MiB";
    }

    if (bytesf >= 1024.) {
      bytesf /= 1024.;
      benchmark::DoNotOptimize(bytesf);
      suffix = "GiB";
    }

    if (bytesf >= 1024.) {
      bytesf /= 1024.;
      benchmark::DoNotOptimize(bytesf);
      suffix = "TiB";
    }

    if (bytesf >= 1024.) {
      bytesf /= 1024.;
      benchmark::DoNotOptimize(bytesf);
      suffix = "PiB"; // Dream big...
    }
    benchmark::DoNotOptimize(bytesf);
  }
}

static void fastHumanSize(benchmark::State &state) {
  for (auto _ : state) {
    std::int64_t bytes = state.range(0);
    std::int64_t bytes_minor = bytes;
    int prec = 2;

    constexpr const char *units[] = {"bytes", "KiB", "MiB",
                                     "GiB",   "TiB", "PiB"};
    int i = 0;
    if (bytes >= 1024) {
      bytes = bytes >> 10;
      benchmark::DoNotOptimize(bytes);
      i++;
    }

    if (bytes >= 1024) {
      bytes_minor = bytes;
      bytes = bytes >> 10;
      benchmark::DoNotOptimize(bytes);
      i++;
    }

    if (bytes >= 1024) {
      bytes_minor = bytes;
      bytes = bytes >> 10;
      benchmark::DoNotOptimize(bytes);
      i++;
    }

    if (bytes >= 1024) {
      bytes_minor = bytes;
      bytes = bytes >> 10;
      benchmark::DoNotOptimize(bytes);
      i++;
    }

    if (bytes >= 1024) {
      bytes_minor = bytes;
      benchmark::DoNotOptimize(bytes);
      i++;
    }

    // Might truncate, but it really doesn't matter unless the precision arg
    // is obscenely huge.
    const double bytesf = (bytes_minor == bytes)
                              ? static_cast<double>(bytes_minor)
                              : (static_cast<double>(bytes_minor) / 1024.0);
    benchmark::DoNotOptimize(bytesf);
  }
}

static void fasterHumanSize(benchmark::State &state) {
  for (auto _ : state) {
    std::int64_t bytes = state.range(0);
    int prec = 2;
    std::int64_t bytes_major = bytes;
    std::int64_t bytes_minor = bytes;

    constexpr const char *units[] = {"bytes", "KiB", "MiB",
                                     "GiB",   "TiB", "PiB", "EiB"};

    // this way reduces the number of float divisions we do
    int i = 0;
    while (bytes_major > 1024) {
      bytes_minor = bytes_major;
      bytes_major = bytes_major >> 10; // shift up by 1024
      ++i;
      benchmark::DoNotOptimize(bytes_major);
    }

    const double bytesf = (i == 0) ? static_cast<double>(bytes_minor)
                                   : static_cast<double>(bytes_minor) / 1024.;
    benchmark::DoNotOptimize(bytesf);
  }
}

static void fastestHumanSize(benchmark::State &state) {
  for (auto _ : state) {

    std::int64_t bytes = state.range(0);
    if (bytes == 0) {
      return;
    }

    const int lz = __builtin_clzl(bytes);

    // uint64 with top 10 bits set:
    const std::uint64_t top10Mask = 0xFFC0000000000000;

    // Shift down by the number of leading zeros in bytes,
    // so that bits [msb-9:msb] are set.
    const std::uint64_t msb10Mask = top10Mask >> lz;

    // Mask with every 10th bit set, starting at 1:
    // (so 1, 1024, 1024^2, 1024^3, ....)
    const std::uint64_t divisorMask = 0x1004010040100401;

    const std::uint64_t divisor = msb10Mask & divisorMask;

    // least significant bit of divisor:
    const int divisorLSB = __builtin_ffsl(divisor);

    // Mask of all bits below divisorLSB:
    const std::uint64_t indexMask = 0xFFFFFFFFFFFFFFFF >> (64 - divisorLSB);

    // Number of bits in union of divisorMask and indexMask is i:
    const int i = __builtin_popcount(divisorMask & indexMask);

    const double bytesf =
        static_cast<double>(bytes) / static_cast<double>(divisor);

    benchmark::DoNotOptimize(bytesf);
  }
}

// Register the function as a benchmark
BENCHMARK(slowHumanSize)->RangeMultiplier(128)->Range(8 << 2, 8 << 24);
BENCHMARK(fastHumanSize)->RangeMultiplier(128)->Range(8 << 2, 8 << 24);
BENCHMARK(fasterHumanSize)->RangeMultiplier(128)->Range(8 << 2, 8 << 24);
BENCHMARK(fastestHumanSize)->RangeMultiplier(128)->Range(8 << 2, 8 << 24);
BENCHMARK_MAIN();
