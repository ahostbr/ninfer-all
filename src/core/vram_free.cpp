#include "core/vram_free.h"

#include "core/device_launch.h"

#include <cuda_runtime_api.h>

#include <mutex>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace ninfer {
namespace {

// NVML's stable C ABI, loaded at run time as core/gpu_clock_floor.cpp does: no NVML headers at
// build time, and a host without the library still starts.
using NvmlReturn = int;
using NvmlDevice = void*;
struct NvmlMemory {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
};
using InitFn                     = NvmlReturn (*)();
using ErrorFn                    = const char* (*)(NvmlReturn);
using ByPciFn                    = NvmlReturn (*)(const char*, NvmlDevice*);
using MemoryFn                   = NvmlReturn (*)(NvmlDevice, NvmlMemory*);
constexpr NvmlReturn kNvmlSuccess = 0;

struct Nvml {
    ErrorFn error_string = nullptr;
    ByPciFn by_pci       = nullptr;
    MemoryFn memory      = nullptr;
    std::string problem; // set when the library could not be used

    [[nodiscard]] std::string error(NvmlReturn result) const {
        const char* text = error_string != nullptr ? error_string(result) : nullptr;
        return text != nullptr ? text : "NVML error " + std::to_string(result);
    }
};

// Loaded once and kept for the process's life (never shut down: other threads may ask later).
const Nvml& nvml() {
    static Nvml loaded;
    static std::once_flag once;
    std::call_once(once, [] {
#if defined(_WIN32)
        // The driver installs nvml.dll in System32; drivers before 2019 kept it in NVSMI.
        HMODULE module = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (module == nullptr) {
            wchar_t root[MAX_PATH];
            const DWORD length = GetEnvironmentVariableW(L"ProgramW6432", root, MAX_PATH);
            if (length > 0 && length < MAX_PATH) {
                const std::wstring path =
                    std::wstring(root) + L"\\NVIDIA Corporation\\NVSMI\\nvml.dll";
                module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            }
        }
        const auto symbol = [&](const char* name) {
            return module != nullptr ? reinterpret_cast<void*>(GetProcAddress(module, name)) : nullptr;
        };
        const bool found = module != nullptr;
#else
        void* library     = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        const auto symbol = [&](const char* name) {
            return library != nullptr ? dlsym(library, name) : nullptr;
        };
        const bool found = library != nullptr;
#endif
        if (!found) {
            loaded.problem = "the NVIDIA management library (NVML) is not available";
            return;
        }
        const auto init     = reinterpret_cast<InitFn>(symbol("nvmlInit_v2"));
        loaded.error_string = reinterpret_cast<ErrorFn>(symbol("nvmlErrorString"));
        loaded.by_pci       = reinterpret_cast<ByPciFn>(symbol("nvmlDeviceGetHandleByPciBusId_v2"));
        loaded.memory       = reinterpret_cast<MemoryFn>(symbol("nvmlDeviceGetMemoryInfo"));
        if (init == nullptr || loaded.by_pci == nullptr || loaded.memory == nullptr) {
            loaded.problem = "the NVML library lacks the memory-information entry points";
            loaded.by_pci  = nullptr;
            return;
        }
        if (const NvmlReturn started = init(); started != kNvmlSuccess) {
            loaded.problem = "NVML did not start: " + loaded.error(started);
            loaded.by_pci  = nullptr;
        }
    });
    return loaded;
}

} // namespace

VramFree read_vram_free() {
    VramFree out;
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    out.cuda = free;

    const Nvml& library = nvml();
    if (library.by_pci == nullptr) {
        out.nvml_problem = library.problem;
        return out;
    }
    int device = 0;
    char bus[32] = {};
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetPCIBusId(bus, static_cast<int>(sizeof(bus)), device) != cudaSuccess) {
        (void)cudaGetLastError();
        out.nvml_problem = "no PCI bus id for the device";
        return out;
    }
    NvmlDevice handle = nullptr;
    NvmlMemory memory{};
    NvmlReturn result = library.by_pci(bus, &handle);
    if (result == kNvmlSuccess) { result = library.memory(handle, &memory); }
    if (result != kNvmlSuccess) {
        out.nvml_problem = library.error(result);
        return out;
    }
    out.nvml = memory.free;
    return out;
}

} // namespace ninfer
