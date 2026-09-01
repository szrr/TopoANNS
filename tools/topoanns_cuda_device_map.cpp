#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

std::string FormatUuid(const cudaUUID_t& uuid) {
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (int i = 0; i < 16; ++i) {
        out << std::setw(2) << static_cast<unsigned int>(
                                    static_cast<unsigned char>(uuid.bytes[i]));
    }
    return out.str();
}

}  // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        std::cerr << "[topoanns_cuda_device_map] cudaGetDeviceCount failed" << std::endl;
        return 1;
    }

    for (int device = 0; device < count; ++device) {
        cudaDeviceProp props{};
        if (cudaGetDeviceProperties(&props, device) != cudaSuccess) {
            std::cerr << "[topoanns_cuda_device_map] cudaGetDeviceProperties failed for "
                      << device << std::endl;
            return 1;
        }
        char pci_bus_id[32] = {};
        if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device) != cudaSuccess) {
            std::cerr << "[topoanns_cuda_device_map] cudaDeviceGetPCIBusId failed for "
                      << device << std::endl;
            return 1;
        }
        std::cout << "cuda_device=" << device
                  << " name=\"" << props.name << "\""
                  << " pci_bus_id=" << pci_bus_id
                  << " uuid_hex=" << FormatUuid(props.uuid)
                  << std::endl;
    }
    return 0;
}
