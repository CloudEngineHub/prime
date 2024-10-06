#include <torch/torch.h>
#include <SocketCommunicator.cpp>

namespace py = pybind11;

// PyBind11 module
PYBIND11_MODULE(communicator, m) {
    py::class_<SocketCommunicator>(m, "SocketCommunicator")
        .def(py::init<const std::string &, unsigned short>());
}