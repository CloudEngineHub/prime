#include <torch/extension.h>
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <iostream>

void send_recv(torch::Tensor& tensor, const c10::intrusive_ptr<c10d::ProcessGroup>& pg) {
    std::cout << "Hello from cpp w" ;
    int rank = pg->getRank();

    if (rank == 1) {
        pg->send({tensor}, {0}, 0);
    } else {
        pg->recv({tensor}, {1}, 0);
    }
    
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("send_recv", &send_recv, "Send and receive a tensor between two processes");
}