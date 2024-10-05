#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <compression.cpp>

constexpr int BUFFER_COUNT = 2;

void ring_allreduce(
    torch::Tensor& tensor,
    c10d::ReduceOp op,
    c10d::ProcessGroup* group
) {
    TORCH_CHECK(group != nullptr, "Group must be provided");
    TORCH_CHECK(op == c10d::ReduceOp::SUM || op == c10d::ReduceOp::AVG, "Unsupported reduce operation. Only SUM and AVG are supported.");

    int world_size = group->getSize();
    int rank = group->getRank();

    // Divide the tensor into chunks
    auto flat_tensor = tensor.view({tensor.numel()});
    std::vector<torch::Tensor> chunks = flat_tensor.chunk(world_size * BUFFER_COUNT);

    TORCH_CHECK(flat_tensor.size(0) % (world_size * BUFFER_COUNT) == 0, "Tensor size must be divisible by world size");

    // Temporary buffers for transferring data
    int num_buffers = BUFFER_COUNT * world_size;
    std::vector<torch::Tensor> recv_buffer;
    std::vector<torch::Tensor> send_buffer;
    std::vector<torch::Tensor> send_lookup_buffer;
    std::vector<torch::Tensor> recv_lookup_buffer;
    std::vector<c10::intrusive_ptr<c10d::Work>> send_lookup_work(BUFFER_COUNT);
    std::vector<c10::intrusive_ptr<c10d::Work>> recv_lookup_work(BUFFER_COUNT);
    std::vector<c10::intrusive_ptr<c10d::Work>> send_work(BUFFER_COUNT);
    std::vector<c10::intrusive_ptr<c10d::Work>> recv_work(BUFFER_COUNT);

    for (int i = 0; i < BUFFER_COUNT; ++i) {
        recv_buffer.push_back(torch::empty_like(chunks[0], torch::kUInt8));
        send_buffer.push_back(torch::Tensor());
        send_lookup_buffer.push_back(torch::Tensor());
        recv_lookup_buffer.push_back(torch::empty({256}, chunks[0].options()));
    }

    // Send and receive ranks
    int send_rank = (rank + 1) % world_size;
    int recv_rank = (rank - 1 + world_size) % world_size;

    // Reduce-scatter loop
    for (int step = 1; step <= world_size * BUFFER_COUNT; ++step) {
        int send_chunk = (rank * BUFFER_COUNT - step + num_buffers) % num_buffers;

        if (send_work[step % BUFFER_COUNT]) {
            send_work[step % BUFFER_COUNT]->wait();
            recv_work[step % BUFFER_COUNT]->wait();
            send_lookup_work[step % BUFFER_COUNT]->wait();
            recv_lookup_work[step % BUFFER_COUNT]->wait();
            chunks[send_chunk].add_(recv_lookup_buffer[step % BUFFER_COUNT].index({recv_buffer[step % BUFFER_COUNT].to(torch::kLong)}));
        }

        if (step <= (world_size - 1) * BUFFER_COUNT) {
            // Quantize and send
            std::tie(send_buffer[step % BUFFER_COUNT], send_lookup_buffer[step % BUFFER_COUNT]) = uniform_8bit_quantize(chunks[send_chunk], false);

            std::vector<torch::Tensor> send_tensors = {send_lookup_buffer[step % BUFFER_COUNT]};
            send_lookup_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step + 1000);

            std::vector<torch::Tensor> recv_tensors = {recv_lookup_buffer[step % BUFFER_COUNT]};
            recv_lookup_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step + 1000);

            send_tensors = {send_buffer[step % BUFFER_COUNT]};
            send_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step);

            recv_tensors = {recv_buffer[step % BUFFER_COUNT]};
            recv_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step);
        }
    }

    if (op == c10d::ReduceOp::AVG) {
        for (int i = 0; i < BUFFER_COUNT; ++i) {
            chunks[i + rank * BUFFER_COUNT].div_(world_size);
        }
    }
    
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        std::tie(send_buffer[0], send_lookup_buffer[0]) = uniform_8bit_quantize(chunks[i + rank * BUFFER_COUNT], true);
        chunks[i + rank * BUFFER_COUNT].copy_(send_lookup_buffer[0].index({send_buffer[0].to(torch::kLong)}));
    }

    // Reset buffers for the second phase
    recv_buffer.clear();
    send_buffer.clear();
    send_lookup_buffer.clear();
    recv_lookup_buffer.clear();
    for (int i = 0; i < BUFFER_COUNT; ++i) {
        recv_buffer.push_back(torch::empty_like(chunks[0], torch::kUInt8));
        send_buffer.push_back(torch::Tensor());
        send_lookup_buffer.push_back(torch::Tensor());
        recv_lookup_buffer.push_back(torch::empty({256}, chunks[0].options()));
    }
    std::fill(send_work.begin(), send_work.end(), nullptr);
    std::fill(recv_work.begin(), recv_work.end(), nullptr);
    std::fill(send_lookup_work.begin(), send_lookup_work.end(), nullptr);
    std::fill(recv_lookup_work.begin(), recv_lookup_work.end(), nullptr);

    for (int step = 1; step <= world_size * BUFFER_COUNT; ++step) {
        int send_chunk = (rank * BUFFER_COUNT + BUFFER_COUNT - step + num_buffers) % num_buffers;

        if (send_work[step % BUFFER_COUNT]) {
            send_work[step % BUFFER_COUNT]->wait();
            recv_work[step % BUFFER_COUNT]->wait();
            send_lookup_work[step % BUFFER_COUNT]->wait();
            recv_lookup_work[step % BUFFER_COUNT]->wait();
            auto a = recv_lookup_buffer[step % BUFFER_COUNT].index({recv_buffer[step % BUFFER_COUNT].to(torch::kLong)});
            chunks[send_chunk].copy_(a);
            //chunks[send_chunk].copy_(recv_lookup_buffer[step % BUFFER_COUNT].index({recv_buffer[step % BUFFER_COUNT].to(torch::kLong)}).to(tensor.dtype()));
        }

        if (step <= (world_size - 1) * BUFFER_COUNT) {
            // Quantize and send
            std::tie(send_buffer[step % BUFFER_COUNT], send_lookup_buffer[step % BUFFER_COUNT]) = uniform_8bit_quantize(chunks[send_chunk], false);

            std::vector<torch::Tensor> send_tensors = {send_lookup_buffer[step % BUFFER_COUNT]};
            send_lookup_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step + 1000);

            std::vector<torch::Tensor> recv_tensors = {recv_lookup_buffer[step % BUFFER_COUNT]};
            recv_lookup_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step + 1000);

            send_tensors = {send_buffer[step % BUFFER_COUNT]};
            send_work[step % BUFFER_COUNT] = group->send(send_tensors, send_rank, step);

            recv_tensors = {recv_buffer[step % BUFFER_COUNT]};
            recv_work[step % BUFFER_COUNT] = group->recv(recv_tensors, recv_rank, step);
        }
    }
}

PYBIND11_MODULE(collectives, m) {
    m.def(
        "ring_allreduce",
        &ring_allreduce,
        "Ring allreduce implementation",
        py::arg("tensor"),
        py::arg("op"),
        py::arg("pg")
    )
}