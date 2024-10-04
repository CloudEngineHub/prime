#include <torch/torch.h>

namespace py = pybind11;

constexpr int n_bins = 256;  // 8-bit quantization
constexpr double RANGE_IN_SIGMAS = 6.0;
const int max_num_threads = std::thread::hardware_concurrency();

torch::Tensor quantize_per_tensor_multithreaded(const torch::Tensor& tensor, float scale, int32_t zero_point, int num_threads) {
    torch::TensorOptions options = tensor.options().dtype(torch::kByte);
    torch::Tensor quantized_tensor = torch::empty_like(tensor, options);
    
    float* tensor_data = tensor.data_ptr<float>();
    uint8_t* quant_data = quantized_tensor.data_ptr<uint8_t>();
    int64_t numel = tensor.numel();
    float inv_scale = 1.0f / scale;

    std::vector<std::thread> threads;
    int64_t chunk_size = numel / num_threads;

    auto quantize_chunk = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            int32_t quant_val = static_cast<int32_t>(std::round(tensor_data[i] * inv_scale)) + zero_point;
            quant_data[i] = static_cast<uint8_t>(std::clamp(quant_val, 0, 255));
        }
    };

    for (int i = 0; i < num_threads - 1; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = (i + 1) * chunk_size;
        threads.emplace_back(quantize_chunk, start, end);
    }

    // Handle the last chunk (which may be slightly larger due to rounding)
    threads.emplace_back(quantize_chunk, (num_threads - 1) * chunk_size, numel);

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    return quantized_tensor;
}

torch::Tensor average_buckets_multithread(const torch::Tensor& tensor, const torch::Tensor& quant_weight, int64_t n_bins, int num_threads) {
    torch::NoGradGuard no_grad;
    auto flat_tensor = tensor.flatten().contiguous();
    auto flat_quant_weight = quant_weight.flatten().contiguous();
    auto options = flat_tensor.options();
    auto bin_sums = torch::zeros({n_bins}, options);
    auto bin_counts = torch::zeros({n_bins}, options.dtype(torch::kLong));

    // Get raw pointers
    float* tensor_data = flat_tensor.data_ptr<float>();
    uint8_t* quant_data = flat_quant_weight.data_ptr<uint8_t>();
    float* sums_data = bin_sums.data_ptr<float>();
    int64_t* counts_data = bin_counts.data_ptr<int64_t>();
    int64_t numel = flat_tensor.numel();

    // Create a vector to hold our threads
    std::vector<std::thread> threads;

    // Lambda function for the work each thread will do
    auto worker = [&](int64_t start, int64_t end) {
        std::vector<float> local_sums(n_bins, 0.0f);
        std::vector<int64_t> local_counts(n_bins, 0);

        for (int64_t i = start; i < end; ++i) {
            uint8_t bin = quant_data[i];
            if (bin < n_bins) {  // No need to check for >= 0 as uint8_t is always non-negative
                local_sums[bin] += tensor_data[i];
                local_counts[bin]++;
            }
        }

        // Use a mutex to safely update the shared data
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        for (int64_t i = 0; i < n_bins; ++i) {
            sums_data[i] += local_sums[i];
            counts_data[i] += local_counts[i];
        }
    };

    // Divide the work among threads
    int64_t chunk_size = numel / num_threads;
    for (unsigned int i = 0; i < num_threads; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = (i == num_threads - 1) ? numel : (i + 1) * chunk_size;
        threads.emplace_back(worker, start, end);
    }

    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }

    // Compute averages
    for (int64_t i = 0; i < n_bins; ++i) {
        sums_data[i] = counts_data[i] > 0 ? sums_data[i] / counts_data[i] : 0.0f;
    }

    return bin_sums;
}

std::tuple<torch::Tensor, torch::Tensor> uniform_8bit_quantize(torch::Tensor tensor, bool inplace) {
    int offset = n_bins / 2;
    
    // Centered tensor handling (currently commented out, so no centering)
    torch::Tensor centered_tensor = tensor;

    // Calculate unbiased standard deviation
    double std_unbiased = centered_tensor.norm().item<double>() / std::sqrt(centered_tensor.numel() - 1);

    // Calculate scale for quantization
    double scale = RANGE_IN_SIGMAS * std_unbiased / n_bins;

    // Perform quantization
    torch::Tensor quantized_tensor = quantize_per_tensor_multithreaded(centered_tensor, scale, offset, max_num_threads);

    // Call average_buckets to create the lookup table
    torch::Tensor lookup = average_buckets_multithread(tensor, quantized_tensor, n_bins, max_num_threads);

    return std::make_tuple(quantized_tensor, lookup);
}


// PyBind11 module
PYBIND11_MODULE(compression, m) {
    m.def(
        "average_buckets",
        &average_buckets_multithread,
        "Average buckets for quantized values",
        py::arg("tensor"),
        py::arg("quant_weight"),
        py::arg("n_bins"),
        py::arg("num_threads") = max_num_threads
    )
    .def(
        "uniform_8bit_quantize",
        &uniform_8bit_quantize,
        "Uniform 8-bit quantization function",
        py::arg("tensor"),
        py::arg("inplace") = true
    )
    .def(
        "quantize_per_tensor_uint8",
        &quantize_per_tensor_multithreaded,
        "Faster torch::quantize_per_tensor",
        py::arg("tensor"),
        py::arg("scale"),
        py::arg("zero_point"),
        py::arg("num_threads") = max_num_threads
    );
}
