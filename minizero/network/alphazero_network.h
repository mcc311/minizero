#pragma once

#include "network.h"
#include "utils.h"
#include <algorithm>
#include <execution>
#include <memory>
#include <mutex>
#include <string>
#include <torch/cuda.h>
#include <vector>

namespace minizero::network {

class AlphaZeroNetworkOutput : public NetworkOutput {
public:
    float value_;
    std::vector<float> policy_;
    std::vector<float> policy_logits_;

    AlphaZeroNetworkOutput(int policy_size)
    {
        value_ = 0.0f;
        policy_.resize(policy_size, 0.0f);
        policy_logits_.resize(policy_size, 0.0f);
    }
};

class AlphaZeroNetwork : public Network {
public:
    AlphaZeroNetwork()
    {
        clear();
        device_count = torch::cuda::device_count();
        networks_.reserve(device_count);
    }

    void loadModel(const std::string& nn_file_name, const int gpu_id) override
    {
        assert(batch_size_ == 0); // should avoid loading model when batch size is not 0
        // Network::loadModel(nn_file_name, gpu_id);
        network_file_name_ = nn_file_name;
        Network::loadModel(nn_file_name, -1);
        for (int i = 0; i < static_cast<int>(device_count); ++i) {
            networks_.emplace_back();
            networks_.back().loadModel(nn_file_name, i);

            std::cout << "Loaded model: " << nn_file_name << " on GPU: " << i << std::endl;
            // network hyper-parameter
        }
        clear();
        std::cout << toString() << std::endl;
    }

    std::string toString() const override
    {
        std::ostringstream oss;
        oss << Network::toString();
        return oss.str();
    }

    int pushBack(std::vector<float> features)
    {
        assert(static_cast<int>(features.size()) == getNumInputChannels() * getInputChannelHeight() * getInputChannelWidth());
        assert(batch_size_ < kReserved_batch_size);

        int index;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            index = batch_size_++;
            tensor_input_.resize(batch_size_);
        }
        tensor_input_[index] = torch::from_blob(features.data(), {1, getNumInputChannels(), getInputChannelHeight(), getInputChannelWidth()}).clone();
        return index;
    }

    std::vector<std::shared_ptr<NetworkOutput>> forward()
    {
        assert(batch_size_ > 0);
        std::cout << "batch_size_: " << batch_size_ << std::endl;
        // split tensor_input_ into chunks
        std::vector<torch::Tensor> inputs_split = torch::cat(tensor_input_).chunk(device_count);
        size_t num_chunks = inputs_split.size();

        std::vector<torch::jit::IValue> inputs;
        inputs.resize(num_chunks);
        std::for_each(
            std::execution::par,
            std::begin(inputs_split),
            std::end(inputs_split),
            [&inputs, this, &inputs_split](torch::Tensor& input) {
                auto index = &input - &inputs_split[0];
                // inputs.emplace_back(input.to(torch::Device(torch::kCUDA, index)));
                inputs[index] = input.to(torch::Device(torch::kCUDA, index));
            });

        std::vector<torch::jit::IValue> forward_results;
        forward_results.resize(num_chunks);
        std::for_each(
            std::execution::par,
            std::begin(inputs),
            std::end(inputs),
            [&forward_results, this, &inputs](torch::jit::IValue& input) {
                auto index = &input - &inputs[0];
                forward_results[index] = networks_[index].forward({input});
            });
        // std::transform(
        //     std::execution::par,
        //     std::begin(inputs),
        //     std::end(inputs),
        //     std::begin(forward_results),
        //     [this, &inputs](const torch::jit::IValue& input) -> torch::jit::IValue {
        //         auto index = &input - &inputs[0];
        //         return networks_[index].forward({input});
        //     }
        // );
        std::cout << "forward_results.size(): " << forward_results.size() << std::endl;
        std::vector<torch::Tensor> policy_outputs;
        policy_outputs.reserve(device_count);
        std::vector<torch::Tensor> policy_logits_outputs;
        policy_logits_outputs.reserve(device_count);
        std::vector<torch::Tensor> value_outputs;
        value_outputs.reserve(device_count);
        for (size_t i = 0; i < forward_results.size(); ++i) {
            auto forward_result = forward_results[i].toGenericDict();
            policy_outputs.emplace_back(forward_result.at("policy").toTensor().to(at::kCPU));
            policy_logits_outputs.emplace_back(forward_result.at("policy_logit").toTensor().to(at::kCPU));
            value_outputs.emplace_back(forward_result.at("value").toTensor().to(at::kCPU));
        }
        auto policy_output = torch::cat(policy_outputs);
        auto policy_logits_output = torch::cat(policy_logits_outputs);
        auto value_output = torch::cat(value_outputs);
        assert(policy_output.numel() == batch_size_ * getActionSize());
        assert(policy_logits_output.numel() == batch_size_ * getActionSize());
        assert(value_output.numel() == batch_size_ * getDiscreteValueSize());

        const int policy_size = getActionSize();
        std::vector<std::shared_ptr<NetworkOutput>> network_outputs;
        for (int i = 0; i < batch_size_; ++i) {
            network_outputs.emplace_back(std::make_shared<AlphaZeroNetworkOutput>(policy_size));
            auto alphazero_network_output = std::static_pointer_cast<AlphaZeroNetworkOutput>(network_outputs.back());

            // policy & policy logits
            std::copy(policy_output.data_ptr<float>() + i * policy_size,
                      policy_output.data_ptr<float>() + (i + 1) * policy_size,
                      alphazero_network_output->policy_.begin());
            std::copy(policy_logits_output.data_ptr<float>() + i * policy_size,
                      policy_logits_output.data_ptr<float>() + (i + 1) * policy_size,
                      alphazero_network_output->policy_logits_.begin());

            // value
            if (getDiscreteValueSize() == 1) {
                alphazero_network_output->value_ = value_output[i].item<float>();
            } else {
                int start_value = -getDiscreteValueSize() / 2;
                alphazero_network_output->value_ = std::accumulate(value_output.data_ptr<float>() + i * getDiscreteValueSize(),
                                                                   value_output.data_ptr<float>() + (i + 1) * getDiscreteValueSize(),
                                                                   0.0f,
                                                                   [&start_value](const float& sum, const float& value) { return sum + value * start_value++; });
                alphazero_network_output->value_ = utils::invertValue(alphazero_network_output->value_);
            }
        }

        clear();
        return network_outputs;
    }

    inline int getBatchSize() const { return batch_size_; }

private:
    inline void clear()
    {
        batch_size_ = 0;
        tensor_input_.clear();
        tensor_input_.reserve(kReserved_batch_size);
    }

    int batch_size_;
    std::mutex mutex_;
    std::vector<torch::Tensor> tensor_input_;
    std::vector<Network> networks_;

    const int kReserved_batch_size = 4096;
    size_t device_count;
};

} // namespace minizero::network
