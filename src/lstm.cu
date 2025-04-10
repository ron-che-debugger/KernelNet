#include "lstm.hpp"

using namespace std;

pair<VarPtr, VarPtr> LSTMCellFunction::apply(const VarPtr &input,
                                             const VarPtr &h_prev,
                                             const VarPtr &c_prev,
                                             const VarPtr &weight_ih,
                                             const VarPtr &weight_hh,
                                             const VarPtr &bias_ih,
                                             const VarPtr &bias_hh,
                                             int input_dim,
                                             int hidden_dim) {
    auto func = make_shared<LSTMCellFunction>();

    // Save inputs and parameters.
    func->saved_input = input;
    func->saved_h_prev = h_prev;
    func->saved_c_prev = c_prev;
    func->saved_weight_ih = weight_ih;
    func->saved_weight_hh = weight_hh;
    func->saved_bias_ih = bias_ih;
    func->saved_bias_hh = bias_hh;
    func->input_dim = input_dim;
    func->hidden_dim = hidden_dim;

    func->inputs.push_back(input);
    func->inputs.push_back(h_prev);
    func->inputs.push_back(c_prev);
    func->inputs.push_back(weight_ih);
    func->inputs.push_back(weight_hh);
    func->inputs.push_back(bias_ih);
    func->inputs.push_back(bias_hh);

    // Determine batch size (assuming input shape: [batch_size, input_dim]).
    int batch_size = input->data.size() / input_dim;
    func->batch_size = batch_size;

    // Compute the linear transformation on the input: x_part = input * weight_ih.
    VarPtr x_part = MatMulFunction::apply(input, weight_ih, batch_size, input_dim, 4 * hidden_dim);
    // Compute h_part = h_prev * weight_hh.
    VarPtr h_part = MatMulFunction::apply(h_prev, weight_hh, batch_size, hidden_dim, 4 * hidden_dim);

    // Sum biases: bias_sum = bias_ih + bias_hh.
    VarPtr bias_sum = AddFunction::apply(bias_ih, bias_hh);

    // Total gate pre-activations: gates = x_part + h_part + bias_sum.
    VarPtr gates = AddFunction::apply(AddFunction::apply(x_part, h_part), bias_sum);

    // Slice the combined gates into four parts:
    // i_gate: [0, hidden_dim), f_gate: [hidden_dim, 2*hidden_dim),
    // g_gate: [2*hidden_dim, 3*hidden_dim), o_gate: [3*hidden_dim, 4*hidden_dim).
    VarPtr i_gate = SliceFunction::apply(gates, batch_size, 0, hidden_dim);
    VarPtr f_gate = SliceFunction::apply(gates, batch_size, hidden_dim, 2 * hidden_dim);
    VarPtr g_gate = SliceFunction::apply(gates, batch_size, 2 * hidden_dim, 3 * hidden_dim);
    VarPtr o_gate = SliceFunction::apply(gates, batch_size, 3 * hidden_dim, 4 * hidden_dim);

    // Save raw gate activations.
    func->saved_i_gate = i_gate;
    func->saved_f_gate = f_gate;
    func->saved_g_gate = g_gate;
    func->saved_o_gate = o_gate;

    // Apply nonlinearities.
    VarPtr i_gate_act = SigmoidFunction::apply(i_gate);
    VarPtr f_gate_act = SigmoidFunction::apply(f_gate);
    VarPtr g_gate_act = TanhFunction::apply(g_gate);
    VarPtr o_gate_act = SigmoidFunction::apply(o_gate);

    // Save activated gate values.
    func->saved_i_gate_act = i_gate_act;
    func->saved_f_gate_act = f_gate_act;
    func->saved_g_gate_act = g_gate_act;
    func->saved_o_gate_act = o_gate_act;

    // Compute new cell state: c_new = f_gate_act * c_prev + i_gate_act * g_gate_act.
    VarPtr f_c = MultiplyFunction::apply(f_gate_act, c_prev);
    VarPtr i_g = MultiplyFunction::apply(i_gate_act, g_gate_act);
    VarPtr c_new = AddFunction::apply(f_c, i_g);
    func->saved_c_new = c_new;

    // Compute new hidden state: h_new = o_gate_act * tanh(c_new).
    VarPtr tanh_c_new = TanhFunction::apply(c_new);
    VarPtr h_new = MultiplyFunction::apply(o_gate_act, tanh_c_new);

    // Set autograd creator for outputs.
    h_new->set_creator(func);
    // c_new->set_creator(func);
    func->output = h_new; // Primary output is h_new.

    return {h_new, c_new};
}

// Order of gradients dependency:
// delta_c(Cell State Gradient) -> <di, df, dg, do>(Gate Gradients) -> dG(Pre-activation[raw] Gradients)
// -> grad_bias<grad_bias_ih, grad_bias_hh>(Bias Gradients) -> <grad_c_prev(Previous Cell Gradient), grad_input(Input Gradient),
// grad_h_prev(Previous Hidden Gradient), grad_weights<grad_weight_ih, grad_weight_hh>(Weight Gradients)>

// Kernel to compute T = tanh(c_new).
__global__ void compute_T_kernel(const float *c_new, float *T, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T[idx] = tanhf(c_new[idx]);
    }
}

// Kernel to compute delta_c = grad_c + grad_h ⊙ o_act ⊙ (1 - T^2).
__global__ void compute_delta_c_kernel(const float *grad_h, const float *grad_c,
                                       const float *o_act, const float *T,
                                       float *delta_c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float one_minus_T2 = 1.0f - T[idx] * T[idx];
        delta_c[idx] = grad_c[idx] + grad_h[idx] * o_act[idx] * one_minus_T2;
    }
}

// Kernel to compute dG (raw gate gradients) for each sample element.
// For each sample b and index j in [0, H):
//   di_raw = (delta_c ⊙ g_act)[b,j] * (i_act[b,j]*(1-i_act[b,j]))
//   df_raw = (delta_c ⊙ c_prev)[b,j] * (f_act[b,j]*(1-f_act[b,j]))
//   dg_raw = (delta_c ⊙ i_act)[b,j] * (1 - g_act[b,j]^2)
//   do_raw = (grad_h ⊙ T)[b,j] * (o_act[b,j]*(1-o_act[b,j]))
// and concatenated into dG at offsets.
__global__ void compute_dG_kernel(const float *delta_c, const float *g_act,
                                  const float *i_act, const float *f_act,
                                  const float *grad_h, const float *T,
                                  const float *o_act, const float *c_prev,
                                  float *dG, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < B * H) {
        int b = idx / H;
        int j = idx % H;

        float di = delta_c[idx] * g_act[idx];
        float di_raw = di * i_act[idx] * (1.0f - i_act[idx]);

        float df = delta_c[idx] * c_prev[idx];
        float df_raw = df * f_act[idx] * (1.0f - f_act[idx]);

        float dg = delta_c[idx] * i_act[idx];
        float dg_raw = dg * (1.0f - g_act[idx] * g_act[idx]);

        float dout = grad_h[idx] * T[idx];
        float do_raw = dout * o_act[idx] * (1.0f - o_act[idx]);

        int base = b * (4 * H);
        dG[base + j] = di_raw;
        dG[base + H + j] = df_raw;
        dG[base + 2 * H + j] = dg_raw;
        dG[base + 3 * H + j] = do_raw;
    }
}

// Kernel to reduce (sum) dG over batch dimension to obtain bias gradients.
// grad_bias[j] = sum_{b=0}^{B-1} dG[b, j] for j in [0, 4H).
__global__ void reduce_sum_dG_kernel(const float *dG, float *grad_bias, int B, int fourH) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < fourH) {
        float sum = 0.0f;
        for (int b = 0; b < B; b++) {
            sum += dG[b * fourH + j];
        }
        grad_bias[j] = sum;
    }
}

// Kernel to compute grad_c_prev = delta_c ⊙ f_act.
__global__ void compute_grad_c_prev_kernel(const float *delta_c, const float *f_act,
                                           float *grad_c_prev, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_c_prev[idx] = delta_c[idx] * f_act[idx];
    }
}

// Kernel for splitting grad_output when only h_new gradient is provided.
// It copies the first half into grad_h and sets grad_c to zero.
__global__ void split_grad_kernel_single(const float *grad_out, float *grad_h, float *grad_c, int half_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_size) {
        grad_h[idx] = grad_out[idx];
        grad_c[idx] = 0.0f;
    }
}

// Kernel for splitting grad_output when both h_new and c_new gradients are provided.
// It copies the first half into grad_h and the second half (starting at index half_size)
// into grad_c.
__global__ void split_grad_kernel_double(const float *grad_out, float *grad_h, float *grad_c, int half_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_size) {
        grad_h[idx] = grad_out[idx];
        grad_c[idx] = grad_out[half_size + idx];
    }
}

vector<Tensor> LSTMCellFunction::backward(const Tensor &grad_output) {
    // grad_output has shape (B, 2H) where the first B*H elements are grad_h and
    // the next B*H are grad_c.
    bool use_gpu = (grad_output.device() == CUDA);
    int B = batch_size;
    int H = hidden_dim;
    int D = input_dim;

    // Determine if we only have h_new's gradient (size == B*H)
    bool single_output = (grad_output.size() == B * H);

    // Allocate tensors for grad_h and grad_c on the proper device.
    Tensor grad_h(B * H, grad_output.device());
    Tensor grad_c(B * H, grad_output.device());

    if (use_gpu) {
        // When using CUDA, we cannot iterate directly over device memory from host.
        int half_size = B * H;
        dim3 blockSize(256);
        dim3 gridSize((half_size + blockSize.x - 1) / blockSize.x);
        if (single_output) {
            // Only h_new's gradient is provided—set grad_c to zero.
            split_grad_kernel_single<<<gridSize, blockSize>>>(grad_output.data(), grad_h.data(), grad_c.data(), half_size);
        } else {
            // Both gradients are provided.
            split_grad_kernel_double<<<gridSize, blockSize>>>(grad_output.data(), grad_h.data(), grad_c.data(), half_size);
        }
        cudaDeviceSynchronize();
    } else {
        // CPU branch: iterate over host memory.
        if (single_output) {
            for (int i = 0; i < B * H; i++) {
                grad_h.data()[i] = grad_output.data()[i];
                grad_c.data()[i] = 0.0f;
            }
        } else {
            const float *src = grad_output.data();
            for (int i = 0; i < B * H; i++) {
                grad_h.data()[i] = src[i];
                grad_c.data()[i] = src[B * H + i];
            }
        }
    }

    // Allocate temporary tensors on the same device.
    Tensor T_tensor(B * H, grad_output.device()); // T = tanh(c_new)
    Tensor delta_c(B * H, grad_output.device());  // effective cell gradient
    Tensor dG(B * 4 * H, grad_output.device());   // concatenated raw gate gradients

    dim3 blockSize(256);
    dim3 gridSize((B * H + blockSize.x - 1) / blockSize.x);

    if (use_gpu) {
        // GPU branch: launch kernels. Use data() from each Tensor.
        compute_T_kernel<<<gridSize, blockSize>>>(saved_c_new->data.data(), T_tensor.data(), B * H);
        cudaDeviceSynchronize();
        compute_delta_c_kernel<<<gridSize, blockSize>>>(grad_h.data(), grad_c.data(),
                                                        saved_o_gate_act->data.data(), T_tensor.data(),
                                                        delta_c.data(), B * H);
        cudaDeviceSynchronize();
        // blocks = (B * H + threads - 1) / threads;
        compute_dG_kernel<<<gridSize, blockSize>>>(delta_c.data(), saved_g_gate_act->data.data(),
                                                   saved_i_gate_act->data.data(), saved_f_gate_act->data.data(),
                                                   grad_h.data(), T_tensor.data(),
                                                   saved_o_gate_act->data.data(), saved_c_prev->data.data(),
                                                   dG.data(), H, B);
        cudaDeviceSynchronize();
    } else {
        // CPU branch.
        for (int i = 0; i < B * H; i++) {
            T_tensor.data()[i] = tanh(saved_c_new->data.data()[i]);
        }
        for (int i = 0; i < B * H; i++) {
            float one_minus_T2 = 1.0f - T_tensor.data()[i] * T_tensor.data()[i];
            delta_c.data()[i] = grad_c.data()[i] + grad_h.data()[i] * saved_o_gate_act->data.data()[i] * one_minus_T2;
        }
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < H; j++) {
                int idx = b * H + j;
                float di = delta_c.data()[idx] * saved_g_gate_act->data.data()[idx];
                float di_raw = di * saved_i_gate_act->data.data()[idx] * (1.0f - saved_i_gate_act->data.data()[idx]);
                float df = delta_c.data()[idx] * saved_c_prev->data.data()[idx];
                float df_raw = df * saved_f_gate_act->data.data()[idx] * (1.0f - saved_f_gate_act->data.data()[idx]);
                float dg = delta_c.data()[idx] * saved_i_gate_act->data.data()[idx];
                float dg_raw = dg * (1.0f - saved_g_gate_act->data.data()[idx] * saved_g_gate_act->data.data()[idx]);
                float dout = grad_h.data()[idx] * T_tensor.data()[idx];
                float do_raw = dout * saved_o_gate_act->data.data()[idx] * (1.0f - saved_o_gate_act->data.data()[idx]);
                int base = b * (4 * H);
                dG.data()[base + j] = di_raw;
                dG.data()[base + H + j] = df_raw;
                dG.data()[base + 2 * H + j] = dg_raw;
                dG.data()[base + 3 * H + j] = do_raw;
            }
        }
    }

    // Compute bias gradients by reducing dG over the batch dimension.
    Tensor grad_bias_sum(4 * H, grad_output.device());
    gridSize = dim3((4 * H + blockSize.x - 1) / blockSize.x);
    if (use_gpu) {
        reduce_sum_dG_kernel<<<gridSize, blockSize>>>(dG.data(), grad_bias_sum.data(), B, 4 * H);
        cudaDeviceSynchronize();
    } else {
        vector<float> bias_sum_host(4 * H, 0.0f);
        for (int b = 0; b < B; b++) {
            for (int j = 0; j < 4 * H; j++) {
                bias_sum_host[j] += dG.data()[b * (4 * H) + j];
            }
        }
        memcpy(grad_bias_sum.data(), bias_sum_host.data(), 4 * H * sizeof(float));
    }

    // Split bias gradients equally.
    vector<float> bias_grad_host(4 * H, 0.0f);
    {
        vector<float> temp(4 * H, 0.0f);
        if (use_gpu) {
            cudaMemcpy(temp.data(), grad_bias_sum.data(), 4 * H * sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            for (int j = 0; j < 4 * H; j++) {
                temp[j] = grad_bias_sum.data()[j];
            }
        }
        for (int j = 0; j < 4 * H; j++) {
            bias_grad_host[j] = 0.5f * temp[j];
        }
    }
    Tensor grad_bias_ih(4 * H, grad_output.device());
    Tensor grad_bias_hh(4 * H, grad_output.device());
    if (use_gpu) {
        cudaMemcpy(grad_bias_ih.data(), bias_grad_host.data(), 4 * H * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(grad_bias_hh.data(), bias_grad_host.data(), 4 * H * sizeof(float), cudaMemcpyHostToDevice);
    } else {
        for (int j = 0; j < 4 * H; j++) {
            grad_bias_ih.data()[j] = bias_grad_host[j];
            grad_bias_hh.data()[j] = bias_grad_host[j];
        }
    }

    // Compute grad_c_prev = delta_c ⊙ f_act.
    Tensor grad_c_prev(B * H, grad_output.device());
    gridSize = dim3((B * H + blockSize.x - 1) / blockSize.x);
    if (use_gpu) {
        compute_grad_c_prev_kernel<<<gridSize, blockSize>>>(delta_c.data(), saved_f_gate_act->data.data(),
                                                            grad_c_prev.data(), B * H);
        cudaDeviceSynchronize();
    } else {
        for (int i = 0; i < B * H; i++) {
            grad_c_prev.data()[i] = delta_c.data()[i] * saved_f_gate_act->data.data()[i];
        }
    }

    // Compute linear gradients using Tensor functions.
    Tensor trans_Wih = Tensor::transpose(saved_weight_ih->data, D, 4 * H);
    Tensor grad_input = Tensor::matmul(dG, trans_Wih, B, 4 * H, D);

    Tensor trans_input = Tensor::transpose(saved_input->data, B, D);
    Tensor grad_weight_ih = Tensor::matmul(trans_input, dG, D, B, 4 * H);

    Tensor trans_Whh = Tensor::transpose(saved_weight_hh->data, H, 4 * H);
    Tensor grad_h_prev = Tensor::matmul(dG, trans_Whh, B, 4 * H, H);

    Tensor trans_h_prev = Tensor::transpose(saved_h_prev->data, B, H);
    Tensor grad_weight_hh = Tensor::matmul(trans_h_prev, dG, H, B, 4 * H);

    vector<Tensor> grads;
    grads.push_back(grad_input);     // grad_input
    grads.push_back(grad_h_prev);    // grad_h_prev
    grads.push_back(grad_c_prev);    // grad_c_prev
    grads.push_back(grad_weight_ih); // grad_weight_ih
    grads.push_back(grad_weight_hh); // grad_weight_hh
    grads.push_back(grad_bias_ih);   // grad_bias_ih
    grads.push_back(grad_bias_hh);   // grad_bias_hh
    return grads;
}

static void init_tensor(Tensor &t, float limit) {
    size_t size = t.size();
    for (size_t i = 0; i < size; i++) {
        t.data()[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * limit;
    }
}

LSTMCell::LSTMCell(int input_dim, int hidden_dim, Device device)
    : input_dim(input_dim), hidden_dim(hidden_dim), device(device) {
    int gate_size = 4 * hidden_dim;

    // Create weight_ih on CPU first.
    Tensor w_ih(input_dim * gate_size, CPU);
    float limit_ih = sqrt(6.0f / (input_dim + gate_size));
    init_tensor(w_ih, limit_ih);
    if (device == CUDA)
        w_ih.toCUDA();
    weight_ih = make_shared<Variable>(w_ih, true, "weight_ih");

    // Create weight_hh on CPU first.
    Tensor w_hh(hidden_dim * gate_size, CPU);
    float limit_hh = sqrt(6.0f / (hidden_dim + gate_size));
    init_tensor(w_hh, limit_hh);
    if (device == CUDA)
        w_hh.toCUDA();
    weight_hh = make_shared<Variable>(w_hh, true, "weight_hh");

    // Create bias_ih on CPU and convert if needed.
    Tensor b_ih(gate_size, CPU);
    b_ih.fill(0.0f);
    if (device == CUDA)
        b_ih.toCUDA();
    bias_ih = make_shared<Variable>(b_ih, true, "bias_ih");

    // Create bias_hh on CPU and convert if needed.
    Tensor b_hh(gate_size, CPU);
    b_hh.fill(0.0f);
    if (device == CUDA)
        b_hh.toCUDA();
    bias_hh = make_shared<Variable>(b_hh, true, "bias_hh");
}

vector<VarPtr> LSTMCell::forward(const vector<VarPtr> &inputs) {
    // Make sure exactly three inputs are provided.
    assert(inputs.size() == 3 && "LSTMCell expects exactly three inputs: {input, h_prev, c_prev}");

    // Use our LSTMCellFunction::apply method. This function returns a pair {h_new, c_new}.
    auto outputs = LSTMCellFunction::apply(inputs[0], inputs[1], inputs[2],
                                           weight_ih, weight_hh,
                                           bias_ih, bias_hh,
                                           input_dim, hidden_dim);
    // Return both outputs as a vector.
    return {outputs.first, outputs.second};
}

vector<VarPtr> LSTMCell::parameters() {
    return {weight_ih, weight_hh, bias_ih, bias_hh};
}