#include "lstm.hpp"

namespace kernelnet {
namespace nn {
/**
 * @brief Performs the forward pass of the LSTM cell.
 *
 * Calculates the new hidden and cell states given the current input, previous hidden state,
 * previous cell state, and the corresponding weights and biases. The operation includes a linear
 * transformation followed by slicing into four gates and applying nonlinearities.
 *
 * @param input Input variable at the current time step (shape: [batch_size, input_dim]).
 * @param h_prev Previous hidden state (shape: [batch_size, hidden_dim]).
 * @param c_prev Previous cell state (shape: [batch_size, hidden_dim]).
 * @param weight_ih Input-to-hidden weight variable (shape: [input_dim, 4 * hidden_dim]).
 * @param weight_hh Hidden-to-hidden weight variable (shape: [hidden_dim, 4 * hidden_dim]).
 * @param bias_ih Input-to-hidden bias variable (shape: [4 * hidden_dim]).
 * @param bias_hh Hidden-to-hidden bias variable (shape: [4 * hidden_dim]).
 * @param input_dim Dimensionality of the input.
 * @param hidden_dim Dimensionality of the hidden state.
 * @return A pair {h_new, c_new} containing the new hidden and cell states.
 */
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

    // Determine batch size (assumes input shape: [batch_size, input_dim]).
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
    // c_new->set_creator(func);  // Only h_new is used as primary output.
    func->output = h_new;

    return {h_new, c_new};
}

/* Order of gradients dependency:
   delta_c(Cell State Gradient) -> <di, df, dg, do>(Gate Gradients) -> dG (Pre-activation Gradients)
   -> grad_bias <grad_bias_ih, grad_bias_hh>(Bias Gradients) ->
      <grad_c_prev, grad_input, grad_h_prev, grad_weights <grad_weight_ih, grad_weight_hh>>
*/

/**
 * @brief CUDA kernel to compute T = tanh(c_new) element-wise.
 *
 * @param c_new Pointer to the new cell state array.
 * @param T Pointer to the output array for tanh(c_new).
 * @param size Number of elements (B * H).
 */
__global__ void compute_T_kernel(const float *c_new, float *T, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        T[idx] = tanhf(c_new[idx]);
    }
}

/**
 * @brief CUDA kernel to compute the effective cell gradient: delta_c.
 *
 * Calculates delta_c = grad_c + grad_h ⊙ o_gate_act ⊙ (1 - T^2).
 *
 * @param grad_h Pointer to the gradient from h_new.
 * @param grad_c Pointer to the gradient from c_new.
 * @param o_act Pointer to the activated output gate.
 * @param T Pointer to tanh(c_new).
 * @param delta_c Pointer to the output delta_c array.
 * @param size Number of elements (B * H).
 */
__global__ void compute_delta_c_kernel(const float *grad_h, const float *grad_c,
                                       const float *o_act, const float *T,
                                       float *delta_c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float one_minus_T2 = 1.0f - T[idx] * T[idx];
        delta_c[idx] = grad_c[idx] + grad_h[idx] * o_act[idx] * one_minus_T2;
    }
}

/**
 * @brief CUDA kernel to compute raw gate gradients (dG) for LSTM cell.
 *
 * For each sample and each hidden state index j:
 *   - di_raw = (delta_c ⊙ g_gate_act)[b,j] * (i_gate_act[b,j]*(1-i_gate_act[b,j]))
 *   - df_raw = (delta_c ⊙ c_prev)[b,j] * (f_gate_act[b,j]*(1-f_gate_act[b,j])
 *   - dg_raw = (delta_c ⊙ i_gate_act)[b,j] * (1 - g_gate_act[b,j]^2)
 *   - do_raw = (grad_h ⊙ tanh(c_new))[b,j] * (o_gate_act[b,j]*(1-o_gate_act[b,j])
 * The gradients are concatenated into dG.
 *
 * @param delta_c Pointer to the effective cell gradients.
 * @param g_act Pointer to activated candidate gate (g_gate_act).
 * @param i_act Pointer to activated input gate (i_gate_act).
 * @param f_act Pointer to activated forget gate (f_gate_act).
 * @param grad_h Pointer to the gradient from h_new.
 * @param T Pointer to tanh(c_new).
 * @param o_act Pointer to activated output gate (o_gate_act).
 * @param c_prev Pointer to the previous cell state.
 * @param dG Pointer to the output pre-activation gradient array.
 * @param H Hidden dimension.
 * @param B Batch size.
 */
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

/**
 * @brief CUDA kernel to reduce (sum) the pre-activation gradients dG over the batch dimension,
 *        to compute bias gradients.
 *
 * For each gate index j in [0, 4H), computes:
 *    grad_bias[j] = sum_{b=0}^{B-1} dG[b,j]
 *
 * @param dG Pointer to the concatenated pre-activation gradients (shape: [B, 4H]).
 * @param grad_bias Pointer to the output bias gradient array (length: 4H).
 * @param B Batch size.
 * @param fourH Total number of gate elements per sample (4 * H).
 */
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

/**
 * @brief CUDA kernel to compute the gradient with respect to the previous cell state.
 *
 * Computes grad_c_prev = delta_c ⊙ f_gate_act.
 *
 * @param delta_c Pointer to the effective cell gradients (delta_c).
 * @param f_act Pointer to the activated forget gate (f_gate_act).
 * @param grad_c_prev Pointer to the output previous cell gradient array.
 * @param size Total number of elements (B * H).
 */
__global__ void compute_grad_c_prev_kernel(const float *delta_c, const float *f_act,
                                           float *grad_c_prev, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_c_prev[idx] = delta_c[idx] * f_act[idx];
    }
}

/**
 * @brief CUDA kernel for splitting grad_output when only h_new's gradient is provided.
 *
 * Copies the first half of grad_output into grad_h and sets grad_c to zero.
 *
 * @param grad_out Pointer to the input gradient array (size: B*H).
 * @param grad_h Pointer to the output gradient for h_new.
 * @param grad_c Pointer to the output gradient for c_new (set to zero).
 * @param half_size Half of the total number of hidden elements (B * H).
 */
__global__ void split_grad_kernel_single(const float *grad_out, float *grad_h, float *grad_c, int half_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_size) {
        grad_h[idx] = grad_out[idx];
        grad_c[idx] = 0.0f;
    }
}

/**
 * @brief CUDA kernel for splitting grad_output when both h_new and c_new gradients are provided.
 *
 * Splits grad_output into grad_h (first half) and grad_c (second half).
 *
 * @param grad_out Pointer to the input gradient array (size: 2 * half_size).
 * @param grad_h Pointer to the output gradient for h_new.
 * @param grad_c Pointer to the output gradient for c_new.
 * @param half_size Half the total number of hidden elements (B * H).
 */
__global__ void split_grad_kernel_double(const float *grad_out, float *grad_h, float *grad_c, int half_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < half_size) {
        grad_h[idx] = grad_out[idx];
        grad_c[idx] = grad_out[half_size + idx];
    }
}

/**
 * @brief Computes the backward pass for the LSTM cell.
 *
 * Given the gradient for the output of the LSTM cell (which may consist of gradients for h_new and c_new),
 * the function splits the gradients appropriately, computes the effective cell gradient delta_c,
 * determines raw gate gradients (dG), computes bias gradients, previous cell gradients, and finally
 * uses matrix multiplications (via Tensor functions) to compute gradients for the input, weights, and previous hidden state.
 *
 * @param grad_output Gradient tensor of shape (B, 2H) if both h_new and c_new are provided,
 *                    or (B, H) if only h_new is provided.
 * @return A vector containing gradients for:
 *         {input, h_prev, c_prev, weight_ih, weight_hh, bias_ih, bias_hh}.
 */
vector<Tensor> LSTMCellFunction::backward(const Tensor &grad_output) {
    // grad_output has shape (B, 2H) where the first B*H elements are grad_h and the next B*H are grad_c.
    bool use_gpu = (grad_output.device() == CUDA);
    int B = batch_size;
    int H = hidden_dim;
    int D = input_dim;

    // Determine if only h_new's gradient is provided.
    bool single_output = (grad_output.size() == B * H);

    // Allocate grad_h and grad_c on the proper device.
    Tensor grad_h(B * H, grad_output.device());
    Tensor grad_c(B * H, grad_output.device());

    if (use_gpu) {
        int half_size = B * H;
        dim3 blockSize(256);
        dim3 gridSize((half_size + blockSize.x - 1) / blockSize.x);
        if (single_output) {
            split_grad_kernel_single<<<gridSize, blockSize>>>(grad_output.data(), grad_h.data(), grad_c.data(), half_size);
        } else {
            split_grad_kernel_double<<<gridSize, blockSize>>>(grad_output.data(), grad_h.data(), grad_c.data(), half_size);
        }
        cudaDeviceSynchronize();
    } else {
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

    // Allocate temporary tensors.
    Tensor T_tensor(B * H, grad_output.device()); // T = tanh(c_new)
    Tensor delta_c(B * H, grad_output.device());  // effective cell gradient
    Tensor dG(B * 4 * H, grad_output.device());   // concatenated raw gate gradients

    dim3 blockSize(256);
    dim3 gridSize((B * H + blockSize.x - 1) / blockSize.x);

    if (use_gpu) {
        compute_T_kernel<<<gridSize, blockSize>>>(saved_c_new->data.data(), T_tensor.data(), B * H);
        cudaDeviceSynchronize();
        compute_delta_c_kernel<<<gridSize, blockSize>>>(grad_h.data(), grad_c.data(),
                                                        saved_o_gate_act->data.data(), T_tensor.data(),
                                                        delta_c.data(), B * H);
        cudaDeviceSynchronize();
        compute_dG_kernel<<<gridSize, blockSize>>>(delta_c.data(), saved_g_gate_act->data.data(),
                                                   saved_i_gate_act->data.data(), saved_f_gate_act->data.data(),
                                                   grad_h.data(), T_tensor.data(),
                                                   saved_o_gate_act->data.data(), saved_c_prev->data.data(),
                                                   dG.data(), H, B);
        cudaDeviceSynchronize();
    } else {
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

    // Split bias gradients equally (using a 0.5 scaling factor).
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

    // Compute grad_c_prev = delta_c ⊙ f_gate_act.
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

    // Compute gradients with respect to input, weight, and previous hidden state using Tensor functions.
    Tensor trans_Wih = Tensor::transpose(saved_weight_ih->data, D, 4 * H);
    Tensor grad_input = Tensor::matmul(dG, trans_Wih, B, 4 * H, D);

    Tensor trans_input = Tensor::transpose(saved_input->data, B, D);
    Tensor grad_weight_ih = Tensor::matmul(trans_input, dG, D, B, 4 * H);

    Tensor trans_Whh = Tensor::transpose(saved_weight_hh->data, H, 4 * H);
    Tensor grad_h_prev = Tensor::matmul(dG, trans_Whh, B, 4 * H, H);

    Tensor trans_h_prev = Tensor::transpose(saved_h_prev->data, B, H);
    Tensor grad_weight_hh = Tensor::matmul(trans_h_prev, dG, H, B, 4 * H);

    // Collect and return gradients in order.
    vector<Tensor> grads;
    grads.push_back(grad_input);     // Gradient w.r.t. input.
    grads.push_back(grad_h_prev);    // Gradient w.r.t. previous hidden state.
    grads.push_back(grad_c_prev);    // Gradient w.r.t. previous cell state.
    grads.push_back(grad_weight_ih); // Gradient w.r.t. weight_ih.
    grads.push_back(grad_weight_hh); // Gradient w.r.t. weight_hh.
    grads.push_back(grad_bias_ih);   // Gradient w.r.t. bias_ih.
    grads.push_back(grad_bias_hh);   // Gradient w.r.t. bias_hh.
    return grads;
}

/**
 * @brief Initializes the elements of a tensor with random values scaled by a limit.
 *
 * @param t Reference to the tensor to initialize.
 * @param limit Scaling constant determining the range of initialization.
 */
static void init_tensor(Tensor &t, float limit) {
    size_t size = t.size();
    for (size_t i = 0; i < size; i++) {
        t.data()[i] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * limit;
    }
}

/**
 * @brief Constructs an LSTMCell.
 *
 * Initializes the weight matrices and biases for the LSTM cell using uniform initialization,
 * and transfers them to CUDA if required.
 *
 * @param input_dim Dimensionality of the input vector.
 * @param hidden_dim Dimensionality of the hidden state.
 * @param device Target device (CPU or CUDA).
 */
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

    // Create bias_ih on CPU and transfer if needed.
    Tensor b_ih(gate_size, CPU);
    b_ih.fill(0.0f);
    if (device == CUDA)
        b_ih.toCUDA();
    bias_ih = make_shared<Variable>(b_ih, true, "bias_ih");

    // Create bias_hh on CPU and transfer if needed.
    Tensor b_hh(gate_size, CPU);
    b_hh.fill(0.0f);
    if (device == CUDA)
        b_hh.toCUDA();
    bias_hh = make_shared<Variable>(b_hh, true, "bias_hh");
}

/**
 * @brief Forward pass of the LSTM cell module.
 *
 * Processes a sequence time step by using LSTMCellFunction::apply on the provided inputs.
 *
 * @param inputs A vector containing exactly three inputs: {input, h_prev, c_prev}.
 * @return A vector containing the new hidden state and new cell state.
 */
vector<VarPtr> LSTMCell::forward(const vector<VarPtr> &inputs) {
    // Ensure exactly three inputs are provided.
    assert(inputs.size() == 3 && "LSTMCell expects exactly three inputs: {input, h_prev, c_prev}");

    // Compute the new states using the LSTM cell function.
    auto outputs = LSTMCellFunction::apply(inputs[0], inputs[1], inputs[2],
                                           weight_ih, weight_hh,
                                           bias_ih, bias_hh,
                                           input_dim, hidden_dim);
    return {outputs.first, outputs.second};
}

/**
 * @brief Returns the learnable parameters of the LSTM cell.
 *
 * @return A vector containing weight_ih, weight_hh, bias_ih, and bias_hh.
 */
vector<VarPtr> LSTMCell::parameters() {
    return {weight_ih, weight_hh, bias_ih, bias_hh};
}
} // namespace nn
} // namespace kernelnet