#include "trainer.hpp"

namespace kernelnet {
namespace trainer {

/**
 * @brief Constructs a Trainer object.
 *
 * @param model Shared pointer to a Sequential model.
 * @param optimizer SGD optimizer instance.
 * @param loss_fn Optional loss function (defaults to MSE).
 */
Trainer::Trainer(const shared_ptr<Sequential> &model, const SGD &optimizer,
                 LossFunction loss_fn)
    : model(model), optimizer(optimizer), loss_fn(loss_fn) {}

/**
 * @brief Trains the model for one epoch on the provided data.
 *
 * Performs the following steps for each input-target pair:
 * - Forward pass
 * - Loss computation
 * - Backward pass
 * - Optimizer step
 * - Gradient reset
 *
 * @param inputs Vector of input VarPtr (batch of samples).
 * @param targets Vector of ground truth VarPtr (same size as inputs).
 */
void Trainer::trainEpoch(const vector<VarPtr> &inputs, const vector<VarPtr> &targets) {
    assert(inputs.size() == targets.size() && "Mismatched number of inputs and targets");

    for (size_t i = 0; i < inputs.size(); ++i) {
        // Forward pass.
        VarPtr prediction = model->forward(inputs[i]);

        // Compute loss.
        VarPtr loss = loss_fn(prediction, targets[i]->data);
        // cout << "Loss: " << loss->data.data()[0] << endl;

        // Backward pass to compute gradients.
        Tensor grad_seed(1, loss->data.device());
        grad_seed.fill(1.0f);

        /**
         * Uncomment for debugging:
         * if (grad_seed.device() == CUDA) {
         *     vector<float> host_grad(1);
         *     cudaMemcpy(host_grad.data(), grad_seed.data(), sizeof(float), cudaMemcpyDeviceToHost);
         *     cout << "grad_seed[0] (CUDA) = " << host_grad[0] << endl;
         * } else {
         *     cout << "grad_seed[0] (CPU) = " << grad_seed.data()[0] << endl;
         * }
         */

        loss->backward(grad_seed);

        // Parameter update step.
        optimizer.step();

        // Clear gradients before the next sample.
        optimizer.zero_grad();
    }
}

} // namespace trainer
} // namespace kernelnet