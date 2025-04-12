#pragma once
#include "kernelnet.hpp"

using namespace std;

inline void runSeqLSTMTests() {
    // Common training settings.
    int batch_size = 4;
    int sequence_length = 5;
    int input_dim = 3;
    int hidden_dim = 4;
    int output_dim = 1; // Our network output is scalar per sample.
    // Use the same learning rate and number of epochs on both devices.
    float learning_rate = 0.0001f;
    int num_epochs = 2000;

    cout << "===== Running LSTM Training Test on CPU =====" << endl;
    {
        Device dev = CPU;

        // Create synthetic data on CPU.
        Tensor input_data, target_data;
        generateSequenceData(batch_size, sequence_length, input_dim, input_data, target_data);

        // Wrap input data into a Variable.
        VarPtr input_var = make_shared<Variable>(input_data, false, "lstm_input");

        // Build network layers (all on CPU).
        LSTMCell lstm(input_dim, hidden_dim, dev);
        Dense fc(hidden_dim, output_dim, dev);

        // Combine parameters from both layers.
        vector<VarPtr> params = lstm.parameters();
        vector<VarPtr> fc_params = fc.parameters();
        params.insert(params.end(), fc_params.begin(), fc_params.end());
        SGD optimizer(params, learning_rate);

        float loss_value = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            // Initialize hidden and cell states as zeros.
            Tensor h0(batch_size * hidden_dim, dev);
            Tensor c0(batch_size * hidden_dim, dev);
            h0.fill(0.0f);
            c0.fill(0.0f);
            VarPtr h_prev = make_shared<Variable>(h0, true, "h0");
            VarPtr c_prev = make_shared<Variable>(c0, true, "c0");

            // Process the sequence one time step at a time.
            VarPtr current_hidden;
            for (int t = 0; t < sequence_length; t++) {
                int offset = t * input_dim;
                // Assume SliceFunction::apply returns a single VarPtr for the given slice.
                VarPtr x_t = SliceFunction::apply(input_var, batch_size, offset, offset + input_dim);
                // Pack the three inputs as a vector for the LSTMCell.
                vector<VarPtr> lstm_inputs = {x_t, h_prev, c_prev};
                // Call the LSTMCell's forward method (which now returns a vector of outputs).
                vector<VarPtr> lstm_outputs = lstm.forward(lstm_inputs);
                // Extract new hidden and cell states.
                h_prev = lstm_outputs[0];
                c_prev = lstm_outputs[1];
                current_hidden = h_prev;
            }
            // Final hidden state goes through Dense layer.
            VarPtr output = fc.forward(current_hidden);

            // Compute MSE loss.
            VarPtr loss = MSEFunction::apply(output, target_data);
            loss_value = loss->data.sum(); // obtain scalar loss

            // Backward pass.
            loss->backward(loss->data);
            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 100 == 0) {
                cout << "Epoch " << epoch << " Loss: " << loss_value << endl;
            }
        }
        // Final evaluation.
        VarPtr h_prev_eval, c_prev_eval;
        {
            Tensor h0(batch_size * hidden_dim, dev);
            Tensor c0(batch_size * hidden_dim, dev);
            h0.fill(0.0f);
            c0.fill(0.0f);
            h_prev_eval = make_shared<Variable>(h0, false, "h0_eval");
            c_prev_eval = make_shared<Variable>(c0, false, "c0_eval");
        }

        VarPtr current_hidden;
        for (int t = 0; t < sequence_length; t++) {
            int offset = t * input_dim;
            VarPtr x_t = SliceFunction::apply(input_var, batch_size, offset, offset + input_dim);
            vector<VarPtr> lstm_inputs = {x_t, h_prev_eval, c_prev_eval};
            vector<VarPtr> lstm_outputs = lstm.forward(lstm_inputs);
            h_prev_eval = lstm_outputs[0];
            c_prev_eval = lstm_outputs[1];
            current_hidden = h_prev_eval;
        }
        // Get final prediction.
        VarPtr final_pred = fc.forward(current_hidden);
        final_pred->data.toCPU();
        cout << "Final Prediction (CPU): ";
        for (int i = 0; i < output_dim; i++) {
            cout << final_pred->data.data()[i] << " ";
        }
        cout << endl;

        target_data.toCPU();
        cout << "Ground Truth (CPU): ";
        for (int i = 0; i < target_data.size(); i++) {
            cout << target_data.data()[i] << " ";
        }
        cout << endl;
    }

    cout << "\n===== Running LSTM Training Test on CUDA =====" << endl;
    {
        Device dev = CUDA;

        // Generate synthetic data on CPU, then move to CUDA.
        Tensor input_data, target_data;
        generateSequenceData(batch_size, sequence_length, input_dim, input_data, target_data);
        input_data.toCUDA();
        target_data.toCUDA();

        // Wrap input.
        VarPtr input_var = make_shared<Variable>(input_data, false, "lstm_input");

        // Build network layers on CUDA.
        LSTMCell lstm(input_dim, hidden_dim, dev);
        Dense fc(hidden_dim, output_dim, dev);
        vector<VarPtr> params = lstm.parameters();
        vector<VarPtr> fc_params = fc.parameters();
        params.insert(params.end(), fc_params.begin(), fc_params.end());
        SGD optimizer(params, learning_rate);

        float loss_value = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            // Initialize hidden and cell states to zeros.
            Tensor h0(batch_size * hidden_dim, dev);
            Tensor c0(batch_size * hidden_dim, dev);
            h0.fill(0.0f);
            c0.fill(0.0f);
            VarPtr h_prev = make_shared<Variable>(h0, true, "h0");
            VarPtr c_prev = make_shared<Variable>(c0, true, "c0");

            VarPtr current_hidden;
            for (int t = 0; t < sequence_length; t++) {
                int offset = t * input_dim;
                VarPtr x_t = SliceFunction::apply(input_var, batch_size, offset, offset + input_dim);
                vector<VarPtr> lstm_inputs = {x_t, h_prev, c_prev};
                vector<VarPtr> lstm_outputs = lstm.forward(lstm_inputs);
                h_prev = lstm_outputs[0];
                c_prev = lstm_outputs[1];
                current_hidden = h_prev;
            }
            VarPtr output = fc.forward(current_hidden);
            VarPtr loss = MSEFunction::apply(output, target_data);
            loss_value = loss->data.sum();
            loss->backward(loss->data);
            optimizer.step();
            optimizer.zero_grad();

            if (epoch % 100 == 0) {
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }
        // Final forward pass.
        VarPtr h_prev_eval, c_prev_eval;
        {
            Tensor h0(batch_size * hidden_dim, dev);
            Tensor c0(batch_size * hidden_dim, dev);
            h0.fill(0.0f);
            c0.fill(0.0f);
            h_prev_eval = make_shared<Variable>(h0, false, "h0_eval");
            c_prev_eval = make_shared<Variable>(c0, false, "c0_eval");
        }
        VarPtr current_hidden;
        for (int t = 0; t < sequence_length; t++) {
            int offset = t * input_dim;
            VarPtr x_t = SliceFunction::apply(input_var, batch_size, offset, offset + input_dim);
            vector<VarPtr> lstm_inputs = {x_t, h_prev_eval, c_prev_eval};
            vector<VarPtr> lstm_outputs = lstm.forward(lstm_inputs);
            h_prev_eval = lstm_outputs[0];
            c_prev_eval = lstm_outputs[1];
            current_hidden = h_prev_eval;
        }
        VarPtr final_pred = fc.forward(current_hidden);
        final_pred->data.toCPU();
        cout << "Final Prediction (CUDA): ";
        for (int i = 0; i < output_dim; i++) {
            cout << final_pred->data.data()[i] << " ";
        }
        cout << endl;

        target_data.toCPU();
        cout << "Ground Truth (CUDA): ";
        for (int i = 0; i < target_data.size(); i++) {
            cout << target_data.data()[i] << " ";
        }
        cout << endl;
    }
}