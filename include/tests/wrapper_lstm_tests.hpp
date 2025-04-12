#pragma once

#include "kernelnet.hpp"

using namespace std;

inline void runWrapperLSTMTests() {
    int batch_size = 4;
    int sequence_length = 5;
    int input_dim = 3;
    int hidden_dim = 4;
    int output_dim = 1; // Network output is scalar per sample.
    float learning_rate = 0.0001f;
    int num_epochs = 2000;

    cout << "===== Running Wrapper LSTM Test on CPU =====" << endl;
    {
        Device dev = CPU;
        Tensor input_data, target_data;
        generateSequenceData(batch_size, sequence_length, input_dim, input_data, target_data);
        VarPtr input_var = make_shared<Variable>(input_data, false, "lstm_input");

        // Build network layers: LSTMWrapper then Dense.
        auto lstmWrapper = make_shared<LSTM>(batch_size, sequence_length, input_dim, hidden_dim, dev);
        auto dense = make_shared<Dense>(hidden_dim, output_dim, dev);

        // Build a Sequential container.
        auto model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
            lstmWrapper, dense});

        vector<VarPtr> params = model->parameters();
        SGD optimizer(params, learning_rate);
        Trainer trainer(model, optimizer);

        // Wrap target data.
        VarPtr target_var = make_shared<Variable>(target_data, false, "target");

        // Train for num_epochs.
        vector<VarPtr> inputs = {input_var};
        vector<VarPtr> targets = {target_var};
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            trainer.trainEpoch(inputs, targets);
            if (epoch % 100 == 0) {
                VarPtr prediction = model->forward(input_var);
                VarPtr loss = MSEFunction::apply(prediction, target_data);
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        // Final evaluation.
        VarPtr final_pred = model->forward(input_var);
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

    cout << "\n===== Running Wrapper LSTM Test on CUDA =====" << endl;
    {
        Device dev = CUDA;
        int batch_size = 4, sequence_length = 5, input_dim = 3, hidden_dim = 4, output_dim = 1;
        Tensor input_data, target_data;
        generateSequenceData(batch_size, sequence_length, input_dim, input_data, target_data);
        input_data.toCUDA();
        target_data.toCUDA();
        VarPtr input_var = make_shared<Variable>(input_data, false, "lstm_input");

        auto lstmWrapper = make_shared<LSTM>(batch_size, sequence_length, input_dim, hidden_dim, dev);
        auto dense = make_shared<Dense>(hidden_dim, output_dim, dev);
        auto model = make_shared<Sequential>(initializer_list<shared_ptr<SingleInputModule>>{
            lstmWrapper, dense});

        vector<VarPtr> params = model->parameters();
        SGD optimizer(params, learning_rate);
        Trainer trainer(model, optimizer);

        VarPtr target_var = make_shared<Variable>(target_data, false, "target");

        vector<VarPtr> inputs = {input_var};
        vector<VarPtr> targets = {target_var};

        for (int epoch = 0; epoch < num_epochs; epoch++) {
            trainer.trainEpoch(inputs, targets);
            if (epoch % 100 == 0) {
                VarPtr prediction = model->forward(input_var);
                VarPtr loss = MSEFunction::apply(prediction, target_data);
                loss->data.toCPU();
                cout << "Epoch " << epoch << " Loss: " << loss->data.data()[0] << endl;
            }
        }

        VarPtr final_pred = model->forward(input_var);
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