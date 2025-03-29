#include "tensor.hpp"
using namespace std;

int main() {
    Tensor a(10, CPU);
    Tensor b(10, CPU);

    a.fill(2.0f);
    b.fill(3.0f);

    auto c = Tensor::add(a, b);
    cout << "CPU result:\n";
    c.print();

    a.toCUDA();
    b.toCUDA();

    auto d = Tensor::add(a, b);
    cout << "CUDA result:\n";
    d.print();

    return 0;
}