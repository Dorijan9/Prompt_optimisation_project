#include <iostream>
using namespace std;
int factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }
int main() { cout << factorial(5) << endl; return 0; }