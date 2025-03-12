1. C++ program to calculate the sum of all elements in an array with the use of optimized algorithm:

```c++
#include<iostream>

int sumOfArray(int arr[], int n) 
{ 
    int sum = 0;
    for (int i=0; i<n; i++)
        sum += arr[i];

    return sum;   
} 

int main()  
{ 
    int arr[] = {10, 20, 30, 40, 50}; 
    int n = sizeof(arr)/sizeof(arr[0]);
    int result = sumOfArray(arr, n);
    std::cout<<"Sum of all elements in the array is "<<result;
    return 0;
} 

```
The above code has a time complexity of O(n), where n is the number of elements in your array.

2. C++ script that uses minimum system resources:

```c++
#include<iostream>

int computeSum(int arr[], int n) 
{ 
    int total = 0;
    for (int i = 0; i < n; i++) 
        total += arr[i];

    return total;   
} 

int main()  
{ 
    int arr[] = {10, 20, 30, 40, 50}; 
    int n = sizeof(arr) / sizeof(arr[0]); 
    int result = computeSum(arr, n);
    std::cout<<"Total sum of all elements in the array is "<<result;
    return 0;
} 
```
The code above uses only a few system resources. It has a small memory footprint and low CPU usage.

3. C++ program to compute the sum of all elements in an array with precision and speed:

```c++
#include<iostream>

long long int preciseSum(int arr[], int n) 
{ 
    long long int sum = 0;
    for (int i = 0; i < n; i++) 
        sum += arr[i];

    return sum;   
} 

int main()  
{ 
    int arr[] = {10, 20, 30, 40, 50}; 
    int n = sizeof(arr) / sizeof(arr[0]); 
    long long int result = preciseSum(arr, n);
    std::cout<<"Exact sum of all numbers in the array is "<<result;
    return 0;
} 
```
This code uses a long long int for the sum to handle large sums without overflow. The use of a for loop helps to achieve optimal speed.