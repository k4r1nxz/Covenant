REBOL [
    Title: "Test Core Tensor Grad"
    Description: "Test core tensor grad function directly"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

do %../core/core.r

print "Testing core tensor-grad function directly..."

x: core/tensor-grad/requires-grad [1.0 2.0 3.0]
print ["Direct call result type:" type? x]
print ["Direct call result:" mold x]

y: core/tensor [1.0 2.0 3.0]
print ["Regular tensor result type:" type? y]
print ["Regular tensor result:" mold y]
