REBOL [
    Title: "Test Gradient Tensors"
    Description: "Test gradient tensor functionality"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Testing gradient tensor creation..."

x: covenant/tensor/requires_grad [1.0 2.0 3.0]
print ["Tensor x type:" type? x]
print ["Tensor x fields:" words-of x]
print ["Tensor x data exists:" exists? in x 'data]
print ["Tensor x requires_grad exists:" exists? in x 'requires-grad]

if x/data [
    print ["Tensor x data:" mold x/data]
]
if x/requires-grad [
    print ["Tensor x requires_grad:" x/requires-grad]
]