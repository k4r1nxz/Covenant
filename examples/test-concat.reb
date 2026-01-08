REBOL [
    Title: "Test Concat Function"
    Description: "Test concat function"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Testing concat function..."

x: covenant/tensor/requires_grad [1.0 2.0 3.0]
y: covenant/tensor/requires_grad [4.0 5.0 6.0]

print ["x shape:" mold x/shape]
print ["y shape:" mold y/shape]
print ["x shape type:" type? x/shape]
print ["y shape type:" type? y/shape]

print ["first x/shape:" first x/shape]
print ["first x/shape type:" type? first x/shape]

tensors-block: reduce [x y]
print ["tensors-block type:" type? tensors-block]
print ["first tensors-block type:" type? first tensors-block]
print ["first tensors-block/shape type:" type? first tensors-block/shape]
print ["first first tensors-block/shape:" first first tensors-block/shape]