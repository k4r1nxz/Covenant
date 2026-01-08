REBOL [
    Title: "Simple Test"
    Description: "Simple test to isolate the issue"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Testing basic functionality..."

x: covenant/tensor [1.0 2.0 3.0]
print ["Created tensor x:" mold x/data]

y: covenant/tensor [4.0 5.0 6.0]
print ["Created tensor y:" mold y/data]

z: covenant/add x y
print ["Added tensors:" mold z/data]

w: covenant/mul x y
print ["Multiplied tensors:" mold w/data]

print "Basic operations work fine."

; Test mean separately
mean-tensor: covenant/mean w
print ["Mean tensor shape:" mold mean-tensor/shape "data:" mold mean-tensor/data]