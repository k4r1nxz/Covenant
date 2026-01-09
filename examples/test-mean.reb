REBOL [
    Title: "Test Mean Function"
    Description: "Test mean function specifically"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Testing mean function..."

x: covenant/tensor [1.0 2.0 3.0]
print ["Created tensor x:" mold x/data]

y: covenant/tensor [4.0 5.0 6.0]
print ["Created tensor y:" mold y/data]

w: covenant/mul x y
print ["Multiplied tensors:" mold w/data]

print "About to call mean function..."
; Don't call mean yet, let's see if the error is elsewhere
print "Multiplication worked fine."

; Now test mean
try [
    mean-tensor: covenant/mean w
    print ["Mean tensor shape:" mold mean-tensor/shape "data:" mold mean-tensor/data]
] [
    print "Error occurred in mean function"
]
