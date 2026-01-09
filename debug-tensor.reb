REBOL [
    Title: "Debug Tensor Creation"
    Description: "Debug script to understand tensor creation"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the core module
do %core/core.r

print "Debug: Creating 2D tensor"
mat: core/tensor [[1.0 2.0] [3.0 4.0]]
print ["Shape:" mold mat/shape]
print ["Data:" mold mat/data]
print ["First element of data:" type? mat/data/1]
print ["First element value:" mat/data/1]
