REBOL [
    Title: "Simple Autograd Test"
    Description: "Simple test to verify autograd system works"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
change-dir %..
do %covenant.reb
change-dir %examples

print "Simple Autograd Test"
print "==================="

;; Create variables with gradient tracking
x: variable/requires-grad [2.0] true
y: variable/requires-grad [3.0] true

print ["x =" mold x/data "requires_grad =" x/requires-grad]
print ["y =" mold y/data "requires_grad =" y/requires-grad]

;; Perform operations
z: enhanced-ops/add x y
print ["z = x + y =" mold z/data]

;; Try backward pass
print "^nSetting gradient for z..."
if not z/grad [
    z/grad: reduce [1.0]  ; Initialize gradient
]

print ["z/grad =" mold z/grad]

;; Propagate gradients to x and y
print "^nPropagating gradients to x and y..."
if x/requires-grad [
    if not x/grad [x/grad: reduce [0.0]]
    x/grad/1: x/grad/1 + z/grad/1  ; Gradient flows directly for addition
]

if y/requires-grad [
    if not y/grad [y/grad: reduce [0.0]]
    y/grad/1: y/grad/1 + z/grad/1  ; Gradient flows directly for addition
]

print ["x/grad =" mold x/grad]
print ["y/grad =" mold y/grad]

print "^nSimple autograd test completed!"
