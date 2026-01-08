REBOL [
    Title: "Covenant Autograd Test"
    Description: "Test the autograd system in Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the Covenant framework
change-dir %..
do %covenant.reb
change-dir %examples

print "Covenant AI Framework - Autograd Test"
print "====================================="

;; Test basic variable creation with gradient tracking
print "^n1. Testing variable creation with gradient tracking..."
x: variable/requires-grad [2.0] true
print ["Variable x:" mold x/data "requires_grad:" x/requires-grad]

y: variable/requires-grad [3.0] true
print ["Variable y:" mold y/data "requires_grad:" y/requires-grad]

;; Test basic operations
print "^n2. Testing basic operations with gradients..."
z: enhanced-ops/add x y
print ["z = x + y =" mold z/data]

w: enhanced-ops/mul x y
print ["w = x * y =" mold w/data]

;; Test sigmoid operation
print "^n3. Testing sigmoid operation with gradients..."
s: enhanced-ops/sigmoid x
print ["sigmoid(x) =" mold s/data]

;; Test ReLU operation
print "^n4. Testing ReLU operation with gradients..."
r: enhanced-ops/relu x
print ["relu(x) =" mold r/data]

;; Test backward propagation
print "^n5. Testing backward propagation..."
; Set up a simple computational graph: z = x + y, w = z * x
z: enhanced-ops/add x y
w: enhanced-ops/mul z x

print ["Computational graph: z = x + y, w = z * x"]
print ["z =" mold z/data]
print ["w =" mold w/data]

; Perform backward pass
w/backward

print ["After backward pass:"]
if x/grad [print ["x.grad =" mold x/grad]]
if y/grad [print ["y.grad =" mold y/grad]]
if z/grad [print ["z.grad =" mold z/grad]]

print "^nAutograd test completed successfully!"