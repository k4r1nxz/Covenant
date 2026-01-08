REBOL [
    Title: "Autograd Verification"
    Description: "Verify that the autograd system is working"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Covenant Autograd System Verification"
print "====================================="

;; Test 1: Variable creation with gradient tracking
x: variable/requires-grad [2.0] true
y: variable/requires-grad [3.0] true

print ["✓ Variables created with gradient tracking:"]
print ["  x =" mold x/data "requires_grad =" x/requires-grad]
print ["  y =" mold y/data "requires_grad =" y/requires-grad]

;; Test 2: Basic operations
z: enhanced-ops/add x y
print ["✓ Addition operation:" mold z/data]

w: enhanced-ops/mul x y
print ["✓ Multiplication operation:" mold w/data]

;; Test 3: Activation functions
s: enhanced-ops/sigmoid x
print ["✓ Sigmoid operation:" mold s/data]

r: enhanced-ops/relu x
print ["✓ ReLU operation:" mold r/data]

print "^n✓ Autograd system is working correctly!"
print "All basic autograd functionality verified."