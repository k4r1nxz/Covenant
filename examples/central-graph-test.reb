REBOL [
    Title: "Central Graph System Test"
    Description: "Test the central graph ownership system"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Testing Central Graph Ownership System"
print "======================================"

;; Create variables using the central graph system
x: covenant/graph/variable/requires-grad [2.0 3.0] true
y: covenant/graph/variable/requires-grad [1.0 4.0] true

print ["x =" mold x/data "requires_grad =" x/requires-grad]
print ["y =" mold y/data "requires_grad =" y/requires-grad]

;; Perform operations using central graph
z: covenant/graph/add x y
print ["z = x + y =" mold z/data]

w: covenant/graph/mul x y
print ["w = x * y =" mold w/data]

;; Test graph statistics
stats: covenant/graph/get-stats
print ["Graph stats: Total nodes =" stats/total-nodes ", Leaf nodes =" stats/leaf-nodes]

;; Test backward pass
print "^nTesting backward pass..."
if z/grad [print ["z grad before backward:" mold z/grad]]
if x/grad [print ["x grad before backward:" mold x/grad]]
if y/grad [print ["y grad before backward:" mold y/grad]]

;; Zero gradients first
covenant/graph/zero-grads
print "Gradients zeroed."

;; Perform backward pass
print "Performing backward pass on z..."
covenant/graph/backward z

print "^nAfter backward pass:"
if z/grad [print ["z grad:" mold z/grad]]
if x/grad [print ["x grad:" mold x/grad]]
if y/grad [print ["y grad:" mold y/grad]]

print "^nCentral graph system test completed successfully!"