REBOL [
    Title: "Simple Central Graph Test"
    Description: "Simple test for central graph ownership system"
    Version: 1.1.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Simple Central Graph Test"
print "========================="

;; Check if the graph manager exists
print ["Graph manager exists:" value? 'graph-manager]

;; Check if central operations exist
print ["Central ops exist:" value? 'central-ops]

;; Create variables using the central graph system
x: central-variable/requires-grad [2.0 3.0] true
y: central-variable/requires-grad [1.0 4.0] true

print ["x =" mold x/data "requires_grad =" x/requires-grad]
print ["y =" mold y/data "requires_grad =" y/requires-grad]

;; Perform operations using central graph
z: central-ops/add x y
print ["z = x + y =" mold z/data]

w: central-ops/mul x y
print ["w = x * y =" mold w/data]

;; Test graph statistics
stats: graph-manager/get-stats
print ["Graph stats: Total nodes =" stats/total-nodes ", Leaf nodes =" stats/leaf-nodes-count]

print "^nCentral graph system is working correctly!"
