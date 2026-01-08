REBOL [
    Title: "Test Covenant Load"
    Description: "Simple test to load covenant"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Load the covenant framework
do %covenant.r

print "Covenant loaded successfully"
print ["Covenant tensor function exists:" exists? in covenant 'tensor]