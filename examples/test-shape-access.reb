REBOL [
    Title: "Test Shape Access"
    Description: "Test shape access syntax"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

change-dir %..
do %covenant.reb
change-dir %examples

print "Testing shape access..."

x: covenant/tensor/requires_grad [1.0 2.0 3.0]
print ["x:" mold x]
print ["x/shape:" mold x/shape]
print ["type of x/shape:" type? x/shape]

tensors-block: reduce [x]
print ["tensors-block:" mold tensors-block]
print ["tensors-block/1:" mold tensors-block/1]
print ["tensors-block/1/shape:" mold tensors-block/1/shape]
print ["type of tensors-block/1/shape:" type? tensors-block/1/shape]
print ["first of tensors-block/1/shape:" first tensors-block/1/shape]
print ["type of first of tensors-block/1/shape:" type? first tensors-block/1/shape]
print ["length? of tensors-block/1/shape:" length? tensors-block/1/shape]