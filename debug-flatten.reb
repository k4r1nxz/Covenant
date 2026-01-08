REBOL [
    Title: "Debug Flatten Function"
    Description: "Debug script to understand flatten function"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Define the flatten function
flatten-data: func [
    data [block! number!]
] [
    if number? data [return reduce [data]]
    if empty? data [return copy []]
    
    result: copy []
    foreach item data [
        either block? item [
            append result flatten-data item
        ] [
            append result reduce [item]
        ]
    ]
    result
]

test-data: [[1.0 2.0] [3.0 4.0]]
print ["Original data:" mold test-data]
flattened: flatten-data test-data
print ["Flattened data:" mold flattened]