REBOL [
    Title: "Debug Flatten Function Step by Step"
    Description: "Debug script to understand flatten function step by step"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Define the flatten function with debug prints
flatten-data: func [
    data [block! number!]
] [
    print ["Flattening:" mold data]
    if number? data [
        print ["  Number case, returning:" mold reduce [data]]
        return reduce [data]
    ]
    if empty? data [
        print ["  Empty case, returning: []"]
        return copy []
    ]
    
    result: copy []
    print ["  Initial result: []"]
    foreach item data [
        print ["  Processing item:" mold item]
        either block? item [
            flattened-item: flatten-data item
            print ["    Item is block, flattened to:" mold flattened-item]
            append result flattened-item
            print ["    Result after append:" mold result]
        ] [
            print ["    Item is not block, appending:" mold reduce [item]]
            append result reduce [item]
            print ["    Result after append:" mold result]
        ]
    ]
    print ["  Final result:" mold result]
    result
]

test-data: [[1.0 2.0] [3.0 4.0]]
print ["Original data:" mold test-data]
flattened: flatten-data test-data
print ["Final flattened data:" mold flattened]