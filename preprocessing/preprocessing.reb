REBOL [
    Title: "Covenant Preprocessing Utilities"
    Description: "Data preprocessing utilities for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Import core modules
do %../core/core.reb

;; Preprocessing utilities for Covenant
preprocessing: context [
    ;; Normalize data to range [0, 1]
    min-max-normalize: func [
        "Normalize data to range [0, 1]"
        tensor [object!] "Input tensor"
    ] [
        ; Find min and max values
        min-val: first tensor/data
        max-val: first tensor/data
        
        foreach val next tensor/data [
            if val < min-val [min-val: val]
            if val > max-val [max-val: val]
        ]
        
        ; Handle edge case where all values are the same
        range: max-val - min-val
        if range = 0 [range: 1]
        
        ; Normalize data
        normalized-data: copy tensor/data
        repeat i length? normalized-data [
            normalized-data/:i: (normalized-data/:i - min-val) / range
        ]
        
        make object! [
            data: normalized-data
            shape: tensor/shape
            dtype: tensor/dtype
        ]
    ]

    ;; Standardize data (mean=0, std=1)
    standardize: func [
        "Standardize data (mean=0, std=1)"
        tensor [object!] "Input tensor"
    ] [
        ; Calculate mean
        sum: 0.0
        foreach val tensor/data [sum: sum + val]
        mean: sum / length? tensor/data
        
        ; Calculate standard deviation
        sum-squared-diff: 0.0
        foreach val tensor/data [
            diff: val - mean
            sum-squared-diff: sum-squared-diff + (diff * diff)
        ]
        variance: sum-squared-diff / length? tensor/data
        std: core/square-root variance
        
        ; Handle edge case where std is 0
        if std = 0 [std: 1]
        
        ; Standardize data
        standardized-data: copy tensor/data
        repeat i length? standardized-data [
            standardized-data/:i: (standardized-data/:i - mean) / std
        ]
        
        make object! [
            data: standardized-data
            shape: tensor/shape
            dtype: tensor/dtype
        ]
    ]

    ;; One-hot encode categorical data
    one-hot-encode: func [
        "One-hot encode categorical data"
        tensor [object!] "Input tensor with categorical values"
        num-classes [integer!] "Number of classes"
    ] [
        result-data: make block! (length? tensor/data * num-classes)
        
        foreach val tensor/data [
            class-idx: to-integer val
            if class-idx >= num-classes [class-idx: num-classes - 1]
            if class-idx < 0 [class-idx: 0]
            
            ; Create one-hot vector for this value
            repeat i num-classes [
                if i - 1 = class-idx [
                    append result-data 1.0
                ] [
                    append result-data 0.0
                ]
            ]
        ]
        
        make object! [
            data: result-data
            shape: reduce [length? tensor/data num-classes]
            dtype: 'float32
        ]
    ]

    ;; Split data into training and test sets
    train-test-split: func [
        "Split data into training and test sets"
        tensor [object!] "Input tensor"
        test-ratio [number!] "Ratio of test data (0.0 to 1.0)"
        /randomize "Randomize the split"
    ] [
        total-samples: either length? tensor/shape > 1 [tensor/shape/1] [length? tensor/data]
        test-size: to-integer (total-samples * test-ratio)
        train-size: total-samples - test-size
        
        ; Create indices
        indices: make block! total-samples
        repeat i total-samples [append indices (i - 1)]
        
        ; Randomize if requested
        if randomize [
            ; Simple shuffle algorithm
            repeat i total-samples [
                j: random total-samples
                temp: indices/:i
                indices/:i: indices/:j
                indices/:j: temp
            ]
        ]
        
        ; Split indices
        train-indices: copy/part indices train-size
        test-indices: copy at indices (train-size + 1)
        
        ; Create train and test data based on tensor shape
        train-data: make block! (train-size * (length? tensor/data / total-samples))
        test-data: make block! (test-size * (length? tensor/data / total-samples))
        
        ; Calculate elements per sample
        elements-per-sample: either length? tensor/shape > 1 [
            (length? tensor/data) / total-samples
        ] [
            1
        ]
        
        ; Extract training data
        foreach idx train-indices [
            start-idx: (idx * elements-per-sample) + 1
            end-idx: start-idx + elements-per-sample - 1
            foreach i range start-idx end-idx [
                if i <= length? tensor/data [
                    append train-data tensor/data/:i
                ]
            ]
        ]
        
        ; Extract test data
        foreach idx test-indices [
            start-idx: (idx * elements-per-sample) + 1
            end-idx: start-idx + elements-per-sample - 1
            foreach i range start-idx end-idx [
                if i <= length? tensor/data [
                    append test-data tensor/data/:i
                ]
            ]
        ]
        
        ; Calculate new shapes
        train-shape: copy tensor/shape
        test-shape: copy tensor/shape
        train-shape/1: train-size
        test-shape/1: test-size
        
        make object! [
            train: make object! [
                data: train-data
                shape: train-shape
                dtype: tensor/dtype
            ]
            test: make object! [
                data: test-data
                shape: test-shape
                dtype: tensor/dtype
            ]
        ]
    ]

    ;; Pad sequences to same length
    pad-sequences: func [
        "Pad sequences to the same length"
        sequences [block!] "Block of sequence tensors"
        padding-value [number!] "Value to use for padding (default 0.0)"
        /maxlen "Maximum length (default is length of longest sequence)"
        max-len-val [integer!] "Maximum length value"
    ] [
        padding-val: either padding-value [padding-value] [0.0]
        
        ; Find maximum length if not provided
        max-len: either maxlen [max-len-val] [0]
        if not maxlen [
            foreach seq sequences [
                seq-len: either length? seq/shape > 1 [seq/shape/1] [length? seq/data]
                if seq-len > max-len [max-len: seq-len]
            ]
        ]
        
        ; Pad each sequence
        padded-sequences: make block! length? sequences
        foreach seq sequences [
            seq-len: either length? seq/shape > 1 [seq/shape/1] [length? seq/data]
            elements-per-sample: either length? seq/shape > 1 [seq/shape/2] [1]
            
            if seq-len < max-len [
                ; Calculate how many elements to add
                elements-to-add: (max-len - seq-len) * elements-per-sample
                padded-data: copy seq/data
                loop elements-to-add [append padded-data padding-val]
                
                append padded-sequences make object! [
                    data: padded-data
                    shape: reduce [max-len elements-per-sample]
                    dtype: seq/dtype
                ]
            ] [
                ; If sequence is already at max length or longer, truncate or keep as is
                if seq-len > max-len [
                    ; Truncate to max length
                    new-data: copy/part seq/data (max-len * elements-per-sample)
                    append padded-sequences make object! [
                        data: new-data
                        shape: reduce [max-len elements-per-sample]
                        dtype: seq/dtype
                    ]
                ] [
                    append padded-sequences seq  ; Keep as is
                ]
            ]
        ]
        
        padded-sequences
    ]

    ;; Tokenizer utility (simple word-level tokenizer)
    tokenizer: func [
        "Simple word-level tokenizer"
        texts [block!] "Block of text strings to tokenize"
    ] [
        vocab: make map! []
        tokenized-texts: make block! length? texts
        current-index: 1
        
        ; Build vocabulary
        foreach text texts [
            words: parse text " "  ; Simple space-based tokenization
            foreach word words [
                if not in vocab word [
                    vocab/(word): current-index
                    current-index: current-index + 1
                ]
            ]
        ]
        
        ; Tokenize texts
        foreach text texts [
            words: parse text " "
            tokenized: make block! length? words
            foreach word words [
                append tokenized vocab/(word)
            ]
            append tokenized-texts tokenized
        ]
        
        make object! [
            vocab: vocab
            tokenized: tokenized-texts
            vocab-size: length? vocab
        ]
    ]
]