REBOL [
    Title: "Covenant Data Loading Utilities"
    Description: "Data loading and preprocessing utilities for Covenant AI Framework"
    Version: 1.0.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Data loading utilities
data: context [
    ;; Dataset base class
    dataset: func [
        "Base dataset class"
        data-path [file!] "Path to data"
    ] [
        make object! [
            data-path: data-path
            data: none
            targets: none
            
            load-data: func [] [
                ; Load data from file
                raw-data: read/lines data-path
                parsed-data: make block! length? raw-data
                foreach line raw-data [
                    ; Parse each line as a block of numbers
                    append parsed-data load line
                ]
                self/data: parsed-data
            ]
            
            get-item: func [idx [integer!]] [
                ; Return data item at index
                make object! [
                    input: self/data/:idx/1
                    target: self/data/:idx/2
                ]
            ]
            
            length: func [] [
                length? self/data
            ]
        ]
    ]
    
    ;; DataLoader class for batching and shuffling
    dataloader: func [
        "DataLoader for batching and shuffling datasets"
        dataset [object!] "Dataset to load from"
        batch-size [integer!] "Size of each batch"
        /shuffle "Whether to shuffle data"
    ] [
        make object! [
            dataset: dataset
            batch-size: batch-size
            shuffle: either shuffle [true] [false]
            indices: make block! 0
            
            init: does [
                ; Initialize indices
                repeat i dataset/length [
                    append self/indices i
                ]
                if self/shuffle [
                    ; Simple shuffle implementation
                    shuffled: make block! length? self/indices
                    remaining: copy self/indices
                    while [not empty? remaining] [
                        idx: random length? remaining
                        append shuffled take/last/part remaining idx
                    ]
                    self/indices: shuffled
                ]
            ]
            
            ; Iterator functions
            current-index: 1
            
            next: func [
                "Get the next batch of data"
            ] [
                if self/current-index > length? self/indices [
                    return none  ; End of data
                ]
                
                batch-indices: make block! self/batch-size
                end-idx: min (self/current-index + self/batch-size - 1) length? self/indices
                
                repeat i (end-idx - self/current-index + 1) [
                    append batch-indices self/indices/(self/current-index + i - 1)
                ]
                
                self/current-index: end-idx + 1
                
                ; Get data for batch indices
                batch-inputs: make block! length? batch-indices
                batch-targets: make block! length? batch-indices
                
                foreach idx batch-indices [
                    item: self/dataset/get-item idx
                    append batch-inputs item/input
                    append batch-targets item/target
                ]
                
                make object! [
                    inputs: batch-inputs
                    targets: batch-targets
                    size: length? batch-indices
                ]
            ]
            
            reset: func [
                "Reset the dataloader to the beginning"
            ] [
                self/current-index: 1
                if self/shuffle [
                    ; Re-shuffle on reset if shuffle is enabled
                    shuffled: make block! length? self/indices
                    remaining: copy self/indices
                    while [not empty? remaining] [
                        idx: random length? remaining
                        append shuffled take/last/part remaining idx
                    ]
                    self/indices: shuffled
                ]
            ]
        ]
    ]
    
    ;; TensorDataset - a dataset for tensor data
    tensordataset: func [
        "Dataset for tensor data"
        tensors [block!] "Block of tensors [inputs, targets]"
    ] [
        inputs: first tensors
        targets: second tensors
        
        make object! [
            inputs: inputs
            targets: targets
            
            get-item: func [idx [integer!]] [
                make object! [
                    input: core/tensor reduce [self/inputs/:idx]
                    target: core/tensor reduce [self/targets/:idx]
                ]
            ]
            
            length: func [] [
                length? self/inputs
            ]
        ]
    ]
    
    ;; Transformations
    transforms: context [
        ;; Normalize tensor values
        normalize: func [
            "Normalize tensor with mean and std"
            tensor [object!]
            mean [number!]
            std [number!]
        ] [
            normalized-data: copy tensor/data
            repeat i length? normalized-data [
                normalized-data/:i: (normalized-data/:i - mean) / std
            ]
            make object! [
                data: normalized-data
                shape: tensor/shape
                dtype: tensor/dtype
            ]
        ]
        
        ;; Scale tensor values to range [0, 1]
        to-range: func [
            "Scale tensor values to [0, 1] range"
            tensor [object!]
        ] [
            ; Find min and max
            min-val: first tensor/data
            max-val: first tensor/data
            foreach val tensor/data [
                if val < min-val [min-val: val]
                if val > max-val [max-val: val]
            ]
            
            range: max-val - min-val
            if range = 0.0 [range: 1.0]  ; Avoid division by zero
            
            scaled-data: copy tensor/data
            repeat i length? scaled-data [
                scaled-data/:i: (scaled-data/:i - min-val) / range
            ]
            
            make object! [
                data: scaled-data
                shape: tensor/shape
                dtype: tensor/dtype
            ]
        ]
    ]
]

;; The data loading functionality is now integrated directly in the utils module
;; This file provides the data structures and utilities