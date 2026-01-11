REBOL [
    Title: "Covenant Logging System"
    Description: "Logging utilities for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Logging system for Covenant
logging: context [
    ;; Log levels
    DEBUG: 0
    INFO: 1
    WARNING: 2
    ERROR: 3
    CRITICAL: 4

    ;; Current log level
    current-level: INFO

    ;; Set log level
    set-level: func [
        "Set the minimum level for logging"
        level [integer!] "Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    ] [
        current-level: level
    ]

    ;; Log a message if level is sufficient
    log: func [
        "Log a message with specified level"
        level [integer!] "Log level"
        message [string!] "Message to log"
        /prefix "Add timestamp prefix"
    ] [
        if level >= current-level [
            output: either prefix [
                join (join "[" now/time "] ") message
            ] [
                message
            ]
            print output
        ]
    ]

    ;; Convenience functions for each level
    debug: func [
        "Log a debug message"
        message [string!] "Message to log"
        /prefix "Add timestamp prefix"
    ] [
        log DEBUG message prefix
    ]

    info: func [
        "Log an info message"
        message [string!] "Message to log"
        /prefix "Add timestamp prefix"
    ] [
        log INFO message prefix
    ]

    warning: func [
        "Log a warning message"
        message [string!] "Message to log"
        /prefix "Add timestamp prefix"
    ] [
        log WARNING message prefix
    ]

    error: func [
        "Log an error message"
        message [string!] "Message to log"
        /prefix "Add timestamp prefix"
    ] [
        log ERROR message prefix
    ]

    critical: func [
        "Log a critical message"
        message [string!] "Message to log"
        /prefix "Add timestamp prefix"
    ] [
        log CRITICAL message prefix
    ]
]