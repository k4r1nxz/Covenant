REBOL [
    Title: "Covenant Testing Framework"
    Description: "Unit testing framework for Covenant AI Framework"
    Version: 1.2.0
    Author: "Karina Mikhailovna Chernykh"
]

;; Testing framework for Covenant
testing: context [
    ;; Counter for test results
    passed: 0
    failed: 0
    total: 0

    ;; Reset counters
    reset-counters: func [] [
        passed: 0
        failed: 0
        total: 0
    ]

    ;; Assert function for testing
    assert-equal: func [
        "Assert that two values are equal"
        actual "Actual value"
        expected "Expected value"
        message [string!] "Test message"
    ] [
        total: total + 1
        if actual = expected [
            passed: passed + 1
            print ["✓ PASSED:" message]
        ] [
            failed: failed + 1
            print ["✗ FAILED:" message]
            print ["  Expected:" mold expected]
            print ["  Actual:" mold actual]
        ]
    ]

    ;; Assert that a value is true
    assert-true: func [
        "Assert that a value is true"
        condition "Condition to test"
        message [string!] "Test message"
    ] [
        total: total + 1
        if condition [
            passed: passed + 1
            print ["✓ PASSED:" message]
        ] [
            failed: failed + 1
            print ["✗ FAILED:" message]
            print ["  Expected: true"]
            print ["  Actual: false"]
        ]
    ]

    ;; Assert that a value is false
    assert-false: func [
        "Assert that a value is false"
        condition "Condition to test"
        message [string!] "Test message"
    ] [
        total: total + 1
        if not condition [
            passed: passed + 1
            print ["✓ PASSED:" message]
        ] [
            failed: failed + 1
            print ["✗ FAILED:" message]
            print ["  Expected: false"]
            print ["  Actual: true"]
        ]
    ]

    ;; Run a test suite
    run-tests: func [
        "Run a suite of tests"
        tests [block!] "Block of test functions to run"
    ] [
        reset-counters
        foreach test tests [
            do test
        ]
        print ["^nTest Results: " passed "passed, " failed "failed, " total "total"]
    ]

    ;; Report test results
    report: func [] [
        print ["^nFinal Test Report:"]
        print ["  Total Tests: " total]
        print ["  Passed:      " passed]
        print ["  Failed:      " failed]
        print ["  Success Rate:" (to-integer (passed * 100 / total)) "%"]
    ]
]