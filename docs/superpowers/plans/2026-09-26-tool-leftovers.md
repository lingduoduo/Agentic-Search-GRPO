# Plan: /tool disconnect + tool leftovers

1. Raw-ASGI disconnect test: red (the run is never cancelled once the response has ended), then add the `finally` → green
2. Mutation: without the `finally` it is red; removing only `task.cancel()` stays green, because `await task` under the cancel scope propagates. `cancel()` is kept for the GeneratorExit path
3. Fix the docstring example (run it), the web_search wording, and the two doc sentences
4. Full unit suite → 5170 passed
