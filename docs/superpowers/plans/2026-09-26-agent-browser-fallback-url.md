# Plan: /api/agent browser fallback URL

1. Tests: the loader reads the variable (None when unset), and `from_app_settings` carries it → red, 2 fail
2. Add the field, the loader line, and the `from_app_settings` line → green
3. Mutation check: drop the `from_app_settings` line → red
4. Update the configuration.md and request-routing.md rows that described the gap
5. Full unit suite → 5163 passed
