---
doc_id: KB-429
title: API request limits
doc_type: known_issue
owner_group: api
visibility: customer_safe
status: active
---
# API request limits

## Symptoms
Business tier API clients receive too many requests errors when a burst
exceeds their request allowance.

## Resolution
Honor the retry delay, reduce request concurrency, and retry with bounded
backoff. Contact support if the required workload exceeds the tier allowance.
