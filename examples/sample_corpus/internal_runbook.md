---
doc_id: RB-1001
title: Investigating login failures
doc_type: runbook
owner_group: identity
visibility: internal
status: active
---
# Investigating login failures

## Diagnosis
For login failures after certificate rotation, inspect signing metadata and
compare certificate fingerprints across application nodes.

## Recovery
Refresh stale caches, check the trust configuration, and verify login
through each node. Record the recovery steps in the incident timeline.
