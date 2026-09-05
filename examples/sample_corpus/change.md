---
doc_id: CHG-2407
title: Signing certificate maintenance
doc_type: change
owner_group: identity
visibility: internal
status: completed
---
# Signing certificate maintenance

## Change record
CHG-2407 schedules certificate rotation for the identity service. Record the
new certificate fingerprint and confirm application nodes refresh metadata.

## Verification
Test login on each node after certificate rotation. Investigate login
failures before removing the previous signing certificate.
