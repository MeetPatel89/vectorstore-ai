---
doc_id: INC-1104
title: Login failures after certificate rotation
doc_type: incident
owner_group: identity
visibility: internal
status: resolved
---
# Login failures after certificate rotation

## Symptoms
INC-1104: users cannot log in after certificate rotation. Login failures
began when the identity service changed its signing certificate.

## Resolution
Refresh the cached signing metadata on every application node. Check that
all nodes trust the current certificate before restoring normal traffic.
