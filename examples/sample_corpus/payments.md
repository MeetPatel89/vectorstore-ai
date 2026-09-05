---
doc_id: INC-2201
title: Payment export reconciliation
doc_type: known_issue
owner_group: payments
visibility: customer_safe
status: active
---
# Payment export reconciliation

## Symptoms
Payment export totals do not match dashboard totals when their reporting
windows use different time zones.

## Resolution
Choose the same date range and time zone for the export and dashboard.
Regenerate the report after pending transactions have settled.
