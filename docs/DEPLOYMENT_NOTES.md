# Deployment Notes

Build a small review app:

- upload COCO annotation JSON
- run QA checks
- display flagged annotations
- export corrected review queue

Serving notes:

- never expose private images without permission
- preserve original annotation IDs
- store QA flags separately from source annotations
- keep thresholds configurable
