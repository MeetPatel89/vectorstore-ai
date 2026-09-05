# Security policy

This is a pre-1.0 project. Security fixes target the latest released version;
older versions have no maintenance guarantee. There is no promised response
SLA or completed independent security audit.

Please do not disclose suspected vulnerabilities in public issues. Use
[GitHub private vulnerability reporting](https://github.com/MeetPatel89/vectorstore-ai/security/advisories/new)
once the repository owner enables it. If that option is unavailable, open a
content-free issue asking the maintainer to enable a private reporting channel;
do not include exploit details or sensitive data until a private channel exists.

Include affected versions, a minimal synthetic reproduction, expected impact,
and suggested mitigations when available. Do not include credentials, private
documents, customer data, or production connection strings.

Applications remain responsible for authentication, supplying the correct
`RetrievalScope`, database permissions, and validating imported data. Unlabeled
catalog documents can be visible to all scopes; review that policy before
loading sensitive data. Retrieval fallback is an availability feature, not an
authorization bypass. Custom observers must avoid exporting document contents
and sensitive metadata. Treat model files and persisted indexes as trusted
inputs; do not load untrusted artifacts.

CI runs PR code with read-only repository permissions and no production
credentials. Only the post-validation release job receives write permission.
Release tags must be protected against unauthorized creation, mutation, and
deletion; see the owner checklist in [RELEASING](docs/RELEASING.md).
