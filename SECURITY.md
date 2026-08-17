# Security policy

Please report suspected vulnerabilities privately through the repository's GitHub security advisory
form. Do not include model prompts, API keys, pairing material, certificates, personal paths, or an
unredacted diagnostic bundle in a public issue.

Only the latest released SharedLocalLLM version is supported with security fixes. This early
implementation has not completed physical two-computer acceptance and must not be exposed to the
internet or used as a multi-tenant service.

The upstream `llama.cpp` RPC backend is experimental and explicitly unsuitable for open or
untrusted networks. SharedLocalLLM's raw RPC process must remain bound to loopback behind the
authenticated application tunnel. A LAN-reachable raw RPC port, a non-loopback inference API, or
execution of a runtime whose digest was not verified should be treated as a security defect and the
cluster should be stopped immediately. The Windows network category (Public/Private/Domain) is
informational only and is not itself a security defect.
