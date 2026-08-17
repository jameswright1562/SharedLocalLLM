# Two-computer troubleshooting

Use the in-app diagnostics first: **Settings > Logs** shows redacted pairing, reconnect, cluster,
network, and benchmark events. **Open logs folder** opens the persistent
`shared-local-llm.log`. Logs omit prompts, images, credentials, and personal path prefixes.

The examples below call the machines **Computer A (Desktop/coordinator)** and **Computer B
(Laptop/worker)** only to make direction explicit. Your app may assign the opposite roles. Open
PowerShell normally unless a step explicitly says **Run as administrator**.

## The peer does not appear

1. On both computers, confirm the same SharedLocalLLM version is open. The authenticated peer
   listener and private-LAN discovery broadcaster run while the app is open.
2. Confirm both are on the same network (Ethernet, Wi-Fi, or a direct cable), with client isolation
   disabled on the router or access point if one is present.
3. Check the Windows network category on each computer.

**Computer A (Desktop)—inspect only:**

```powershell
Get-NetConnectionProfile | Format-Table Name, InterfaceAlias, NetworkCategory, IPv4Connectivity
```

**Computer B (Laptop)—inspect only:**

```powershell
Get-NetConnectionProfile | Format-Table Name, InterfaceAlias, NetworkCategory, IPv4Connectivity
```

The network category (Public/Private) is informational only; SharedLocalLLM pairs and launches
identically on every category. No profile change is required.

4. On the computer entering the six-digit code, fill in **Ethernet IPv4 address (optional)** with the
   address of the computer showing the code. The normal port is `49158`, so an address such as
   `192.168.1.20` or `169.254.20.8` is enough. Never forward this port on the router.
5. Confirm both installations run the same application and peer-protocol version.

If either computer was paired with version 0.1.1, upgrade both to 0.1.2. On each computer that still
lists the old peer, open **Nodes**, choose **Forget**, confirm the warning, and pair again. The reset
does not remove model sources or model files. After re-pairing, close one app, reopen it, and allow
one health-refresh interval for its status to return to Reachable.

The persistent log is at
`%LOCALAPPDATA%\SharedLocalLLM\logs\shared-local-llm.log`. Look for
`peer_listener_ready`, `peer_disconnected`, `peer_reconnected`, or a redacted actionable error.

To inspect IPv4 addresses:

**Computer A (Desktop)—inspect only:**

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Format-Table InterfaceAlias, IPAddress, PrefixLength
```

**Computer B (Laptop)—inspect only:**

```powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Format-Table InterfaceAlias, IPAddress, PrefixLength
```

On a direct cable with no router, a `169.254.x.x` address is expected. Use the address attached to
the Ethernet interface, not a Wi-Fi, VPN, loopback, or virtual-machine adapter. Manual entry skips
UDP discovery and connects directly to TCP port `49158`.

## Direct Ethernet cable (two computers, no router)

A direct cable between two computers has no router or DHCP server. Two address schemes work, and
neither requires a Windows network-profile change:

- **Static addressing:** set `10.10.10.1/24` on one computer and `10.10.10.2/24` on the other, with
  no gateway and no DNS. Then use the manual IPv4 entry field with the code-showing computer's
  `10.10.10.x` address.
- **Automatic link-local:** with no DHCP server, Windows assigns each adapter a `169.254.x.x`
  link-local address. That is valid; enter the code-showing computer's `169.254.x.x` Ethernet
  address manually.

Manual entry skips UDP discovery and connects directly to TCP port `49158`.

> WSL/Hyper-V note: virtual adapters such as `vEthernet (WSL)` advertise a fake 10 Gbps and used to
> win the "fastest link" pick. The app now skips virtual adapters (Hyper-V, WSL, loopback, VPN, TAP)
> when choosing the network path, so the physical Ethernet adapter is reported instead.

## Pairing codes do not match

Cancel on both computers. Do not approve either identity. Restart pairing from one computer and
compare the newly generated six-digit code in person. A mismatch can mean you selected a different
peer or traffic was intercepted. Deleting a trusted peer requires pairing it again; it does not
delete models or conversations.

## Firewall or connection failure

SharedLocalLLM runs elevated and creates a program-scoped Windows Firewall rule for TCP `49158` and
UDP `49157` across all network profiles (Profile Any). No network-profile change is needed. Do not
create a broad port rule manually.

Inspect the app's rules after repair:

**Computer A (Desktop)—inspect only:**

```powershell
Get-NetFirewallRule -DisplayName '*SharedLocalLLM*' |
  Format-Table DisplayName, Enabled, Profile, Direction, Action
```

**Computer B (Laptop)—inspect only:**

```powershell
Get-NetFirewallRule -DisplayName '*SharedLocalLLM*' |
  Format-Table DisplayName, Enabled, Profile, Direction, Action
```

If endpoint security blocks the executable, allow the signed SharedLocalLLM application through that
product's UI. Do not allow `ggml-rpc-server.exe`: it must remain loopback-only.

## Verify raw RPC is not exposed

While a distributed session is running, inspect listeners. Any `ggml-rpc-server` listener must have
local address `127.0.0.1` or `::1`; a LAN address or `0.0.0.0` is a security failure. Stop the cluster
and attach a redacted diagnostic bundle to an issue.

**Computer A (Desktop)—inspect only:**

```powershell
Get-NetTCPConnection -State Listen |
  Sort-Object LocalPort |
  Format-Table LocalAddress, LocalPort, OwningProcess
```

**Computer B (Laptop)—inspect only:**

```powershell
Get-NetTCPConnection -State Listen |
  Sort-Object LocalPort |
  Format-Table LocalAddress, LocalPort, OwningProcess
```

The inference API must likewise listen only on loopback. Its default port is 11435. The peer's app
listener is the only SharedLocalLLM service expected on a private LAN address.

## Network result is poor or uses the wrong adapter

Run the in-app bidirectional test again with VPNs disconnected. Compare its displayed local/remote
address and adapter with Windows' active adapters:

**Computer A (Desktop)—inspect only:**

```powershell
Get-NetAdapter |
  Where-Object Status -eq 'Up' |
  Format-Table Name, InterfaceDescription, LinkSpeed, MacAddress
```

**Computer B (Laptop)—inspect only:**

```powershell
Get-NetAdapter |
  Where-Object Status -eq 'Up' |
  Format-Table Name, InterfaceDescription, LinkSpeed, MacAddress
```

`LinkSpeed` is negotiation speed, not measured throughput. Use the app's sustained bidirectional
result for recommendations. Prefer wired Ethernet when available; on Wi-Fi, reduce distance and
contention. A Poor rating is a warning rather than a capacity failure, but distributed inference
may be much slower than one computer.

To inspect latency, substitute the peer IPv4 address shown in **Nodes**:

**Computer A (Desktop)—test toward Computer B:**

```powershell
Test-Connection -TargetName 192.168.1.20 -Count 20
```

**Computer B (Laptop)—test toward Computer A:**

```powershell
Test-Connection -TargetName 192.168.1.10 -Count 20
```

The example addresses are placeholders; do not copy them unless they match the addresses shown by
your installations.

## Models are missing

1. Open **Settings > Model sources** on the computer that actually stores the files and refresh it.
2. If LM Studio is installed, launch it once. CLI discovery follows LM Studio's configured catalogue;
   its folder does not have to be the default.
3. Add any custom parent directory with **Add folder**. The source is local to that computer.
4. Confirm the model uses `.gguf`. Other model formats are not indexed.
5. Keep all standard shards together. For vision, keep `mmproj*.gguf` beside the model/shards.

Check what LM Studio reports:

**Computer A (Desktop)—only if LM Studio is installed there:**

```powershell
lms ls --json --detailed
```

**Computer B (Laptop)—only if LM Studio is installed there:**

```powershell
lms ls --json --detailed
```

SharedLocalLLM never moves, renames, deletes, or downloads into a configured model folder.

## LM Studio is already using VRAM

SharedLocalLLM reports material VRAM use rather than terminating another application. Check LM
Studio's loaded models, save any work, and unload them from LM Studio (`lms unload --all`) if you
approve freeing that VRAM. This unloads models from memory; it does not delete them.

**Computer A (Desktop)—inspect LM Studio first:**

```powershell
lms ps --json
```

**Computer B (Laptop)—inspect LM Studio first:**

```powershell
lms ps --json
```

Only after you decide to free all LM Studio model allocations:

**Computer A (Desktop)—optional, user-approved change:**

```powershell
lms unload --all
```

**Computer B (Laptop)—optional, user-approved change:**

```powershell
lms unload --all
```

Do not end the LM Studio process in Task Manager. Refresh hardware status before launching.

## Runtime install or update fails

Open **Settings > Runtime** and read the exact stage: manifest, download, size, digest, extraction,
inventory, or health check. Never bypass a digest mismatch. Retry only after checking the system
clock, available disk space, and access to `github.com`. Repair reinstalls the pinned runtime; a
previous verified copy remains in the runtime `previous` folder when an install replaces `current`.

If security software quarantined an executable, verify the installer and runtime release provenance
before restoring it. Do not download replacement DLLs or executables from third-party DLL sites.

## API port 11435 is occupied

SharedLocalLLM does not choose a new port silently. Find the listener, decide whether it is expected,
then use **Settings > General** to change the local API port. Do not terminate an unknown process
merely to reclaim the default.

**Computer A (Desktop)—inspect only:**

```powershell
Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue |
  Format-Table LocalAddress, LocalPort, OwningProcess
```

**Computer B (Laptop)—inspect only:**

```powershell
Get-NetTCPConnection -State Listen -LocalPort 11435 -ErrorAction SilentlyContinue |
  Format-Table LocalAddress, LocalPort, OwningProcess
```

After selecting another port, regenerate client examples from the API page. Keep the bind address at
`127.0.0.1`; changing the port is not permission to expose the API to the LAN.

## A model does not fit or performs poorly

- Reduce context size before changing quantization; KV cache can materially affect memory use.
- Close other GPU-heavy applications and refresh capacity.
- Review single-node benchmark results. Distributed mode increases capacity but is not always faster.
- Keep CPU/RAM spill enabled only when needed; it can sharply reduce generation speed.
- Re-run the benchmark after a driver/runtime/model/network change. Old results should be invalidated
  automatically.

If benchmarking fails, expand the failed row. It should show the redacted executable error and profile
rather than omitting the result. Export diagnostics if that information is absent.

## Peer disconnects during generation

The current request cannot migrate. Wait for both apps to show **Ready**, then retry. A single-node
retry appears only when the selected model fits one machine. If either app still reports a running
cluster, use **Stop cluster** on Overview, Models, or Chat, wait for cleanup, and inspect
**Settings > Logs** before launching again.

Do not manually start `llama-server.exe` or `ggml-rpc-server.exe` on LAN interfaces as a workaround.
