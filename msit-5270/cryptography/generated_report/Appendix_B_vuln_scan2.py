"""
Vulnerability Scanning Automation Tool
MedSecure Regional Health Network — Week 6 Ethical Hacking Module

A Python wrapper around nmap that performs host discovery and service/version
scanning, OR ingests an existing nmap/Zenmap XML report, classifies discovered
services by risk, and generates both a human-readable text report and a
structured JSON report.

Dependencies: nmap (system binary) for live scans. Install via:
  Ubuntu/Debian: sudo apt install nmap
  macOS:         brew install nmap
  Windows:       https://nmap.org/download.html
For XML ingestion mode, nmap is not required.

Run modes:
  Live scan:    python vuln_scan.py <target>
  XML ingest:   python vuln_scan.py --input-xml <file.xml>

Examples:
  python vuln_scan.py 192.168.1.0/24
  python vuln_scan.py scanme.nmap.org --profile quick
  python vuln_scan.py --input-xml zenmap_scan.xml
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


# ----------------------------------------------------------------------
# Risk classification rules
# ----------------------------------------------------------------------
# Maps service / port indicators to a risk tier and a short rationale.
# In production this would be backed by a CVE feed; for a course
# deliverable, these heuristics illustrate the analysis layer.

RISK_RULES = [
    # (matcher, risk, rationale)
    (lambda p, s: p == 23,        "HIGH",   "Telnet — cleartext credentials, deprecated"),
    (lambda p, s: p == 21,        "HIGH",   "FTP — cleartext credentials unless FTPS"),
    (lambda p, s: p == 445,       "HIGH",   "SMB — historical target of EternalBlue/ransomware"),
    (lambda p, s: p == 3389,      "HIGH",   "RDP — frequent ransomware entry vector"),
    (lambda p, s: p == 139,       "MEDIUM", "NetBIOS — legacy SMB sessions"),
    (lambda p, s: p == 135,       "MEDIUM", "MS-RPC — endpoint mapper, often unneeded externally"),
    (lambda p, s: p == 80,        "MEDIUM", "HTTP — unencrypted; verify redirect to HTTPS"),
    (lambda p, s: p == 22,        "LOW",    "SSH — encrypted, but verify key auth and version"),
    (lambda p, s: p == 443,       "LOW",    "HTTPS — verify TLS 1.2+ and cipher suites"),
    (lambda p, s: p == 53,        "LOW",    "DNS — confirm not an open resolver"),
    (lambda p, s: "telnet" in s,  "HIGH",   "Telnet service detected"),
    (lambda p, s: "ftp" in s and "sftp" not in s, "HIGH", "FTP service detected"),
]


def classify(port: int, service: str) -> tuple[str, str]:
    """Return (risk_level, rationale) for a port/service combination."""
    service_lower = service.lower()
    for matcher, risk, rationale in RISK_RULES:
        if matcher(port, service_lower):
            return risk, rationale
    return "INFO", "No specific rule; review service and version"


# ----------------------------------------------------------------------
# Scan execution
# ----------------------------------------------------------------------

def check_nmap() -> None:
    """Ensure nmap is available on PATH before doing anything else."""
    if shutil.which("nmap") is None:
        print("[ERROR] nmap is not installed or not on PATH.")
        print("        Install it from https://nmap.org/download.html")
        sys.exit(1)


def run_nmap(target: str, scan_profile: str = "default") -> str:
    """Run nmap with a service/version scan and return raw stdout."""
    profiles = {
        # -sV  service/version detection
        # -T4  faster timing template
        # -Pn  treat host as online (skip ping; matters for hosts that drop ICMP)
        # --top-ports limits scope for a course-friendly demo
        "default": ["-sV", "-T4", "-Pn", "--top-ports", "100"],
        "quick":   ["-sV", "-T4", "-Pn", "--top-ports", "20"],
        "full":    ["-sV", "-T4", "-Pn", "-p-"],
    }
    args = profiles.get(scan_profile, profiles["default"])
    cmd = ["nmap", *args, target]
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[ERROR] nmap exited with code {result.returncode}")
        print(result.stderr)
        sys.exit(1)
    return result.stdout


# ----------------------------------------------------------------------
# Output parsing
# ----------------------------------------------------------------------

HOST_RE = re.compile(r"Nmap scan report for (.+)")
PORT_RE = re.compile(r"^(\d+)/(tcp|udp)\s+(\S+)\s+(\S+)(?:\s+(.*))?$")


def parse_nmap_output(raw: str) -> list[dict]:
    """Parse raw nmap stdout into a list of host records."""
    hosts: list[dict] = []
    current: dict | None = None

    for line in raw.splitlines():
        line = line.rstrip()
        host_match = HOST_RE.match(line)
        if host_match:
            if current is not None:
                hosts.append(current)
            current = {"host": host_match.group(1).strip(), "ports": []}
            continue
        if current is None:
            continue
        port_match = PORT_RE.match(line)
        if port_match:
            state = port_match.group(3)
            # Only report ports that are actually open. Filtered/closed
            # ports are not accessible services and would inflate findings.
            if state != "open":
                continue
            port = int(port_match.group(1))
            proto = port_match.group(2)
            service = port_match.group(4)
            version = (port_match.group(5) or "").strip()
            risk, rationale = classify(port, service)
            current["ports"].append({
                "port": port,
                "protocol": proto,
                "state": state,
                "service": service,
                "version": version,
                "risk": risk,
                "rationale": rationale,
            })

    if current is not None:
        hosts.append(current)
    return hosts


def parse_nmap_xml(xml_path: str) -> list[dict]:
    """Parse an nmap/Zenmap XML output file into a list of host records.

    Compatible with XML produced by:
      - nmap -oX output.xml ...
      - Zenmap: Scan -> Save Scan (XML format)
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"[ERROR] Failed to parse XML: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"[ERROR] XML file not found: {xml_path}")
        sys.exit(1)

    root = tree.getroot()
    hosts: list[dict] = []

    for host_elem in root.findall("host"):
        # Prefer hostname, fall back to IP address
        hostname = None
        hostnames_elem = host_elem.find("hostnames")
        if hostnames_elem is not None:
            hn = hostnames_elem.find("hostname")
            if hn is not None:
                hostname = hn.get("name")
        if not hostname:
            address_elem = host_elem.find("address")
            hostname = address_elem.get("addr") if address_elem is not None else "unknown"

        host_record: dict = {"host": hostname, "ports": []}

        ports_elem = host_elem.find("ports")
        if ports_elem is None:
            hosts.append(host_record)
            continue

        for port_elem in ports_elem.findall("port"):
            state_elem = port_elem.find("state")
            if state_elem is None or state_elem.get("state") != "open":
                continue

            port = int(port_elem.get("portid"))
            proto = port_elem.get("protocol", "tcp")
            service_elem = port_elem.find("service")
            service = (service_elem.get("name") if service_elem is not None
                       else "unknown")
            # Build version string from name, product, version, extrainfo
            version_parts = []
            if service_elem is not None:
                for attr in ("product", "version", "extrainfo"):
                    val = service_elem.get(attr)
                    if val:
                        version_parts.append(val)
            version = " ".join(version_parts)

            risk, rationale = classify(port, service)
            host_record["ports"].append({
                "port": port,
                "protocol": proto,
                "state": "open",
                "service": service,
                "version": version,
                "risk": risk,
                "rationale": rationale,
            })
        hosts.append(host_record)

    return hosts


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------

RISK_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}


def summarize(hosts: list[dict]) -> dict:
    """Roll up findings into summary counts."""
    summary = {"hosts": len(hosts), "open_ports": 0,
               "HIGH": 0, "MEDIUM": 0, "LOW": 0, "INFO": 0}
    for h in hosts:
        for p in h["ports"]:
            summary["open_ports"] += 1
            summary[p["risk"]] += 1
    return summary


def generate_text_report(hosts: list[dict], target: str,
                         summary: dict, scan_started: str,
                         scan_ended: str) -> str:
    """Build a human-readable report."""
    out = []
    add = out.append
    add("=" * 72)
    add("  MRHN Vulnerability Scan Report")
    add("=" * 72)
    add(f"  Target:        {target}")
    add(f"  Scan started:  {scan_started}")
    add(f"  Scan finished: {scan_ended}")
    add("")
    add("  Summary")
    add("  -------")
    add(f"  Hosts scanned:      {summary['hosts']}")
    add(f"  Open ports found:   {summary['open_ports']}")
    add(f"  HIGH risk findings: {summary['HIGH']}")
    add(f"  MEDIUM risk:        {summary['MEDIUM']}")
    add(f"  LOW risk:           {summary['LOW']}")
    add(f"  Informational:      {summary['INFO']}")
    add("")
    add("=" * 72)
    add("  Findings by Host")
    add("=" * 72)

    for host in hosts:
        add("")
        add(f"  Host: {host['host']}")
        add("  " + "-" * 70)
        if not host["ports"]:
            add("    No open ports detected.")
            continue
        sorted_ports = sorted(host["ports"],
                              key=lambda p: (RISK_ORDER[p["risk"]], p["port"]))
        add(f"    {'PORT':<10}{'RISK':<10}{'SERVICE':<14}{'VERSION'}")
        for p in sorted_ports:
            port_label = f"{p['port']}/{p['protocol']}"
            add(f"    {port_label:<10}{p['risk']:<10}"
                f"{p['service']:<14}{p['version'][:36]}")
        add("")
        add("    Risk rationale:")
        for p in sorted_ports:
            add(f"      [{p['risk']:<6}] {p['port']}/{p['protocol']}: {p['rationale']}")
    add("")
    add("=" * 72)
    add("  Recommended Next Steps")
    add("=" * 72)
    add("  1. Disable or restrict any HIGH-risk legacy services (Telnet, FTP, SMB).")
    add("  2. Verify TLS configuration on any HTTPS service (cipher suites, version).")
    add("  3. Apply patches to services flagged with outdated versions.")
    add("  4. Re-run scan after remediation to confirm closure.")
    add("=" * 72)
    return "\n".join(out)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MRHN Vulnerability Scanning Automation Tool"
    )
    parser.add_argument("target", nargs="?",
                        help="Target host, range, or CIDR (e.g. 192.168.1.0/24). "
                             "Omit when using --input-xml.")
    parser.add_argument("--input-xml",
                        help="Path to an existing nmap/Zenmap XML output file. "
                             "When set, skips running nmap and analyzes the XML.")
    parser.add_argument("--profile", default="default",
                        choices=["quick", "default", "full"],
                        help="Scan depth profile (default: default). "
                             "Ignored when --input-xml is set.")
    parser.add_argument("--out", default="scan_report",
                        help="Output filename stem (default: scan_report)")
    args = parser.parse_args()

    # Validate input combination
    if args.input_xml and args.target:
        print("[WARN] Both target and --input-xml provided; ignoring target.")
    if not args.input_xml and not args.target:
        parser.error("Provide either a target or --input-xml <file>.")

    if args.input_xml:
        # XML ingestion mode — no nmap required.
        print(f"[INFO] Reading nmap XML report: {args.input_xml}")
        scan_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hosts = parse_nmap_xml(args.input_xml)
        scan_ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_label = f"XML import: {args.input_xml}"
        profile_label = "xml"
    else:
        # Live scan mode.
        check_nmap()
        scan_started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[INFO] Scan started at {scan_started}")
        raw = run_nmap(args.target, args.profile)
        scan_ended = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hosts = parse_nmap_output(raw)
        target_label = args.target
        profile_label = args.profile

    summary = summarize(hosts)
    text_report = generate_text_report(
        hosts, target_label, summary, scan_started, scan_ended
    )

    # Write outputs
    txt_path = Path(f"{args.out}.txt")
    json_path = Path(f"{args.out}.json")
    txt_path.write_text(text_report)
    json_path.write_text(json.dumps({
        "target": target_label,
        "profile": profile_label,
        "scan_started": scan_started,
        "scan_ended": scan_ended,
        "summary": summary,
        "hosts": hosts,
    }, indent=2))

    # Print to terminal too
    print()
    print(text_report)
    print()
    print(f"[OK] Text report:  {txt_path.resolve()}")
    print(f"[OK] JSON report:  {json_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user.")
        sys.exit(0)
