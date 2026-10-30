---
id: M39
type: measurement
status: done
links: [Cluster brainstorm for low-budget scaling]
date: 2026-10-30
---
# Loopback latency floor for a 4 KB exchange (cluster feasibility)

Two Python processes on the Mac, TCP_NODELAY, 4 KB request/response, 30,000 repetitions, machine lightly loaded. Blocking sockets: p50 25 us, p99 105 us, p99.9 151 us, mean 34 us. Busy-polling non-blocking sockets: p50 21 us, p99 115 us, p99.9 242 us, mean 25 us. (A thread-based version was invalid: the interpreter lock made polling look 10x worse.) This is the software floor on one machine; a real LAN adds the NIC, switch and kernel network path (the cluster plan assumed 150-300 us round trips and needs p99 under about 0.3 ms for its 96 exchanges per token). So even loopback p99 is already 0.1 ms; two-machine numbers must be measured with the real adapters before buying anything.
