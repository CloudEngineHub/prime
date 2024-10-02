#!/bin/bash

# Check if the script is run as root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

# Define variables
INTERFACE="lo"  # localhost interface
RATE="500mbit"  # 500 Mbps
BURST="500k"    # burst size
LATENCY="50ms"  # maximum latency

# Remove any existing traffic control rules on the interface
tc qdisc del dev $INTERFACE root 2>/dev/null

# Add the rate limiting rule
tc qdisc add dev $INTERFACE root tbf rate $RATE burst $BURST latency $LATENCY

echo "Bandwidth limit of $RATE has been set on $INTERFACE"

# To remove the limit, run:
# tc qdisc del dev $INTERFACE root