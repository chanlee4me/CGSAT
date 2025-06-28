#!/bin/bash

# Training monitoring script

echo "=== CGSAT Training Monitor ==="
echo "Timestamp: $(date)"
echo ""

# Check if any training processes are running
echo "1. Checking for running training processes..."
PYTHON_PROCS=$(pgrep -f "python.*dqn.py" | wc -l)
echo "Found $PYTHON_PROCS Python training processes running"

if [ $PYTHON_PROCS -gt 0 ]; then
    echo ""
    echo "Running processes:"
    ps aux | grep "python.*dqn.py" | grep -v grep
fi

echo ""
echo "2. GPU Status:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv

echo ""
echo "3. Latest log files (last 5):"
if [ -d "/4T/chenli/data/log/logs" ]; then
    ls -lt /4T/chenli/data/log/logs/*.log 2>/dev/null | head -5
else
    echo "Log directory not found"
fi

echo ""
echo "4. Disk usage of log directory:"
if [ -d "/4T/chenli/data/log" ]; then
    du -sh /4T/chenli/data/log
else
    echo "Log directory not found"
fi

echo ""
echo "5. Recent model checkpoints:"
if [ -d "/4T/chenli/data/log" ]; then
    find /4T/chenli/data/log -name "*.chkp" -type f -exec ls -lt {} + 2>/dev/null | head -10
else
    echo "Log directory not found"
fi

echo ""
echo "6. System resource usage:"
echo "Memory usage:"
free -h
echo ""
echo "CPU usage (top 5 processes):"
ps aux --sort=-%cpu | head -6

echo ""
echo "=== Monitor Complete ==="
