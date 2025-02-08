#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./runner.sh [vsp|mission]"
    exit 1
fi

if [ "$1" != "vsp" ] && [ "$1" != "mission" ]; then
    echo "First argument must be either 'vsp' or 'mission'"
    exit 1
fi

# Set environment variables for docker-compose
export MODE=$1
export IMAGE=jaewonchung7snu/bulnabi_aiaa_container:latest	

echo "Starting workers in $MODE mode using image $IMAGE..."
docker-compose up
