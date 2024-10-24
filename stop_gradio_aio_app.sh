#!/bin/bash

kill $(pgrep -f "^python ${WORKSPACE}/app/app.py$")