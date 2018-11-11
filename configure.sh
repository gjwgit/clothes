#!/bin/bash

# Ubuntu: Tested on Azure 16.04 DLVM

sudo apt-get install -y wajig > /dev/null
wajig update > /dev/null
wajig distupgrade -y > /dev/null
wajig install -y python-opencv python-requests python-matplotlib \
      python-scipy python-sklearn eom \
  | grep -v Reading \
  | grep -v Building \
  | grep -v 'newly installed'
