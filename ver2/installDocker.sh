#!/bin/bash

# Add Docker's official GPG key:
apt-get update -y -q
apt-get install -y -q ca-certificates curl 
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y -q

apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

DOCKER_CONFIG=${DOCKER_CONFIG:-$HOME/.docker}
mkdir -p $DOCKER_CONFIG/cli-plugins
curl -SL https://github.com/docker/compose/releases/download/v2.32.4/docker-compose-linux-x86_64 -o $DOCKER_CONFIG/cli-plugins/docker-compose
chmod +x $DOCKER_CONFIG/cli-plugins/docker-compose

#

echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDreHGoe6DlNJvXo07Yw4JsHNkEidh0lxikU1XjCI9mKrk6HZ2Pg0vNEko8EVw6jmj66rGssXnkSTq9ecwctzeCWMYoNKZXo34a/yk2LgWzXpJ8rh9XOsUpXD0QNJj3SKt7X8kDdCL+r3w/solVu5DELMnUK2jDoG5p8VoZ0hErCDIXGZpMF43W/xJN/GzLuVt4HyJSsYpVVVIwFUmVxk0G+Ba9pUBn4jZAv7Wnr2fjkmYnMLaPsw2XzxX+eZArpBnkcKlCJttUD4AYPwaS5ij1rd0eLjyc+1ngPHZEfPh1gKaKfRCblLs+BHNAG/RRVQT0iK1iFLTKi0bVomzL/PgX KT 서버 test"

chmod 600 /root/.ssh/authorized_keys
/sbin/restorecon ~/.ssh ~/.ssh/authorized_keys
