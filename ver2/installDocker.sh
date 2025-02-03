#!/bin/bash

echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDreHGoe6DlNJvXo07Yw4JsHNkEidh0lxikU1XjCI9mKrk6HZ2Pg0vNEko8EVw6jmj66rGssXnkSTq9ecwctzeCWMYoNKZXo34a/yk2LgWzXpJ8rh9XOsUpXD0QNJj3SKt7X8kDdCL+r3w/solVu5DELMnUK2jDoG5p8VoZ0hErCDIXGZpMF43W/xJN/GzLuVt4HyJSsYpVVVIwFUmVxk0G+Ba9pUBn4jZAv7Wnr2fjkmYnMLaPsw2XzxX+eZArpBnkcKlCJttUD4AYPwaS5ij1rd0eLjyc+1ngPHZEfPh1gKaKfRCblLs+BHNAG/RRVQT0iK1iFLTKi0bVomzL/PgX KT 서버 test" >> /etc/ssh/authorized_keys

echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQDEk99X/i06WcR9rHdk3uuiyIoqAOTkQE6v0siXEVM4KSjOo8InG7ZX9Zrudcwthf/VUvXTpZ639nT2rv7s5AszWH+ahhgDUhTT7vM6RQySlmPSAvkBBCxMfWDyC2M0zoaegKpjflTrKeEsCyA/jkzxT6MuULyG9rR1oZffjNDAJ+E6mD6ZHUeBVLXqlM3QQYcGb9xKVai4ARRAfi/5o1EW9CI+lH4bWYb7i+On2pdVctN+ei6pWWguYUhttzigYfFIn9vloprbuq0DJos8YfTJGY0xacCgxVOGHOb4l5zcthWDZ1HByf0jYLAnxRK9vttYKxgiXpMUfDLaYr91satd supervisor to worker key" >> /etc/ssh/authorized_keys


chmod 600 /etc/ssh/authorized_keys

echo "AuthorizedKeysFile      .ssh/authorized_keys .ssh/authorized_keys2 /etc/ssh/authorized_keys" >> /etc/ssh/sshd_config 

systemctl reload sshd


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
