FROM mcr.microsoft.com/devcontainers/miniconda:1-3

ENV TZ="Asia/Ulaanbaatar"

USER root

RUN apt-get update \
  && apt-get install -y ca-certificates curl gnupg \
  && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /etc/apt/keyrings \
  && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg \
  && echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  > /etc/apt/sources.list.d/docker.list

RUN apt-get update \
  && apt-get install -y docker-ce-cli \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get update \
  && apt-get install -y bash-completion \
  && rm -rf /var/lib/apt/lists/*

RUN echo '\nif [ -f /usr/share/bash-completion/bash_completion ]; then . /usr/share/bash-completion/bash_completion; fi' >> /etc/bash.bashrc
RUN echo '' >> ~/.bashrc && \
  echo '# VS Code shell integration' >> ~/.bashrc && \
  echo '[[ "$TERM_PROGRAM" == "vscode" ]] && . "$(code --locate-shell-integration-path bash)"' >> ~/.bashrc
# uncomment this to install ORACLE
# COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/

# RUN bash /tmp/library-scripts/oracle-instant-client-debian.sh \
#     && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

COPY .devcontainer/environment.yml* /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
  umask 0002 && conda env update -n vscode -f /tmp/conda-tmp/environment.yml; \
  fi \
  && rm -rf /tmp/conda-tmp

RUN echo "conda activate vscode" >> ~/.bashrc
