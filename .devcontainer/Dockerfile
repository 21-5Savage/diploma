FROM mcr.microsoft.com/devcontainers/miniconda:1-3

ENV TZ="Asia/Ulaanbaatar"

USER root

RUN apt-get update \
 && apt-get install -y ca-certificates curl gnupg bash-completion \
 && rm -rf /var/lib/apt/lists/*

# NodeSource repo for current Node.js + npm
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_20.x nodistro main" \
   > /etc/apt/sources.list.d/nodesource.list

# Install Docker client tooling from Debian and Node.js from NodeSource.
# `nodejs` from NodeSource already includes npm.
RUN apt-get update \
 && apt-get install -y --no-install-recommends docker.io nodejs \
 && npm install -g @openai/codex@latest \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN echo '\nif [ -f /usr/share/bash-completion/bash_completion ]; then . /usr/share/bash-completion/bash_completion; fi' >> /etc/bash.bashrc

RUN echo '' >> /root/.bashrc && \
    echo '# VS Code shell integration' >> /root/.bashrc && \
    echo '[[ "$TERM_PROGRAM" == "vscode" ]] && . "$(code --locate-shell-integration-path bash)"' >> /root/.bashrc

# uncomment this to install ORACLE
# COPY .devcontainer/library-scripts/*.sh /tmp/library-scripts/
# RUN bash /tmp/library-scripts/oracle-instant-client-debian.sh \
#     && rm -rf /var/lib/apt/lists/* /tmp/library-scripts

COPY .devcontainer/environment.yml* /tmp/conda-tmp/
COPY requirements.txt /tmp/conda-tmp/requirements.txt

RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
      umask 0002 && conda env update -n vscode -f /tmp/conda-tmp/environment.yml; \
    fi \
    && rm -rf /tmp/conda-tmp
