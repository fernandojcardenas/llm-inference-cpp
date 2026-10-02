# Builds llmi-server's Release binary and nothing else -- no model weights
# are baked in (this project never commits weights, see THIRD-PARTY.md and
# tools/fetch_model.sh); a model directory is mounted at `docker run` time.
# See docs/server.md for the full run invocation and ADR 0008 for why the
# server binds loopback by default.

FROM debian:12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
      cmake ninja-build g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt THIRD-PARTY.md ./
COPY include include
COPY src src
COPY apps apps
COPY third_party third_party
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLMI_BUILD_TESTS=OFF \
    && cmake --build build --target llmi-server -j "$(nproc)"

FROM debian:12-slim
RUN useradd --system --no-create-home --shell /usr/sbin/nologin llmi
COPY --from=build /src/build/llmi-server /usr/local/bin/llmi-server
USER llmi
# Loopback by default (ADR 0008, and llmi-server's own ServerConfig
# default): map with `-p 127.0.0.1:8080:8080` to keep that guarantee on the
# host, not just inside the container. The model directory is the
# required first argument, mounted as a volume -- there is no default for
# it, so `docker run ... llmi-server /model [extra flags]`.
EXPOSE 8080
ENTRYPOINT ["llmi-server"]
