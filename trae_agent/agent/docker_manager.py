import contextlib
import os
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError

import pexpect
from docker.errors import DockerException, ImageNotFound, NotFound

import docker


class DockerManager:
    """
    Manages Docker container lifecycle and command execution for the agent.
    Supports both stateless (non-interactive) and stateful (interactive) modes.
    """

    CONTAINER_TOOLS_PATH = "/agent_tools"

    def __init__(
        self,
        image: str | None,
        container_id: str | None,
        dockerfile_path: str | None,
        docker_image_file: str | None,
        workspace_dir: str | None = None,
        tools_dir: str | None = None,
        interactive: bool = False,
        security_options: dict | None = None,
    ):
        if not image and not container_id and not dockerfile_path and not docker_image_file:
            raise ValueError(
                "Either a Docker image or a container ID or a dockerfile path or a docker image file (tar) must be provided."
            )
        self.client = docker.from_env()
        self.image = image
        self.container_id = container_id
        self.dockerfile_path = dockerfile_path
        self.docker_image_file = docker_image_file
        self.workspace_dir = workspace_dir
        self.tools_dir = tools_dir
        self.interactive = interactive
        self.security_options = dict(security_options or {})
        self.container_workspace = "/workspace"
        self.container = None
        self.shell = None
        self._is_managed = True
        self._tools_mounted = False

    def start(self):
        """Starts/attaches to the container, mounts the workspace, copies tools, and starts the shell."""
        try:
            if self.dockerfile_path:
                if not os.path.isabs(self.dockerfile_path):
                    raise ValueError("Dockerfile path must be an absolute path.")
                build_context = os.path.dirname(self.dockerfile_path)
                dockerfile_name = os.path.basename(self.dockerfile_path)
                unique_tag = f"trae-agent-custom:{uuid.uuid4()}"
                print(
                    f"Building Docker image from '{self.dockerfile_path}' with tag '{unique_tag}'..."
                )
                try:
                    new_image, build_logs = self.client.images.build(
                        path=build_context, dockerfile=dockerfile_name, tag=unique_tag, rm=True
                    )
                    self.image = new_image.tags[0]
                    print(f"✅ Successfully built image: {self.image}")
                except Exception as e:
                    print("[red]❌ Docker image build failed. See logs below:[/red]")
                    for log_line in e.build_log:
                        if "stream" in log_line:
                            print(log_line["stream"].strip())
                    raise

            elif self.docker_image_file:
                print(f"Loading Docker image from file '{self.docker_image_file}'...")
                try:
                    with open(self.docker_image_file, "rb") as f:
                        loaded_images = self.client.images.load(f.read())
                    if not loaded_images:
                        raise DockerException("Failed to load any images from the provided file.")
                    self.image = loaded_images[0].tags[0]
                    print(f"✅ Successfully loaded image: {self.image}")
                except FileNotFoundError:
                    raise
                except Exception as e:
                    raise DockerException(f"Error loading image from file: {e}") from e

            if self.container_id:
                if self.security_options:
                    raise DockerException(
                        "cannot enforce a sandbox profile on an existing container"
                    )
                print(f"Attaching to existing container: {self.container_id}...")
                self.container = self.client.containers.get(self.container_id)
                self._is_managed = False
                print(f"Successfully attached to container {self.container.short_id}.")
            elif self.image:
                print(f"Starting a new container from image: {self.image}...")
                if self.workspace_dir is not None:
                    os.makedirs(self.workspace_dir, exist_ok=True)
                    volumes = {
                        os.path.abspath(self.workspace_dir): {
                            "bind": self.container_workspace,
                            "mode": "rw",
                        }
                    }
                    if (
                        self.security_options.get("mount_tools_read_only")
                        and self.tools_dir
                        and os.path.isdir(self.tools_dir)
                    ):
                        volumes[os.path.abspath(self.tools_dir)] = {
                            "bind": self.CONTAINER_TOOLS_PATH,
                            "mode": "ro",
                        }
                        self._tools_mounted = True
                    run_options = {
                        key: value for key, value in self.security_options.items()
                        if key != "mount_tools_read_only"
                    }
                    self.container = self.client.containers.run(
                        self.image,
                        command="sleep infinity",
                        detach=True,
                        volumes=volumes,
                        working_dir=self.container_workspace,
                        **run_options,
                    )
                    self.container_id = self.container.id
                    self._is_managed = True
                    print(
                        f"Container {self.container.short_id} created. Workspace '{self.workspace_dir}' is mounted to '{self.container_workspace}'."
                    )
                else:
                    run_options = {
                        key: value for key, value in self.security_options.items()
                        if key != "mount_tools_read_only"
                    }
                    self.container = self.client.containers.run(
                        self.image,
                        command="sleep infinity",
                        detach=True,
                        working_dir=self.container_workspace,
                        **run_options,
                    )
                    self.container_id = self.container.id
                    self._is_managed = True
                    print(f"Container {self.container.short_id} created.")
            self._copy_tools_to_container()
            if self.security_options:
                self._verify_security_options()
            else:
                self._start_persistent_shell()
        except (ImageNotFound, NotFound, DockerException) as e:
            if self.container and self._is_managed:
                with contextlib.suppress(DockerException):
                    self.stop()
            print(f"[red]Failed to start DockerManager: {e}[/red]")
            raise

    def execute(self, command: str, timeout: int = 300) -> tuple[int, str]:
        """
        Executes a command using the configured mode (interactive or stateless).
        """
        if not self.container:
            raise RuntimeError("Container is not running. Call start() first.")

        if self.security_options:
            return self._execute_stateless(command, timeout)
        return self._execute_interactive(command, timeout)

    def stop(self):
        """Stops the pexpect shell and cleans up the container if managed by this instance."""
        if self.shell and self.shell.isalive():
            print("Closing persistent shell...")
            self.shell.close(force=True)
            self.shell = None

        if self.container and self._is_managed:
            print(f"Stopping and removing managed container {self.container.short_id}...")
            try:
                self.container.stop()
                self.container.remove()
                print("Container cleaned up successfully.")
            except DockerException as e:
                container_id = self.container.short_id
                self.container = None
                raise DockerException(
                    f"failed to stop/remove managed container {container_id}: {e}"
                ) from e

        self.container = None

    # --- Private Helper Methods ---

    def _verify_security_options(self):
        """Fail closed unless Docker reports the requested isolation controls."""
        self.container.reload()
        attrs = self.container.attrs
        host = attrs.get("HostConfig", {})
        config = attrs.get("Config", {})
        expected = self.security_options
        memory = expected.get("mem_limit")
        expected_memory = (
            int(str(memory)[:-1]) * 1024 * 1024
            if isinstance(memory, str) and memory.lower().endswith("m") else int(memory or 0)
        )
        checks = {
            "non-root user": str(config.get("User", "")) == str(expected.get("user", "")),
            "network disabled": bool(host.get("NetworkMode") == "none")
            if expected.get("network_disabled") else True,
            "read-only rootfs": bool(host.get("ReadonlyRootfs"))
            if expected.get("read_only") else True,
            "capabilities dropped": "ALL" in (host.get("CapDrop") or [])
            if expected.get("cap_drop") else True,
            "pid limit": int(host.get("PidsLimit") or 0) == int(expected.get("pids_limit") or 0)
            if expected.get("pids_limit") else True,
            "memory limit": int(host.get("Memory") or 0) == expected_memory
            if expected_memory else True,
            "cpu limit": int(host.get("NanoCpus") or 0) == int(expected.get("nano_cpus") or 0)
            if expected.get("nano_cpus") else True,
            "no new privileges": "no-new-privileges:true" in (host.get("SecurityOpt") or [])
            if expected.get("security_opt") else True,
            "tmpfs isolation": bool((host.get("Tmpfs") or {}).get("/tmp"))
            if expected.get("tmpfs") else True,
        }
        mounts = attrs.get("Mounts", [])
        if self.workspace_dir:
            workspace_source = os.path.abspath(self.workspace_dir)
            checks["workspace rw mount"] = any(
                os.path.abspath(str(mount.get("Source", ""))) == workspace_source
                and mount.get("Destination") == self.container_workspace
                and bool(mount.get("RW"))
                for mount in mounts
            )
        if expected.get("mount_tools_read_only") and self.tools_dir:
            tools_source = os.path.abspath(self.tools_dir)
            checks["tools ro mount"] = any(
                os.path.abspath(str(mount.get("Source", ""))) == tools_source
                and mount.get("Destination") == self.CONTAINER_TOOLS_PATH
                and not bool(mount.get("RW"))
                for mount in mounts
            )
        checks["docker socket absent"] = not any(
            str(mount.get("Destination", "")).endswith("docker.sock")
            or str(mount.get("Source", "")).endswith("docker.sock")
            for mount in mounts
        )
        allowed_destinations = {self.container_workspace, self.CONTAINER_TOOLS_PATH}
        checks["no extra host bind mounts"] = all(
            mount.get("Type") != "bind" or mount.get("Destination") in allowed_destinations
            for mount in mounts
        )
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            self.stop()
            raise DockerException(
                "container inspect did not confirm security controls: " + ", ".join(failed)
            )

    def _copy_tools_to_container(self):
        """Copies the local tools directory to a fixed path inside the container."""
        if self._tools_mounted:
            return
        if not self.tools_dir or not os.path.isdir(self.tools_dir):
            print(
                f"[yellow]Packaged tools directory '{self.tools_dir}' not provided or not found, skipping copy.[/yellow]"
            )
            return

        print(
            f"Copying tools from '{self.tools_dir}' to container path '{self.CONTAINER_TOOLS_PATH}'..."
        )
        try:
            cmd = f"docker cp '{os.path.abspath(self.tools_dir)}' '{self.container.id}:{self.CONTAINER_TOOLS_PATH}'"
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            print("Tools copied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[red]Failed to copy tools to container: {e.stderr.decode()}[/red]")
            raise DockerException(f"Failed to copy tools: {e.stderr.decode()}") from e

    def _start_persistent_shell(self):
        """Spawns a persistent bash shell inside the container using pexpect."""
        if not self.container:
            return
        # print("Starting persistent shell for interactive mode...")
        try:
            command = f"docker exec -it {self.container.id} /bin/bash"
            self.shell = pexpect.spawn(command, encoding="utf-8", timeout=120)
            self.shell.expect([r"\$", r"#"], timeout=120)
            print("Persistent shell is ready.")
        except pexpect.exceptions.TIMEOUT:
            print(
                "[red]Timeout waiting for shell prompt. The container might be slow to start or misconfigured.[/red]"
            )
            raise

    def _execute_stateless(self, command: str, timeout: int) -> tuple[int, str]:
        """Execute through the SDK without an interactive pexpect shell."""
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            self.container.exec_run,
            cmd=["/bin/sh", "-lc", command],
            workdir=self.container_workspace,
            demux=True,
        )
        try:
            result = future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            # Removing the worker container is the only reliable way to reclaim all
            # descendants of a timed-out exec.
            self.stop()
            return -1, f"Error: command timed out after {timeout} seconds; container destroyed"
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        stdout, stderr = result.output if result.output else (b"", b"")
        output = (stdout or b"").decode("utf-8", "replace")
        error = (stderr or b"").decode("utf-8", "replace")
        return int(result.exit_code), (output + error).strip()

    def _execute_interactive(self, command: str, timeout: int) -> tuple[int, str]:
        """Executes a command within the existing persistent shell."""
        if not self.shell or not self.shell.isalive():
            print("[yellow]Shell not found or died. Attempting to restart...[/yellow]")
            self._start_persistent_shell()

        if self.shell is None:
            raise RuntimeError("Failed to start or restart the persistent shell.")

        marker = "---CMD_DONE---"
        full_command = command.strip()
        marker_command = f"echo {marker}$?"
        self.shell.sendline(full_command)
        self.shell.sendline(marker_command)
        try:
            self.shell.expect(marker + r"(\d+)", timeout=timeout)
        except pexpect.exceptions.TIMEOUT:
            return (
                -1,
                f"Error: Command '{command}' timed out after {timeout} seconds. Partial output:\n{self.shell.before}",
            )
        exit_code = int(self.shell.match.group(1))

        output_before_marker = self.shell.before

        # 1. Split the raw output into lines
        all_lines = output_before_marker.splitlines()
        # 2. Filter out the lines that are just echoes of our commands
        clean_lines = []
        for line in all_lines:
            stripped_line = line.strip()
            # Ignore the line if it's an echo of the original command OR our marker command
            if stripped_line != full_command and marker_command not in stripped_line:
                clean_lines.append(line)
        # 3. Join the clean lines back together
        cleaned_output = "\n".join(clean_lines)
        # Wait for the next shell prompt to ensure the shell is ready
        self.shell.expect([r"\$", r"#"])
        return exit_code, cleaned_output.strip()
