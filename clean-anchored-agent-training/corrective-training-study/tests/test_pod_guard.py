"""CPU tests for scripts/pod_guard.sh (GUARD_TEST=1: "remote" runs locally, pod stop/status hit a fake state file).
Covers: prompt detection of completion (incl. the stage-1 pgrep self-match regression), archive fetched + sha-verified
before stop, billing-stopped verification, driver death without marker, time cap, SSH loss, and corrupt archive."""
import hashlib
import os
import shutil
import subprocess
import tarfile
import time

import pytest

GUARD = os.path.join(os.path.dirname(__file__), "..", "scripts", "pod_guard.sh")
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None, reason="bash not available")


def P(p):  # posix path for bash
    return str(p).replace("\\", "/")


def setup(tmp, driver_cmd, max_seconds=60, ssh_down=False, corrupt=False):
    pod = tmp / "pod"; pod.mkdir(); local = tmp / "local"
    (pod / "out.txt").write_text("result\n")
    fake = tmp / "fake_pod"; fake.write_text("RUNNING")
    if ssh_down:
        (tmp / "ssh_down").write_text("x")
    # the driver: writes its pid, works, archives, then the marker (exactly the stage-2 driver's contract)
    arc = pod / "stage.tgz"
    script = tmp / "cfp2_driver.sh"
    script.write_text(f"""echo $$ > {P(pod)}/driver.pid
{driver_cmd}
cd {P(pod)} && tar czf stage.tgz out.txt
(cd {P(pod)} && sha256sum stage.tgz > stage.tgz.sha256)
{"echo deadbeef  stage.tgz > " + P(pod) + "/stage.tgz.sha256" if corrupt else ""}
echo DRIVER_DONE >> {P(pod)}/driver.log
sleep 30
""")
    env = dict(os.environ, GUARD_TEST="1", FAKE_POD=P(fake), FAKE_SSH_DOWN=P(tmp / "ssh_down"), POD_ID="testpod",
               T0=str(int(time.time())), MAX_SECONDS=str(max_seconds), SSH_HOST="x", SSH_PORT="0",
               R_LOG=P(pod / "driver.log"), R_PID=P(pod / "driver.pid"), R_ARCHIVE=P(arc), LOCAL_DIR=P(local),
               POLL="1", MAX_SSH_FAIL="3")
    return env, script, fake, local, arc


def run(tmp, driver_cmd, **kw):
    env, script, fake, local, arc = setup(tmp, driver_cmd, **kw)
    drv = subprocess.Popen([BASH, P(script)])
    time.sleep(0.5)
    t = time.time()
    g = subprocess.run([BASH, P(GUARD)], env=env, capture_output=True, text=True, timeout=120)
    drv.kill()
    return g.stdout + g.stderr, time.time() - t, fake.read_text().strip(), local, arc


def test_completion_detected_promptly_and_archive_verified(tmp_path):
    out, dt, state, local, arc = run(tmp_path, "sleep 3")
    assert "GUARD_RESULT reason=DONE archive_verified=YES stopped=YES" in out, out
    assert state == "EXITED"
    assert dt < 3 + 2 + 3, dt                     # stopped within ~1 poll of the marker
    assert (local / "stage.tgz").exists()
    assert hashlib.sha256((local / "stage.tgz").read_bytes()).hexdigest() == hashlib.sha256(arc.read_bytes()).hexdigest()
    with tarfile.open(local / "stage.tgz") as t:
        assert t.extractfile("out.txt").read().strip() == b"result"
    assert arc.exists()                           # the original stays on the volume


def test_driver_name_in_probe_cmdline_does_not_block_detection(tmp_path):
    # regression for the stage-1 bug: the guard's own probe must not be mistaken for a live driver
    out, dt, state, _, _ = run(tmp_path, "sleep 1")
    assert "pgrep" not in open(GUARD).read().split("PROBE=")[1].split("\n\n")[0]
    assert "reason=DONE" in out and state == "EXITED"


def test_driver_dies_without_marker(tmp_path):
    env, script, fake, local, arc = setup(tmp_path, "")
    script.write_text(f"echo $$ > {env['R_PID']}\nsleep 2\nexit 1\n")
    drv = subprocess.Popen([BASH, P(script)]); time.sleep(0.5)
    g = subprocess.run([BASH, P(GUARD)], env=env, capture_output=True, text=True, timeout=60); drv.wait()
    assert "reason=DRIVER_DEAD" in g.stdout and fake.read_text().strip() == "EXITED", g.stdout


def test_time_cap_stops_pod(tmp_path):
    out, dt, state, local, _ = run(tmp_path, "sleep 100", max_seconds=3)
    assert "reason=TIMECAP" in out and "stopped=YES" in out and state == "EXITED", out
    assert dt < 8


def test_ssh_loss_stops_pod(tmp_path):
    out, _, state, _, _ = run(tmp_path, "sleep 100", ssh_down=True)
    assert "reason=SSH_LOST" in out and state == "EXITED", out


def test_corrupt_archive_still_stops_but_reports_unverified(tmp_path):
    out, _, state, _, _ = run(tmp_path, "sleep 1", corrupt=True)
    assert "reason=DONE archive_verified=NO stopped=YES" in out and state == "EXITED", out
