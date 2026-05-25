import subprocess

def test_ruff():
  result = subprocess.run(
    ["ruff", "check", "."],
    cwd=".",
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, f"ruff found issues:\n{result.stdout}"
