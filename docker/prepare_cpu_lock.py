"""Adjust the Docker-local project copy to resolve CPU-only PyTorch wheels."""

from pathlib import Path
import tomllib


project_file = Path("/app/pyproject.toml")
project_text = project_file.read_text(encoding="utf-8")
project = tomllib.loads(project_text)
uv_settings = project.get("tool", {}).get("uv", {})
if any(key in uv_settings for key in ("sources", "index", "environments")):
    raise SystemExit("Docker CPU build needs updated uv source configuration")

project_file.write_text(
    project_text.rstrip()
    + """

[tool.uv]
environments = ["sys_platform == 'linux' and python_version == '3.12'"]

[tool.uv.sources]
torch = { index = "pytorch-cpu" }
torchvision = { index = "pytorch-cpu" }

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
""",
    encoding="utf-8",
)
