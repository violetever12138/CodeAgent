import os
from pathlib import Path

from codeswarm_agent.cli import _load_project_dotenv, build_parser
from trae_agent.utils.config import Config


def test_load_project_dotenv_fills_empty_launcher_value(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=local-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "")

    _load_project_dotenv(Path(tmp_path / "trae_config.yaml"))

    assert os.environ["OPENAI_API_KEY"] == "local-key"


def test_load_project_dotenv_preserves_non_empty_process_value(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=local-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "process-key")

    _load_project_dotenv(Path(tmp_path / "trae_config.yaml"))

    assert os.environ["OPENAI_API_KEY"] == "process-key"


def test_codeswarm_cli_defaults_to_manager_v2_and_accepts_resume_without_issue():
    parser = build_parser()
    fresh = parser.parse_args(["fix regression", "--working-dir", "repo"])
    resumed = parser.parse_args(["--resume-run", "run-dir"])

    assert fresh.architecture_mode == "manager_v2"
    assert resumed.issue is None
    assert resumed.working_dir is None
    assert resumed.resume_run == "run-dir"


def test_qwen_embedding_configuration_is_opt_in_and_parses_from_yaml():
    config = Config.create(config_file=str(Path(__file__).parents[2] / "trae_config.yaml.example"))
    assert config.codeswarm.embedding_provider == "none"
    assert len(config.codeswarm.embedding_revision) == 40
    parsed = build_parser().parse_args(["fix regression", "--embedding-provider", "qwen", "--embedding-device", "cuda:0"])
    assert parsed.embedding_provider == "qwen" and parsed.embedding_device == "cuda:0"
