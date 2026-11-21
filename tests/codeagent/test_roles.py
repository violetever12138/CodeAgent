from codeagent.protocol import WorkerRole
from codeagent.roles import DEFAULT_ROLES


def test_roles_have_complementary_permissions():
    explorer = DEFAULT_ROLES[WorkerRole.EXPLORER]
    developer = DEFAULT_ROLES[WorkerRole.DEVELOPER]
    security = DEFAULT_ROLES[WorkerRole.SECURITY]
    integrator = DEFAULT_ROLES[WorkerRole.INTEGRATOR]
    tester = DEFAULT_ROLES[WorkerRole.TESTER]
    reviewer = DEFAULT_ROLES[WorkerRole.REVIEWER]

    assert explorer.read_only and "repository_read" in explorer.allowed_tools
    assert "bash" not in explorer.allowed_tools
    assert "str_replace_based_edit_tool" in developer.allowed_tools
    assert security.read_only and security.allowed_tools == ()
    assert "patch_apply" in integrator.allowed_tools and not integrator.read_only
    assert "bash" in tester.allowed_tools
    assert "str_replace_based_edit_tool" not in tester.allowed_tools
    assert reviewer.read_only and reviewer.may_approve_patch
    assert "bash" not in reviewer.allowed_tools
    assert reviewer.allowed_tools == ()
    assert "sequentialthinking" not in explorer.allowed_tools
    assert len({role.allowed_tools for role in DEFAULT_ROLES.values()}) > 3
