"""Assembly layer tests (phase 3), driven through a real HTTP stub.

The stub speaks the OpenAI chat-completions protocol, so these exercise the
actual request bodies -- including whether the enum constraints really carry
the catalog -- rather than a mocked-out client.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest

from stub_llm import Reply, StubServer, serving
from urdfgen.catalog.digest import instance_port_options
from urdfgen.catalog.loader import Catalog
from urdfgen.llm.assemble import AssemblyError, assemble
from urdfgen.llm.client import LLMError, LocalLLMClient, extract_json
from urdfgen.llm.config import (
    DEFAULT_BASE_URL,
    LLMConfig,
    LLMConfigError,
    load_config,
)
from urdfgen.llm.prompts import RULES, format_errors, system_prompt
from urdfgen.llm.repair import assemble_with_repair
from urdfgen.llm.schema_gen import (
    connection_schema,
    module_selection_schema,
    parse_port_reference,
)
from urdfgen.validation.result import error, warning

QUAD_SELECTION: dict[str, Any] = {
    "name": "quadruped_a",
    "modules": [
        {"id": "body", "module": "body_quad_v1"},
        {"id": "leg_fl", "module": "leg_3dof_left_v1"},
        {"id": "leg_fr", "module": "leg_3dof_right_v1"},
        {"id": "leg_rl", "module": "leg_3dof_left_v1"},
        {"id": "leg_rr", "module": "leg_3dof_right_v1"},
    ],
    "notes": "Legs follow the chassis mount names.",
}

QUAD_CONNECTIONS: dict[str, Any] = {
    "root": "body",
    "connections": [
        {"parent": "body.leg_mount_fl", "child": "leg_fl.base"},
        {"parent": "body.leg_mount_fr", "child": "leg_fr.base"},
        {"parent": "body.leg_mount_rl", "child": "leg_rl.base"},
        {"parent": "body.leg_mount_rr", "child": "leg_rr.base"},
    ],
}


def quad_replies() -> list[Reply]:
    return [Reply.json_object(QUAD_SELECTION), Reply.json_object(QUAD_CONNECTIONS)]


@pytest.fixture
def stub() -> Iterator[StubServer]:
    """A stub that answers with a correct quadruped, unless a test overrides it."""
    with serving(quad_replies()) as state:
        yield state


def config_for(stub: StubServer, **overrides: Any) -> LLMConfig:
    return LLMConfig(model="stub-model", base_url=stub.base_url, timeout=10.0, **overrides)


def run(stub: StubServer, catalog: Catalog, text: str = "4脚ロボットを作って") -> Any:
    with LocalLLMClient(config_for(stub)) as client:
        return assemble_with_repair(text, catalog, client)


def expect_failure(
    replies: list[Reply], catalog: Catalog, exception: type[Exception], match: str
) -> None:
    """Run one assembly against ``replies`` and require it to fail loudly.

    Used for answers that satisfy the JSON schema but are semantically wrong;
    the point every time is that nothing is quietly patched up.
    """
    with (
        serving(replies) as stub,
        LocalLLMClient(config_for(stub)) as client,
        pytest.raises(exception, match=match),
    ):
        assemble("a quadruped", catalog, client)


# --- configuration --------------------------------------------------------


def test_model_must_be_configured() -> None:
    """Spec 4.1: the model name is never hardcoded, so there is no default."""
    with pytest.raises(LLMConfigError, match="no model configured"):
        load_config(env={})


def test_config_reads_the_environment() -> None:
    config = load_config(
        env={
            "LLM_MODEL": "qwen2.5:7b",
            "LLM_BASE_URL": "http://gpu-box:8000/v1",
            "LLM_TIMEOUT": "300",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_TOKENS": "4096",
        }
    )
    assert config.model == "qwen2.5:7b"
    assert config.chat_url == "http://gpu-box:8000/v1/chat/completions"
    assert (config.timeout, config.temperature, config.max_tokens) == (300.0, 0.2, 4096)


def test_config_defaults_to_a_local_endpoint_and_zero_temperature() -> None:
    config = load_config(env={"LLM_MODEL": "m"})
    assert config.base_url == DEFAULT_BASE_URL
    assert config.temperature == 0.0


def test_cli_overrides_beat_the_environment() -> None:
    config = load_config(
        model="override",
        base_url="http://x/v1",
        env={"LLM_MODEL": "env", "LLM_BASE_URL": "http://y"},
    )
    assert (config.model, config.base_url) == ("override", "http://x/v1")


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"LLM_MODEL": "m", "LLM_TIMEOUT": "soon"}, "not a number"),
        ({"LLM_MODEL": "m", "LLM_STRUCTURED_OUTPUT": "maybe"}, "is not one of"),
    ],
)
def test_bad_configuration_is_rejected(env: dict[str, str], message: str) -> None:
    with pytest.raises(LLMConfigError, match=message):
        load_config(env=env)


# --- generated schemas ----------------------------------------------------


def test_module_enum_is_exactly_the_catalog(catalog: Catalog) -> None:
    """Spec 4.1: the model cannot name a module that does not exist."""
    schema = module_selection_schema(catalog)
    enum = schema["properties"]["modules"]["items"]["properties"]["module"]["enum"]
    assert enum == catalog.ids()


def test_port_enum_is_exactly_the_available_ports(catalog: Catalog) -> None:
    """Spec 4.3: stage 2 chooses from the ports stage 1's modules actually have."""
    options = instance_port_options(catalog, {"body": "body_quad_v1", "leg_fl": "leg_3dof_left_v1"})
    schema = connection_schema(["body", "leg_fl"], options)
    for side in ("parent", "child"):
        assert schema["properties"]["connections"]["items"]["properties"][side]["enum"] == options
    assert "body.leg_mount_fl" in options
    assert "leg_fl.base" in options


def test_schemas_stay_flat(catalog: Catalog) -> None:
    """Nested $defs break small local models, so the schemas must not grow any."""
    for schema in (module_selection_schema(catalog), connection_schema(["a"], ["a.b"])):
        assert "$defs" not in json.dumps(schema)
        assert schema["additionalProperties"] is False


@pytest.mark.parametrize("reference", ["body", "", ".port", "body."])
def test_malformed_port_references_are_rejected(reference: str) -> None:
    with pytest.raises(ValueError, match=r"instance\.port"):
        parse_port_reference(reference)


# --- prompts --------------------------------------------------------------


def test_system_prompt_states_every_required_rule(catalog: Catalog) -> None:
    """Spec 4.4 lists six things the prompt must say."""
    prompt = system_prompt(catalog)
    for number in range(1, 7):
        assert f"\n{number}. " in RULES
    assert "Japanese or English" in prompt
    assert "body_quad_v1" in prompt


def test_system_prompt_carries_no_geometry(catalog: Catalog) -> None:
    """The model is never shown numbers it could copy as dimensions.

    The rules do mention inertia -- to forbid inventing it -- so only the
    catalog section is checked.
    """
    catalog_section = system_prompt(catalog).split("## Available modules", 1)[1]
    for banned in ("ixx", "inertia", "package://", "meshes/", "rpy"):
        assert banned not in catalog_section


def test_errors_are_formatted_with_severity() -> None:
    text = format_errors([error("a", "broken", "connections[0]"), warning("b", "odd")])
    assert "[MUST FIX] (connections[0]) broken" in text
    assert "[note] odd" in text


# --- request construction -------------------------------------------------


def test_requests_carry_the_configured_parameters(stub: StubServer, catalog: Catalog) -> None:
    with LocalLLMClient(config_for(stub, temperature=0.3, max_tokens=999)) as client:
        assemble("a quadruped", catalog, client)

    request = stub.request(0)
    assert request["model"] == "stub-model"
    assert request["temperature"] == 0.3
    assert request["max_tokens"] == 999
    assert request["stream"] is False


def test_both_stages_constrain_generation(stub: StubServer, catalog: Catalog) -> None:
    run(stub, catalog)
    assert stub.response_format_types() == ["json_schema", "json_schema"]
    assert (
        stub.schema_of(0)["properties"]["modules"]["items"]["properties"]["module"]["enum"]
        == catalog.ids()
    )


def test_stage_two_offers_only_the_chosen_instances_ports(
    stub: StubServer, catalog: Catalog
) -> None:
    run(stub, catalog)
    enum = stub.schema_of(1)["properties"]["connections"]["items"]["properties"]["parent"]["enum"]
    assert "body.leg_mount_fl" in enum
    assert "leg_fl.base" in enum
    # gripper_parallel_v1 was not selected, so none of its ports are offered.
    assert not any(option.startswith("gripper") for option in enum)


def test_stage_two_prompt_lists_the_ports(stub: StubServer, catalog: Catalog) -> None:
    run(stub, catalog)
    prompt = stub.request(1)["messages"][-1]["content"]
    assert "body.leg_mount_fl" in prompt
    assert "leg_fl.base" in prompt


def test_schema_is_not_repeated_in_the_prompt_when_enforced(
    stub: StubServer, catalog: Catalog
) -> None:
    """Context is scarce on small models; do not spend it twice on the schema."""
    run(stub, catalog)
    assert "Reply with a single JSON object matching this schema" not in json.dumps(
        stub.request(0)["messages"]
    )


# --- happy path -----------------------------------------------------------


def test_a_valid_quadruped_is_produced(stub: StubServer, catalog: Catalog) -> None:
    outcome = run(stub, catalog)

    assert outcome.succeeded
    assert outcome.attempts == 1
    assert outcome.spec is not None
    assert outcome.spec.name == "quadruped_a"
    assert outcome.spec.root == "body"
    assert len(outcome.spec.modules) == 5
    assert len(outcome.spec.connections) == 4


def test_versions_are_pinned(stub: StubServer, catalog: Catalog) -> None:
    """Spec 2.5: a saved assembly must rebuild identically later."""
    outcome = run(stub, catalog)
    assert outcome.spec is not None
    for instance in outcome.spec.modules:
        assert instance.version == catalog.get(instance.module).version


def test_notes_survive_into_the_spec(stub: StubServer, catalog: Catalog) -> None:
    outcome = run(stub, catalog)
    assert outcome.spec is not None
    assert outcome.spec.notes == "Legs follow the chassis mount names."


def test_the_result_builds_a_urdf(stub: StubServer, catalog: Catalog) -> None:
    from urdfgen.builder.expand import expand

    outcome = run(stub, catalog)
    assert outcome.spec is not None
    robot = expand(outcome.spec, catalog)
    assert len(robot.links) == 17
    assert len(robot.movable_joints) == 12


# --- awkward responses ----------------------------------------------------


def test_code_fenced_json_is_accepted(catalog: Catalog) -> None:
    """Small models wrap answers in ``` even when told not to."""
    fenced = [
        Reply(content=f"```json\n{json.dumps(QUAD_SELECTION)}\n```"),
        Reply(content=f"Here you go:\n```\n{json.dumps(QUAD_CONNECTIONS)}\n```\nHope that helps!"),
    ]
    with serving(fenced) as stub:
        assert run(stub, catalog).succeeded


def test_prose_around_the_json_is_tolerated(catalog: Catalog) -> None:
    chatty = [
        Reply(content=f"Sure! {json.dumps(QUAD_SELECTION)} Let me know if that works."),
        Reply.json_object(QUAD_CONNECTIONS),
    ]
    with serving(chatty) as stub:
        assert run(stub, catalog).succeeded


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('text {"a": {"b": 2}} more', '{"a": {"b": 2}}'),
        ('{"a": "}"}', '{"a": "}"}'),
        ('{"a": "\\""}', '{"a": "\\""}'),
        ("no json here", "no json here"),
    ],
)
def test_extract_json(content: str, expected: str) -> None:
    assert extract_json(content) == expected


def test_non_json_response_is_reported(catalog: Catalog) -> None:
    expect_failure(
        [Reply(content="I am afraid I cannot do that.")],
        catalog,
        LLMError,
        match="valid JSON",
    )


def test_broken_envelope_is_reported(catalog: Catalog) -> None:
    expect_failure(
        [Reply(content="", body='{"unexpected": true}')],
        catalog,
        LLMError,
        match="chat completion",
    )


def test_http_error_is_reported(catalog: Catalog) -> None:
    expect_failure(
        [Reply(content="", status=503, body="model is loading")],
        catalog,
        LLMError,
        match="HTTP 503",
    )


def test_unreachable_server_names_the_fix() -> None:
    config = LLMConfig(model="m", base_url="http://127.0.0.1:1/v1", timeout=2.0)
    with LocalLLMClient(config) as client, pytest.raises(LLMError, match="ollama serve"):
        client.complete_json([{"role": "user", "content": "hi"}], {}, "x")


# --- structured-output fallback -------------------------------------------


def test_server_without_json_schema_falls_back(catalog: Catalog) -> None:
    """llama.cpp and older Ollama reject response_format json_schema."""
    replies = [
        Reply(content="", status=400, body="unsupported response_format"),
        Reply.json_object(QUAD_SELECTION),
        Reply.json_object(QUAD_CONNECTIONS),
    ]
    with serving(replies) as stub:
        outcome = run(stub, catalog)
        assert outcome.succeeded
        assert stub.response_format_types() == ["json_schema", "json_object", "json_object"]


def test_fallback_puts_the_schema_in_the_prompt(catalog: Catalog) -> None:
    """Without server-side enforcement the prompt is the only place to state the shape."""
    replies = [
        Reply(content="", status=400, body="nope"),
        Reply.json_object(QUAD_SELECTION),
        Reply.json_object(QUAD_CONNECTIONS),
    ]
    with serving(replies) as stub:
        run(stub, catalog)
        assert "matching this schema" in json.dumps(stub.request(1)["messages"])
        # The refusal is remembered, so stage 2 does not re-try json_schema.
        assert "matching this schema" in json.dumps(stub.request(2)["messages"])


def test_fallback_is_reported_to_the_caller(catalog: Catalog) -> None:
    replies = [
        Reply(content="", status=400, body="nope"),
        Reply.json_object(QUAD_SELECTION),
        Reply.json_object(QUAD_CONNECTIONS),
    ]
    with serving(replies) as stub:
        assert run(stub, catalog).schema_enforced is False


def test_json_schema_mode_never_falls_back(catalog: Catalog) -> None:
    """An explicit demand for schema enforcement must fail rather than degrade."""
    with serving([Reply(content="", status=400, body="nope")]) as stub:
        config = config_for(stub, structured_output="json_schema")
        with LocalLLMClient(config) as client, pytest.raises(LLMError, match="HTTP 400"):
            assemble("a quadruped", catalog, client)


# --- untrusted output -----------------------------------------------------


def test_a_hallucinated_module_id_is_refused(catalog: Catalog) -> None:
    """Spec 12: schema-valid does not mean semantically valid.

    A small model can ignore the enum, and dropping the entry silently would
    build a robot nobody asked for.
    """
    invented = {**QUAD_SELECTION, "modules": [{"id": "body", "module": "hovercraft_v9"}]}
    expect_failure(
        [Reply.json_object(invented)],
        catalog,
        AssemblyError,
        match="not in the catalog",
    )


def test_a_port_outside_the_offered_set_is_refused(catalog: Catalog) -> None:
    bad = {"root": "body", "connections": [{"parent": "body.wing_mount", "child": "leg_fl.base"}]}
    expect_failure(
        [Reply.json_object(QUAD_SELECTION), Reply.json_object(bad)],
        catalog,
        AssemblyError,
        match="not one of the offered ports",
    )


def test_duplicate_instance_ids_are_refused(catalog: Catalog) -> None:
    duplicated = {
        **QUAD_SELECTION,
        "modules": [
            {"id": "body", "module": "body_quad_v1"},
            {"id": "body", "module": "leg_3dof_left_v1"},
        ],
    }
    expect_failure(
        [Reply.json_object(duplicated)],
        catalog,
        AssemblyError,
        match="used more than once",
    )


def test_a_root_that_was_not_selected_is_refused(catalog: Catalog) -> None:
    bad = {**QUAD_CONNECTIONS, "root": "chassis"}
    expect_failure(
        [Reply.json_object(QUAD_SELECTION), Reply.json_object(bad)],
        catalog,
        AssemblyError,
        match="not one of the chosen instances",
    )


def test_an_empty_selection_is_refused(catalog: Catalog) -> None:
    expect_failure(
        [Reply.json_object({"name": "x", "modules": [], "notes": ""})],
        catalog,
        AssemblyError,
        match="chose no modules",
    )


def test_robot_names_are_made_safe(catalog: Catalog) -> None:
    """The name lands in a URDF attribute, so it cannot be arbitrary text."""
    messy = {**QUAD_SELECTION, "name": '4脚 robot <"A">'}
    with serving([Reply.json_object(messy), Reply.json_object(QUAD_CONNECTIONS)]) as stub:
        outcome = run(stub, catalog)
        assert outcome.spec is not None
        assert all(char.isalnum() or char in "_-" for char in outcome.spec.name)


# --- repair loop ----------------------------------------------------------


def test_validation_errors_are_fed_back_and_fixed(catalog: Catalog) -> None:
    """Spec 4.5: a wrong assembly is retried with the checker's complaint."""
    wrong = {
        "root": "body",
        "connections": [
            {"parent": "body.leg_mount_fl", "child": "leg_fl.base"},
            {"parent": "body.leg_mount_fl", "child": "leg_fr.base"},
        ],
    }
    replies = [
        Reply.json_object(QUAD_SELECTION),
        Reply.json_object(wrong),
        Reply.json_object(QUAD_SELECTION),
        Reply.json_object(QUAD_CONNECTIONS),
    ]
    with serving(replies) as stub:
        outcome = run(stub, catalog)
        assert outcome.succeeded
        assert outcome.attempts == 2

        retry_prompt = json.dumps(stub.request(2)["messages"], ensure_ascii=False)
        assert "rejected by the checker" in retry_prompt
        assert "port_reused" in retry_prompt or "only be connected once" in retry_prompt


def test_malformed_answers_are_also_fed_back(catalog: Catalog) -> None:
    """A reply that is not even an assembly gets the same treatment."""
    invented = {**QUAD_SELECTION, "modules": [{"id": "body", "module": "hovercraft_v9"}]}
    replies = [
        Reply.json_object(invented),
        Reply.json_object(QUAD_SELECTION),
        Reply.json_object(QUAD_CONNECTIONS),
    ]
    with serving(replies) as stub:
        outcome = run(stub, catalog)
        assert outcome.succeeded
        assert outcome.attempts == 2
        assert "not in the catalog" in json.dumps(stub.request(1)["messages"])


def test_three_failures_return_the_partial_spec(catalog: Catalog) -> None:
    """Spec 4.5: do not swallow the failure -- hand back something editable."""
    wrong = {
        "root": "body",
        "connections": [
            {"parent": "body.leg_mount_fl", "child": "leg_fl.base"},
            {"parent": "body.leg_mount_fl", "child": "leg_fr.base"},
        ],
    }
    replies = [Reply.json_object(QUAD_SELECTION), Reply.json_object(wrong)] * 3
    with serving(replies) as stub:
        outcome = run(stub, catalog)

        assert not outcome.succeeded
        assert outcome.attempts == 3
        assert outcome.spec is not None, "a partial spec must still come back"
        assert len(outcome.spec.modules) == 5
        assert any(finding.code == "port_reused" for finding in outcome.findings)


def test_repeated_malformed_answers_leave_no_spec(catalog: Catalog) -> None:
    invented = {**QUAD_SELECTION, "modules": [{"id": "body", "module": "hovercraft_v9"}]}
    with serving([Reply.json_object(invented)]) as stub:
        outcome = run(stub, catalog)
        assert not outcome.succeeded
        assert outcome.spec is None
        assert [finding.code for finding in outcome.findings] == ["malformed_assembly"]


def test_attempt_count_is_configurable(catalog: Catalog) -> None:
    invented = {**QUAD_SELECTION, "modules": [{"id": "body", "module": "hovercraft_v9"}]}
    with serving([Reply.json_object(invented)]) as stub:
        with LocalLLMClient(config_for(stub)) as client:
            outcome = assemble_with_repair("x", catalog, client, max_attempts=1)
        assert outcome.attempts == 1
        assert len(stub.requests) == 1


def test_warnings_alone_do_not_trigger_a_retry(catalog: Catalog) -> None:
    """A quadruped leaves feet unconnected; that is expected, not an error."""
    with serving(quad_replies()) as stub:
        outcome = run(stub, catalog)
        assert outcome.attempts == 1
        assert any(finding.code == "unconnected_port" for finding in outcome.findings)
        assert outcome.succeeded
