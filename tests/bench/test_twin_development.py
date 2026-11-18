"""Development evidence/order/decision fixtures, not physical study results."""

import json

import numpy as np
import pytest

from noether_optics.bench import twin_development as study


@pytest.fixture
def packet(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in study.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    monkeypatch.setattr(study.importlib.metadata, "version", lambda name: "fixture")
    result = study.DevelopmentPacket(tmp_path / "packet", source)
    try:
        yield result
    finally:
        result.close()


def events(packet):
    return [json.loads(line) for line in (packet.root / "events.jsonl").read_text().splitlines()]


def test_dispatch_is_durable_before_callback_and_every_failed_attempt_remains(packet):
    def callback():
        retained = events(packet)
        assert retained[-1]["kind"] == "dispatch"
        assert retained[-1]["identity"] == "objective-000000"
        raise ArithmeticError("injected optical failure")

    with pytest.raises(ArithmeticError, match="injected"):
        packet.call("objective", {"theta": [0, 0]}, callback)
    assert packet.counts["objective"] == 1
    assert [row["kind"] for row in events(packet)] == ["dispatch", "exception"]
    assert not (packet.root / "manifest.json").exists()


def test_nonfinite_arrays_are_retained_and_never_turned_into_missing_case(packet):
    with pytest.raises(FloatingPointError, match="retained"):
        packet.call("reference", {}, lambda: {"power": np.array([np.nan, np.inf])}, arrays=True)
    result = next(row for row in events(packet) if row["kind"] == "result")
    assert result["status"] == "nonfinite"
    with np.load(packet.root / result["raw"]["path"], allow_pickle=False) as archive:
        assert np.isnan(archive["power"][0]) and np.isposinf(archive["power"][1])
    assert not (packet.root / "manifest.json").exists()


def test_nonfinite_objective_and_gradient_are_retained_in_strict_json(packet):
    with pytest.raises(FloatingPointError):
        packet.call("objective", {}, lambda: (np.inf, np.array([np.nan, 1.0])))
    result = next(row for row in events(packet) if row["kind"] == "result")
    assert result["output"]["objective"] == {"nonfinite": "inf"}
    assert result["output"]["gradient"][0] == {"nonfinite": "nan"}


def test_existing_attempt_cannot_be_reopened_even_if_incomplete(packet):
    packet.close()
    with pytest.raises(FileExistsError):
        study.DevelopmentPacket(packet.root, packet.root)


def test_resource_cap_rejects_before_callback_and_dispatch(packet, monkeypatch):
    monkeypatch.setitem(study.LIMITS, "objective", 0)
    before = (packet.root / "events.jsonl").read_bytes()
    with pytest.raises(RuntimeError, match="cap"):
        packet.call("objective", {}, lambda: pytest.fail("must not dispatch"))
    assert (packet.root / "events.jsonl").read_bytes() == before


def test_both_fluxes_share_same_fixed_state_set_and_only_development_namespace():
    contract = study.protocol()
    assert contract["held_out_generation"] is False
    assert len(contract["detector_seeds"]) == 8 and len(set(contract["detector_seeds"])) == 8
    assert contract["plant_grid"] == 512 and contract["candidate_grids"] == [64, 128, 256]
    assert contract["diversities_opd_m"] == [30e-9, -30e-9]
    assert contract["active_surface_axes"] == [7, 18]
    assert len(contract["states_nm"]) == 4 and len(contract["source_photons_per_pair"]) == 2


def case(identity, coarse, fine, choices, *, state=(0.0, 0.0)):
    return {
        "id": identity,
        "evaluator_state_nm": list(state),
        "reference_wfe_squared_nm2": coarse,
        "diagnostic_wfe_squared_nm2": fine,
        "arms": [
            {"grid": n, "fit": {"status": "locally_resolved"}, "choice": {"chosen_id": action}}
            for n, action in zip((64, 128, 256), choices, strict=True)
        ],
    }


def test_all_action_tie_is_a_valid_failed_discrimination_gate():
    ties = {f"increment-{i}": 1.0 for i in range(9)}
    result = study.discrimination([case("tie", ties, ties.copy(), ["increment-0"] * 3)])
    assert result["passed"] is False
    assert result["cross_model_separated_disagreement"] is False
    assert result["contrasts"] == []


def test_cross_model_action_difference_below_grid_sensitivity_is_not_a_win():
    a, b = "increment-0", "increment-1"
    first = case("a", {a: 1.0, b: 2.0}, {a: 3.0, b: 1.0}, [a, b, a])
    second = case("b", {a: 2.0, b: 1.0}, {a: 2.0, b: 1.0}, [b, b, b])
    result = study.discrimination([first, second])
    assert result["state_dependent_best_actions"] is False
    assert result["cross_model_separated_disagreement"] is False
    assert result["passed"] is False


def test_known_finite_grid_action_contrast_can_pass_without_making_physical_claim():
    a, b = "increment-0", "increment-1"
    first = case("a", {a: 1.0, b: 2.0}, {a: 1.01, b: 2.01}, [a, b, a], state=(-20, -10))
    second = case("b", {a: 2.0, b: 1.0}, {a: 2.01, b: 1.01}, [b, b, b], state=(20, 10))
    result = study.discrimination([first, second])
    assert result["passed"] is True
    assert result["contrasts"][0]["reference_difference_nm2"] == -1


def test_adequacy_never_uses_truth_or_substitutes_independent_solution():
    primary = {"status": "unresolved", "estimate_nm": None}
    expanded = {"status": "locally_resolved", "estimate_nm": [0, 0], "best_objective": 1.0}
    assert study.adequacy(primary, expanded, [], np.eye(2))["passed"] is False
    primary = dict(expanded)
    independent = [{"status": "stationary", "theta": [0, 0], "objective": 1.0} for _ in range(2)]
    assert study.adequacy(primary, expanded, independent, np.eye(2))["passed"] is True
    independent[1]["theta"] = [0.1, 0]
    assert study.adequacy(primary, expanded, independent, np.eye(2))["passed"] is False


def test_physical_action_loss_is_double_pass_and_has_distinct_signed_best_actions():
    gram = np.eye(2)
    positive = study.action_scores([20, 10], gram)
    negative = study.action_scores([-20, -10], gram)
    assert min(positive, key=positive.get) == "increment-0"
    assert min(negative, key=negative.get) == "increment-8"
    assert positive["increment-4"] == 4 * (20**2 + 10**2)


def test_full_injected_pilot_shares_observations_freezes_choices_and_retains_costs(tmp_path, monkeypatch):
    """Synthetic wiring test; no HCIPy/dLux acquisition or scientific result."""
    source = tmp_path / "source"
    for name in study.SOURCE_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    monkeypatch.setattr(study.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(study.os, "fsync", lambda fd: None)  # publication ordering is checked above
    root = tmp_path / "pilot"
    seen = {n: set() for n in (64, 128, 256)}
    profiles = np.array([[[1.0, -1.0], [1.0, -1.0]], [[1.0, 1.0], [-1.0, -1.0]]])

    def reference_factory(*, defocus_opd_m):
        def evaluate(n, command, gain, *, component=False):
            current = json.loads((root / "events.jsonl").read_text().splitlines()[-1])
            context = current["context"]
            if context["phase"] == "counterfactual":
                assert (root / f'{context["case"]}-choices.json').is_file()
            surface = np.tensordot(command[[7, 18]], profiles, axes=1)
            power = np.full((65, 65), (0.3 + abs(command[7]) * 1e5 + abs(command[18]) * 1e5) / 65**2)
            values = {"power": power, "incident": np.array(1.0)}
            if component:
                values.update(surface=surface, aperture=np.ones((2, 2), dtype=bool))
            return values

        return evaluate

    def candidate_factory(n):
        def objective(theta, counts, photons):
            seen[n].add((photons, counts.tobytes()))
            difference = theta - np.array([0.25, -0.25])
            return float(difference @ difference / 2), difference

        return {
            "objective": objective,
            "gram": np.eye(2),
            "compile_seconds": 0.01,
            "setup_seconds": 0.001,
            "pupil_samples": n,
        }

    monkeypatch.setattr(study, "reference_backend", reference_factory)
    monkeypatch.setattr(study, "make_candidate", candidate_factory)
    result = study.run_development(root, source_root=source)
    assert result["status"] == "complete_development"
    assert len(result["cases"]) == 8
    assert seen[64] == seen[128] == seen[256] and len(seen[64]) == 8
    assert result["counts"]["reference"] == 92
    objective_calls = 96
    for case_result in result["cases"]:
        assert len(case_result["arms"]) == 3
        for arm in case_result["arms"]:
            objective_calls += arm["fit"]["objective_calls"]
            if "expanded_fit" in arm:
                objective_calls += arm["expanded_fit"]["objective_calls"]
                objective_calls += sum(len(fit["trace"]) for fit in arm["independent_fit"])
    assert result["counts"]["objective"] == objective_calls
    assert result["decision_budget_frozen"] is False
    assert all(len(row["seconds"]) == 32 for row in result["timing"].values())
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert study.file_hash(root / name) == digest
    assert result["development_discrimination_passed"] is False  # all candidates made the same choice


def test_reference_ranking_detects_nonzero_sign_reversal_independently_of_separation():
    a, b = "increment-0", "increment-1"
    reversed_case = case("reversed", {a: 1.0, b: 2.0}, {a: 2.0, b: 1.0}, [a, b, a])
    stable_case = case("stable", {a: 2.0, b: 1.0}, {a: 2.0, b: 1.0}, [b, b, b])
    result = study.discrimination([reversed_case, stable_case])
    ranking = result["reference_ranking"]
    assert ranking["resolved_pair_ordering_stable"] is False
    assert ranking["all_pairs_resolved"] is True
    assert len(ranking["agreed_pairs"]) == 1 and len(ranking["reversed_pairs"]) == 1
    reversal = ranking["reversed_pairs"][0]
    assert reversal["reference_difference_nm2"] == -1
    assert reversal["diagnostic_difference_nm2"] == 1
    assert result["passed"] is False


def test_zero_reference_contrast_is_unresolved_not_agreement_or_reversal():
    a, b = "increment-0", "increment-1"
    result = study.discrimination([case("zero", {a: 1.0, b: 2.0}, {a: 1.0, b: 1.0}, [a, b, a])])
    ranking = result["reference_ranking"]
    assert ranking["resolved_pair_ordering_stable"] is False
    assert ranking["all_pairs_resolved"] is False
    assert ranking["agreed_pairs"] == ranking["reversed_pairs"] == []
    assert len(ranking["unresolved_pairs"]) == 1
    unresolved = ranking["unresolved_pairs"][0]
    assert unresolved["unresolved_grids"] == [1024]
    assert unresolved["diagnostic_difference_nm2"] == 0
    assert unresolved["roundoff_tolerance_nm2"] == 64 * np.finfo(np.float64).eps * 2
    assert result["passed"] is False


def test_near_tie_winning_label_changes_do_not_establish_state_dependent_actions():
    a, b, c = "increment-0", "increment-1", "increment-2"
    delta = np.finfo(np.float64).eps
    first_losses = {a: 1.0, b: 1.0 + delta, c: 2.0}
    second_losses = {a: 1.0 + delta, b: 1.0, c: 2.0}
    assert min(first_losses, key=first_losses.get) != min(second_losses, key=second_losses.get)
    first = case("state-a", first_losses, first_losses.copy(), [a, c, a], state=(-1, 0))
    second = case("state-b", second_losses, second_losses.copy(), [b, c, b], state=(1, 0))
    result = study.discrimination([first, second])
    assert result["reference_ranking"]["resolved_pair_ordering_stable"] is True
    assert result["cross_model_separated_disagreement"] is True
    assert result["state_dependent_best_actions"] is False
    assert result["separated_physical_states"] == []
    assert all(row["near_best_union"] == [a, b] for row in result["near_best_actions_by_state"])
    for row in result["near_best_actions_by_case"]:
        assert row["grids"]["512"]["near_best_ids"] == [a, b]
        assert row["grids"]["1024"]["near_best_ids"] == [a, b]
    assert result["passed"] is False


def test_distinct_physical_states_with_disjoint_best_unions_supply_positive_witness():
    gram = np.eye(2)
    negative_losses = study.action_scores([-20, -10], gram)
    positive_losses = study.action_scores([20, 10], gram)
    negative = case(
        "negative",
        negative_losses,
        negative_losses.copy(),
        ["increment-8", "increment-4", "increment-8"],
        state=(-20, -10),
    )
    positive = case(
        "positive",
        positive_losses,
        positive_losses.copy(),
        ["increment-0", "increment-4", "increment-0"],
        state=(20, 10),
    )
    result = study.discrimination([negative, positive])
    assert result["state_dependent_best_actions"] is True
    assert len(result["separated_physical_states"]) == 1
    assert result["separated_physical_states"][0]["near_best_unions"] == [["increment-8"], ["increment-0"]]
    assert result["passed"] is True


def test_flux_versions_of_one_physical_state_cannot_supply_distinct_state_witness():
    a, b = "increment-0", "increment-1"
    first = case("same-state-low-flux", {a: 1.0, b: 2.0}, {a: 1.0, b: 2.0}, [a, b, a], state=(7, -3))
    second = case("same-state-high-flux", {a: 2.0, b: 1.0}, {a: 2.0, b: 1.0}, [b, a, b], state=(7, -3))
    result = study.discrimination([first, second])
    assert len(result["near_best_actions_by_state"]) == 1
    assert result["near_best_actions_by_state"][0]["near_best_union"] == [a, b]
    assert result["state_dependent_best_actions"] is False
    assert result["separated_physical_states"] == []
    assert result["passed"] is False
