import pytest
import pandas as pd

from cpm.utils.data import convert_to_RLRW, convert_to_PTSM
from cpm.applications.reinforcement_learning import RLRW
from cpm.applications.decision_making import PTSM, PTSM1992, PTSM2025


def test_convert_to_RLRW_creates_expected_columns_and_values():
    data = pd.DataFrame(
        {
            "participant_id": [1, 1, 2],
            "stim_left": [0, 1, 0],
            "stim_right": [1, 0, 1],
            "choice": [1, 0, 1],
            "reward_left": [1, 0, 1],
            "reward_right": [0, 1, 0],
            "block": [1, 1, 2],
        }
    )

    output = convert_to_RLRW(
        data=data,
        human_response="choice",
        reward=["reward_left", "reward_right"],
        stimulus=["stim_left", "stim_right"],
        participant="participant_id",
        block="block",
    )

    assert list(output.columns) == [
        "arm_0",
        "arm_1",
        "response",
        "reward_0",
        "reward_1",
        "participant",
        "block",
    ]
    assert output["arm_0"].tolist() == data["stim_left"].tolist()
    assert output["arm_1"].tolist() == data["stim_right"].tolist()
    assert output["response"].tolist() == data["choice"].tolist()
    assert (output["reward_0"] + output["reward_1"]).tolist() == (
        data["reward_left"] + data["reward_right"]
    ).tolist()
    assert output["participant"].tolist() == data["participant_id"].tolist()
    assert output["block"].tolist() == data["block"].tolist()


def test_convert_to_RLRW_raises_for_missing_additional_column():
    data = pd.DataFrame(
        {
            "participant_id": [1],
            "stimulus": [0],
            "choice": [1],
            "reward": [1],
        }
    )

    with pytest.raises(ValueError, match="Additional column 'missing_col' not found"):
        convert_to_RLRW(
            data=data,
            human_response="choice",
            reward="reward",
            stimulus="stimulus",
            participant="participant_id",
            extra="missing_col",
        )


def test_error_raised_for_missing_stimulus_column():
    data = pd.DataFrame(
        {
            "participant_id": [1],
            "choice": [1],
            "reward": [1],
        }
    )

    with pytest.raises(
        ValueError, match="Stimulus column 'stimulus' not found in data."
    ):
        convert_to_RLRW(
            data=data,
            human_response="choice",
            reward="reward",
            stimulus="stimulus",
            participant="participant_id",
        )


def test_converter_compatibility_with_reinforcement_learning_wrapper():
    data = pd.DataFrame(
        {
            "participant_id": [1, 1, 1],
            "stim_left": [1, 2, 1],
            "stim_right": [2, 1, 2],
            "choice": [1, 0, 1],
            "reward_left": [1, 0, 1],
            "reward_right": [0, 1, 0],
            "block": [1, 1, 1],
        }
    )

    output = convert_to_RLRW(
        data=data,
        human_response="choice",
        reward=["reward_left", "reward_right"],
        stimulus=["stim_left", "stim_right"],
        participant="participant_id",
        block="block",
    )

    model = RLRW(data=output, dimensions=2)

    try:
        model.run()
    except Exception as e:
        pytest.fail(f"RLRW model.run() raised an exception: {e}")


def test_convert_to_RLRW_with_single_stimulus_column():
    data = pd.DataFrame(
        {
            "participant_id": [1, 1, 2],
            "stimulus": [0, 1, 0],
            "choice": [1, 0, 1],
            "reward": [1, 0, 1],
        }
    )

    output = convert_to_RLRW(
        data=data,
        human_response="choice",
        reward="reward",
        stimulus="stimulus",
        participant="participant_id",
    )

    assert "arm" in output.columns
    assert output["arm"].tolist() == data["stimulus"].tolist()


def test_convert_to_RLRW_with_single_reward_column():
    data = pd.DataFrame(
        {
            "participant_id": [1, 1, 2],
            "stimulus": [0, 1, 0],
            "choice": [1, 0, 1],
            "reward": [1, 0, 1],
        }
    )

    output = convert_to_RLRW(
        data=data,
        human_response="choice",
        reward="reward",
        stimulus="stimulus",
        participant="participant_id",
    )

    assert "reward" in output.columns


def test_convert_to_RLRW_missing_participant_column():
    data = pd.DataFrame({"stimulus": [0], "choice": [1], "reward": [1]})

    with pytest.raises(
        ValueError, match="Participant column 'participant_id' not found"
    ):
        convert_to_RLRW(
            data=data,
            human_response="choice",
            reward="reward",
            stimulus="stimulus",
            participant="participant_id",
        )


def test_convert_to_RLRW_missing_human_response_column():
    data = pd.DataFrame({"participant_id": [1], "stimulus": [0], "reward": [1]})

    with pytest.raises(ValueError, match="Human response column 'choice' not found"):
        convert_to_RLRW(
            data=data,
            human_response="choice",
            reward="reward",
            stimulus="stimulus",
            participant="participant_id",
        )


def test_convert_to_RLRW_missing_reward_column():
    data = pd.DataFrame({"participant_id": [1], "stimulus": [0], "choice": [1]})

    with pytest.raises(ValueError, match="Reward column 'reward' not found"):
        convert_to_RLRW(
            data=data,
            human_response="choice",
            reward="reward",
            stimulus="stimulus",
            participant="participant_id",
        )


def test_convert_to_PTSM_creates_expected_columns_and_values():
    data = pd.DataFrame(
        {
            "safe": [1, 1, 2],
            "risky": [3, 4, 5],
            "prob": [0.2, 0.5, 0.8],
            "choice": [0, 1, 0],
            "block": [1, 1, 2],
        }
    )

    output = convert_to_PTSM(
        data=data,
        safe_magnitudes="safe",
        risky_magnitudes="risky",
        risky_probability="prob",
        response="choice",
        block="block",
    )

    assert list(output.columns) == [
        "safe_magnitudes",
        "risky_magnitudes",
        "risky_probability",
        "observed",
        "block",
    ]
    assert output["safe_magnitudes"].tolist() == data["safe"].tolist()
    assert output["risky_magnitudes"].tolist() == data["risky"].tolist()
    assert output["risky_probability"].tolist() == data["prob"].tolist()
    assert output["observed"].tolist() == data["choice"].tolist()
    assert output["block"].tolist() == data["block"].tolist()


def test_convert_to_PTSM_rejects_list_safe_magnitudes():
    data = pd.DataFrame(
        {
            "safe": [1],
            "risky": [2],
            "prob": [0.5],
            "choice": [1],
        }
    )

    with pytest.raises(
        ValueError, match="Safe magnitudes must be a single column name"
    ):
        convert_to_PTSM(
            data=data,
            safe_magnitudes=["safe"],
            risky_magnitudes="risky",
            risky_probability="prob",
            response="choice",
        )


def test_convert_to_PTSM_missing_response_column():
    data = pd.DataFrame(
        {
            "safe": [1],
            "risky": [2],
            "prob": [0.5],
        }
    )

    with pytest.raises(ValueError, match="Response column 'choice' not found"):
        convert_to_PTSM(
            data=data,
            safe_magnitudes="safe",
            risky_magnitudes="risky",
            risky_probability="prob",
            response="choice",
        )


def test_convert_to_PTSM_missing_safe_magnitudes_column():
    data = pd.DataFrame(
        {
            "risky": [2],
            "prob": [0.5],
            "choice": [1],
        }
    )

    with pytest.raises(ValueError, match="Safe magnitudes column 'safe' not found"):
        convert_to_PTSM(
            data=data,
            safe_magnitudes="safe",
            risky_magnitudes="risky",
            risky_probability="prob",
            response="choice",
        )


def test_convert_to_PTSM_missing_risky_magnitudes_column():
    data = pd.DataFrame(
        {
            "safe": [1],
            "prob": [0.5],
            "choice": [1],
        }
    )

    with pytest.raises(ValueError, match="Risky magnitudes column 'risky' not found"):
        convert_to_PTSM(
            data=data,
            safe_magnitudes="safe",
            risky_magnitudes="risky",
            risky_probability="prob",
            response="choice",
        )


def test_convert_to_PTSM_missing_risky_probability_column():
    data = pd.DataFrame(
        {
            "safe": [1],
            "risky": [2],
            "choice": [1],
        }
    )

    with pytest.raises(ValueError, match="Risky probability column 'prob' not found"):
        convert_to_PTSM(
            data=data,
            safe_magnitudes="safe",
            risky_magnitudes="risky",
            risky_probability="prob",
            response="choice",
        )


def test_converter_compatibility_with_decision_making_wrapper():
    data = pd.DataFrame(
        {
            "safe": [1, 1, 2],
            "risky": [3, 4, 5],
            "prob": [0.2, 0.5, 0.8],
            "choice": [0, 1, 0],
            "block": [1, 1, 2],
            "ambiguity": [0.1, 0.2, 0.3],
        }
    )

    output = convert_to_PTSM(
        data=data,
        safe_magnitudes="safe",
        risky_magnitudes="risky",
        risky_probability="prob",
        response="choice",
        block="block",
        ambiguity="ambiguity",
    )

    model = PTSM(data=output)
    model1992 = PTSM1992(data=output)
    model2025 = PTSM2025(data=output)

    try:
        model.run()
    except Exception as e:
        pytest.fail(f"PTSM model.run() raised an exception: {e}")
    try:
        model1992.run()
    except Exception as e:
        pytest.fail(f"PTSM1992 model.run() raised an exception: {e}")
    try:
        model2025.run()
    except Exception as e:
        pytest.fail(f"PTSM2025 model.run() raised an exception: {e}")
