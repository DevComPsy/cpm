import pytest
import pandas as pd

from cpm.utils.data import convert_to_RLRW
from cpm.applications.reinforcement_learning import RLRW


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
