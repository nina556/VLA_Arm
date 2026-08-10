# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.factory import resolve_policy_features
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def test_resolve_policy_features_uses_dataset_shapes() -> None:
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        f"{OBS_IMAGES}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        f"{OBS_IMAGES}.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        f"{OBS_IMAGES}.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,)),
    }

    input_features, output_features = resolve_policy_features(features)

    assert output_features[ACTION].shape == (16,)
    assert input_features[OBS_STATE].shape == (16,)
    assert f"{OBS_IMAGES}.top" in input_features
    assert ACTION not in input_features


def test_resolve_policy_features_applies_rename_map() -> None:
    features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        f"{OBS_IMAGES}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        f"{OBS_IMAGES}.left_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        f"{OBS_IMAGES}.right_wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(16,)),
    }
    rename_map = {
        f"{OBS_IMAGES}.top": f"{OBS_IMAGES}.camera1",
        f"{OBS_IMAGES}.left_wrist": f"{OBS_IMAGES}.camera2",
        f"{OBS_IMAGES}.right_wrist": f"{OBS_IMAGES}.camera3",
    }

    input_features, output_features = resolve_policy_features(features, rename_map=rename_map)

    assert output_features[ACTION].shape == (16,)
    assert input_features[OBS_STATE].shape == (16,)
    assert f"{OBS_IMAGES}.camera1" in input_features
    assert f"{OBS_IMAGES}.camera2" in input_features
    assert f"{OBS_IMAGES}.camera3" in input_features
    assert f"{OBS_IMAGES}.top" not in input_features
