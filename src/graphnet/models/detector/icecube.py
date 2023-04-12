"""IceCube-specific `Detector` class(es)."""

import torch
from torch_geometric.data import Data

from graphnet.models.components.pool import (
    group_pulses_to_dom,
    group_pulses_to_pmt,
    sum_pool_and_distribute,
)
from graphnet.data.constants import FEATURES
from graphnet.models.detector.detector import Detector


class IceCube86(Detector):
    """`Detector` class for IceCube-86."""

    # Implementing abstract class attribute
    features = FEATURES.ICECUBE86

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 500.0  # dom_x
        data.x[:, 1] /= 500.0  # dom_y
        data.x[:, 2] /= 500.0  # dom_z
        data.x[:, 3] = (data.x[:, 3] - 1.0e04) / 3.0e4  # dom_time
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 3.0  # charge
        data.x[:, 5] -= 1.25  # rde
        data.x[:, 5] /= 0.25
        data.x[:, 6] /= 0.05  # pmt_area

        return data


class IceCubeKaggle(Detector):
    """`Detector` class for Kaggle Competition."""

    # Implementing abstract class attribute
    features = FEATURES.KAGGLE
    xy_anisotropy_k = [
        -1.2334245570477371,
        -6.279196009389672,
        2.6169062228573026,
        0.8034497666860473,
        0.17294482605805447,
        -0.38345223229188563,
    ]

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Node features
        # Preprocessing
        data.x[:, 0] /= 500.0  # x
        data.x[:, 1] /= 500.0  # y
        data.x[:, 2] /= 500.0  # z
        data.x[:, 3] = (data.x[:, 3] - 1.0e04) / 3.0e4  # time
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 3.0  # charge
        # data.x[:, 5] is bool auxilaty, no need to normalize
        data.x[:, 6] /= 2.0  # sensor group
        data.x[:, 7] /= 1.35  # sensor relative qe

        # Edge attributes

        # Get edge XYZ vector
        edge_x = data.x[data.edge_index[0], 0] - data.x[data.edge_index[1], 0]
        edge_y = data.x[data.edge_index[0], 1] - data.x[data.edge_index[1], 1]
        edge_z = data.x[data.edge_index[0], 2] - data.x[data.edge_index[1], 2]

        # Angle between projection on XY and symmetry plane
        angles_to_symmetry_xy_plane = []
        for k in self.xy_anisotropy_k:
            k = torch.tensor(k)  # x = k * y, z plane
            dot = (edge_x * 1.0 + edge_y * k)
            angle_to_symmetry_xy_plane = torch.arccos(
                torch.clamp(dot / torch.sqrt(1 + k ** 2) / torch.sqrt(edge_x ** 2 + edge_y ** 2), -1.0, 1.0)
            )
            angle_to_symmetry_xy_plane = torch.where(
                torch.isnan(angle_to_symmetry_xy_plane), 
                torch.full_like(angle_to_symmetry_xy_plane, torch.pi / 2), 
                angle_to_symmetry_xy_plane
            )
            angle_to_symmetry_xy_plane = angle_to_symmetry_xy_plane / torch.pi
            angles_to_symmetry_xy_plane.append(angle_to_symmetry_xy_plane)

        # Angle to (0, 0, 1) vector: same string + up or down
        dot = (edge_x * 0.0 + edge_y * 0.0 + edge_z * 1.0)
        angle_to_z_axis = torch.arccos(
            torch.clamp(dot / torch.sqrt(edge_x ** 2 + edge_y ** 2 + edge_z ** 2), -1.0, 1.0)
        )
        angle_to_z_axis = torch.where(
            torch.isnan(angle_to_z_axis), 
            torch.zeros_like(angle_to_z_axis), 
            angle_to_z_axis
        )
        angle_to_z_axis = angle_to_z_axis / torch.pi
        
        # Sensor group diff
        sensor_group_diff = torch.abs(data.x[data.edge_index[0], 6] - data.x[data.edge_index[1], 6])

        # Eucledian distance between sensor positions
        edge_distance = torch.sqrt(edge_x ** 2 + edge_y ** 2 + edge_z ** 2)

        # Same sensor
        same_sensor = \
            torch.isclose(edge_distance, torch.tensor(0.0), atol=1e-5)
        
        # Time difference
        time_diff = data.x[data.edge_index[0], 3] - data.x[data.edge_index[1], 3]

        # Activation speed, to m / ns then normed by speed of light
        # https://www.kaggle.com/competitions/icecube-neutrinos-in-deep-ice/
        # discussion/382131
        activation_speed = edge_distance * 500.0 / torch.abs(time_diff * 3.0e4) / 0.3

        data.edge_attr = torch.stack(
            [
                *angles_to_symmetry_xy_plane,
                angle_to_z_axis,
                sensor_group_diff,
                edge_distance,
                same_sensor.float(),
                time_diff,
                activation_speed,
            ]
        ).T

        return data
    
    @property
    def nb_edge_attrs(self) -> int:
        """Return number of edge_attrs features.

        This the default, but may be overridden by specific inheriting classes.
        """
        return 12


class IceCubeDeepCore(IceCube86):
    """`Detector` class for IceCube-DeepCore."""

    # Implementing abstract class attribute
    features = FEATURES.DEEPCORE

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 100.0  # dom_x
        data.x[:, 1] /= 100.0  # dom_y
        data.x[:, 2] += 350.0  # dom_z
        data.x[:, 2] /= 100.0
        data.x[:, 3] /= 1.05e04  # dom_time
        data.x[:, 3] -= 1.0
        data.x[:, 3] *= 20.0
        data.x[:, 4] /= 1.0  # charge
        data.x[:, 5] -= 1.25  # rde
        data.x[:, 5] /= 0.25
        data.x[:, 6] /= 0.05  # pmt_area

        return data


class IceCubeUpgrade(IceCubeDeepCore):
    """`Detector` class for IceCube-Upgrade."""

    # Implementing abstract class attribute
    features = FEATURES.UPGRADE

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Preprocessing
        data.x[:, 0] /= 500.0  # dom_x
        data.x[:, 1] /= 500.0  # dom_y
        data.x[:, 2] /= 500.0  # dom_z
        data.x[:, 3] /= 2e04  # dom_time
        data.x[:, 3] -= 1.0
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 2.0  # charge
        # data.x[:,5] /= 1.  # rde
        data.x[:, 6] /= 0.05  # pmt_area
        data.x[:, 7] -= 50.0  # string
        data.x[:, 7] /= 50.0
        data.x[:, 8] /= 20.0  # pmt_number
        data.x[:, 9] -= 60.0  # dom_number
        data.x[:, 9] /= 60.0
        # data.x[:,10] /= 1.  # pmt_dir_x
        # data.x[:,11] /= 1.  # pmt_dir_y
        # data.x[:,12] /= 1.  # pmt_dir_z
        data.x[:, 13] /= 130.0  # dom_type

        return data


class IceCubeUpgrade_V2(IceCubeDeepCore):
    """`Detector` class for IceCube-Upgrade."""

    # Implementing abstract class attribute
    features = FEATURES.UPGRADE

    @property
    def nb_outputs(self) -> int:
        """Return number of output features."""
        return self.nb_inputs + 3

    def _forward(self, data: Data) -> Data:
        """Ingest data, build graph, and preprocess features.

        Args:
            data: Input graph data.

        Returns:
            Connected and preprocessed graph data.
        """
        # Check(s)
        self._validate_features(data)

        # Assign pulse cluster indices to DOMs and PMTs, respectively
        data = group_pulses_to_dom(data)
        data = group_pulses_to_pmt(data)

        # Feature engineering inspired by Linea Hedemark and Tetiana Kozynets.
        xyz = torch.stack((data["dom_x"], data["dom_y"], data["dom_z"]), dim=1)
        pmt_dir = torch.stack(
            (data["pmt_dir_x"], data["pmt_dir_x"], data["pmt_dir_x"]), dim=1
        )
        charge = data["charge"].unsqueeze(dim=1)
        center_of_gravity = sum_pool_and_distribute(
            xyz * charge, data.batch
        ) / sum_pool_and_distribute(charge, data.batch)
        vector_to_center_of_gravity = center_of_gravity - xyz
        distance_to_center_of_gravity = torch.norm(
            vector_to_center_of_gravity, p=2, dim=1
        )
        unit_vector_to_center_of_gravity = vector_to_center_of_gravity / (
            distance_to_center_of_gravity.unsqueeze(dim=1) + 1e-3
        )
        cos_angle_wrt_center_of_gravity = (
            pmt_dir * unit_vector_to_center_of_gravity
        ).sum(dim=1)
        photoelectrons_on_pmt = (
            sum_pool_and_distribute(data["charge"], data.pmt_index, data.batch)
            .floor()
            .clip(1, None)
        )

        # Add new features
        data.x = torch.cat(
            (
                data.x,
                photoelectrons_on_pmt.unsqueeze(dim=1),
                distance_to_center_of_gravity.unsqueeze(dim=1),
                cos_angle_wrt_center_of_gravity.unsqueeze(dim=1),
            ),
            dim=1,
        )

        # Preprocessing
        data.x[:, 0] /= 500.0  # dom_x
        data.x[:, 1] /= 500.0  # dom_y
        data.x[:, 2] /= 500.0  # dom_z
        data.x[:, 3] /= 2e04  # dom_time
        data.x[:, 3] -= 1.0
        data.x[:, 4] = torch.log10(data.x[:, 4]) / 2.0  # charge
        # data.x[:,5] /= 1.  # rde
        data.x[:, 6] /= 0.05  # pmt_area
        data.x[:, 7] -= 50.0  # string
        data.x[:, 7] /= 50.0
        data.x[:, 8] /= 20.0  # pmt_number
        data.x[:, 9] -= 60.0  # dom_number
        data.x[:, 9] /= 60.0
        # data.x[:,10] /= 1.  # pmt_dir_x
        # data.x[:,11] /= 1.  # pmt_dir_y
        # data.x[:,12] /= 1.  # pmt_dir_z
        data.x[:, 13] /= 130.0  # dom_type

        # -- Engineered features
        data.x[:, 14] = (
            torch.log10(data.x[:, 14]) / 2.0
        )  # photoelectrons_on_pmt
        data.x[:, 15] = (
            torch.log10(1e-03 + data.x[:, 15]) / 2.0
        )  # distance_to_center_of_gravity

        return data
