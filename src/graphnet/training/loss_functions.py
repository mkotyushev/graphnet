"""Collection of loss functions.

All loss functions inherit from `LossFunction` which ensures a common syntax,
handles per-event weights, etc.
"""

from abc import abstractmethod
from typing import Any, Optional, Union, List, Dict

import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch import nn
from torch.nn.functional import (
    one_hot,
    cross_entropy,
    binary_cross_entropy,
    softplus,
    cosine_similarity
)

from graphnet.utilities.config import save_model_config
from graphnet.models.model import Model
from graphnet.utilities.decorators import final


class LossFunction(Model):
    """Base class for loss functions in `graphnet`."""

    @save_model_config
    def __init__(self, **kwargs: Any) -> None:
        """Construct `LossFunction`, saving model config."""
        super().__init__(**kwargs)

    @final
    def forward(  # type: ignore[override]
        self,
        prediction: Tensor,
        target: Tensor,
        weights: Optional[Tensor] = None,
        return_elements: bool = False,
    ) -> Tensor:
        """Forward pass for all loss functions.

        Args:
            prediction: Tensor containing predictions. Shape [N,P]
            target: Tensor containing targets. Shape [N,T]
            return_elements: Whether elementwise loss terms should be returned.
                The alternative is to return the averaged loss across examples.

        Returns:
            Loss, either averaged to a scalar (if `return_elements = False`) or
            elementwise terms with shape [N,] (if `return_elements = True`).
        """
        # TODO: fix unsqueezed weights
        if weights is not None and weights.ndim == 2 and weights.shape[1] == 1:
            weights = weights.squeeze(dim=1)
        
        elements = self._forward(prediction, target)
        assert elements.size(dim=0) == target.size(
            dim=0
        ), "`_forward` should return elementwise loss terms."

        if weights is None:
            return elements if return_elements else torch.mean(elements)
        else:
            assert weights.shape == elements.shape
            return \
                (elements * weights) if return_elements else \
                (torch.sum(elements * weights) / torch.sum(weights))

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Syntax like `.forward`, for implentation in inheriting classes."""


class MSELoss(LossFunction):
    """Mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        assert prediction.dim() == 2
        assert prediction.size() == target.size()

        elements = torch.mean((prediction - target) ** 2, dim=-1)
        return elements


class RMSELoss(MSELoss):
    """Root mean squared error loss."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        # Check(s)
        elements = super()._forward(prediction, target)
        elements = torch.sqrt(elements)
        return elements


class LogCoshLoss(LossFunction):
    """Log-cosh loss function.

    Acts like x^2 for small x; and like |x| for large x.
    """

    @classmethod
    def _log_cosh(cls, x: Tensor) -> Tensor:  # pylint: disable=invalid-name
        """Numerically stable version on log(cosh(x)).

        Used to avoid `inf` for even moderately large differences.
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617]
        """
        return x + softplus(-2.0 * x) - np.log(2.0)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implement loss calculation."""
        diff = prediction - target
        elements = self._log_cosh(diff)
        return elements


class CrossEntropyLoss(LossFunction):
    """Compute cross-entropy loss for classification tasks.

    Predictions are an [N, num_class]-matrix of logits (i.e., non-softmax'ed
    probabilities), and targets are an [N,1]-matrix with integer values in
    (0, num_classes - 1).
    """

    @save_model_config
    def __init__(
        self,
        options: Union[int, List[Any], Dict[Any, int]],
        *args: Any,
        **kwargs: Any,
    ):
        """Construct CrossEntropyLoss."""
        # Base class constructor
        super().__init__(*args, **kwargs)

        # Member variables
        self._options = options
        self._nb_classes: int
        if isinstance(self._options, int):
            assert self._options in [torch.int32, torch.int64]
            assert (
                self._options >= 2
            ), f"Minimum of two classes required. Got {self._options}."
            self._nb_classes = options  # type: ignore
        elif isinstance(self._options, list):
            self._nb_classes = len(self._options)  # type: ignore
        elif isinstance(self._options, dict):
            self._nb_classes = len(
                np.unique(list(self._options.values()))
            )  # type: ignore
        else:
            raise ValueError(
                f"Class options of type {type(self._options)} not supported"
            )

        self._loss = nn.CrossEntropyLoss(reduction="none")

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Transform outputs to angle and prepare prediction."""
        if isinstance(self._options, int):
            # Integer number of classes: Targets are expected to be in
            # (0, nb_classes - 1).

            # Target integers are positive
            assert torch.all(target >= 0)

            # Target integers are consistent with the expected number of class.
            assert torch.all(target < self._options)

            assert target.dtype in [torch.int32, torch.int64]
            target_integer = target

        elif isinstance(self._options, list):
            # List of classes: Mapping target classes in list onto
            # (0, nb_classes - 1). Example:
            #    Given options: [1, 12, 13, ...]
            #    Yields: [1, 13, 12] -> [0, 2, 1, ...]
            target_integer = torch.tensor(
                [self._options.index(value) for value in target]
            )

        elif isinstance(self._options, dict):
            # Dictionary of classes: Mapping target classes in dict onto
            # (0, nb_classes - 1). Example:
            #     Given options: {1: 0, -1: 0, 12: 1, -12: 1, ...}
            #     Yields: [1, -1, -12, ...] -> [0, 0, 1, ...]
            target_integer = torch.tensor(
                [self._options[int(value)] for value in target]
            )

        else:
            assert False, "Shouldn't reach here."

        target_one_hot: Tensor = one_hot(target_integer, self._nb_classes).to(
            prediction.device
        )

        return self._loss(prediction.float(), target_one_hot.float())


class BinaryCrossEntropyLoss(LossFunction):
    """Compute binary cross entropy loss.

    Predictions are vector probabilities (i.e., values between 0 and 1), and
    targets should be 0 and 1.
    """

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        return binary_cross_entropy(
            prediction.float(), target.float(), reduction="none"
        )


class LogCMK(torch.autograd.Function):
    """MIT License.

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """

    @staticmethod
    def forward(
        ctx: Any, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Forward pass."""
        dtype = kappa.dtype
        ctx.save_for_backward(kappa)
        ctx.m = m
        ctx.dtype = dtype
        kappa = kappa.double()
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, kappa.cpu().numpy())
        ).to(kappa.device)
        return (
            (m / 2.0 - 1) * torch.log(kappa)
            - torch.log(iv)
            - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name,arguments-differ
        """Backward pass."""
        kappa = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        kappa = kappa.double().cpu().numpy()
        grads = -(
            (scipy.special.iv(m / 2.0, kappa))
            / (scipy.special.iv(m / 2.0 - 1, kappa))
        )
        return (
            None,
            grad_output
            * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )


class VonMisesFisherLoss(LossFunction):
    """General class for calculating von Mises-Fisher loss.

    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """

    @classmethod
    def log_cmk_exact(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss exactly."""
        return LogCMK.apply(m, kappa)

    @classmethod
    def log_cmk_approx(
        cls, m: int, kappa: Tensor
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss approx.

        [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additional minus sign.
        """
        v = m / 2.0 - 0.5
        a = torch.sqrt((v + 1) ** 2 + kappa**2)
        b = v - 1
        return -a + b * torch.log(b + a)

    @classmethod
    def log_cmk(
        cls, m: int, kappa: Tensor, kappa_switch: float = 100.0
    ) -> Tensor:  # pylint: disable=invalid-name
        """Calculate $log C_{m}(k)$ term in von Mises-Fisher loss.

        Since `log_cmk_exact` is diverges for `kappa` >~ 700 (using float64
        precision), and since `log_cmk_approx` is unaccurate for small `kappa`,
        this method automatically switches between the two at `kappa_switch`,
        ensuring continuity at this point.
        """
        kappa_switch = torch.tensor([kappa_switch]).to(kappa.device)
        mask_exact = kappa < kappa_switch

        # Ensure continuity at `kappa_switch`
        offset = cls.log_cmk_approx(m, kappa_switch) - cls.log_cmk_exact(
            m, kappa_switch
        )
        ret = cls.log_cmk_approx(m, kappa) - offset
        ret[mask_exact] = cls.log_cmk_exact(m, kappa[mask_exact])
        return ret

    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a vector in D dimensons.

        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction: Predicted vector, of shape [batch_size, D].
            target: Target unit vector, of shape [batch_size, D].

        Returns:
            Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk(m, k) - dotprod
        return elements

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError


class VonMisesFisher2DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 2D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for an angle in the 2D plane.

        Args:
            prediction: Output of the model. Must have shape [N, 2] where 0th
                column is a prediction of `angle` and 1st column is an estimate
                of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            loss: Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 2
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target[:, 0]
        t = torch.stack(
            [
                torch.cos(angle_true),
                torch.sin(angle_true),
            ],
            dim=1,
        )

        # Formatting prediction
        angle_pred = prediction[:, 0]
        kappa = prediction[:, 1]
        p = kappa.unsqueeze(1) * torch.stack(
            [
                torch.cos(angle_pred),
                torch.sin(angle_pred),
            ],
            dim=1,
        )

        return self._evaluate(p, t)


class EuclideanDistanceLoss(LossFunction):
    """Mean squared error in three dimensions."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate 3D Euclidean distance between predicted and target.

        Args:
            prediction: Output of the model. Must have shape [N, 3]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        return torch.sqrt(
            (prediction[:, 0] - target[:, 0]) ** 2
            + (prediction[:, 1] - target[:, 1]) ** 2
            + (prediction[:, 2] - target[:, 2]) ** 2
        )


class VonMisesFisher3DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 3D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for a direction in the 3D.

        Args:
            prediction: Output of the model. Must have shape [N, 4] where
                columns 0, 1, 2 are predictions of `direction` and last column
                is an estimate of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 4
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        kappa = prediction[:, 3]
        p = kappa.unsqueeze(1) * prediction[:, [0, 1, 2]]
        return self._evaluate(p, target)


class CosineLoss(LossFunction):
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate cosine distance between given angles (on unit sphere).

        Args:
            prediction: Output of the model. Must have shape [N, 2] where
                columns 0, 1 are predictions of `zenith` and `azimuth`
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise cosine distance. Shape [N,]
        """
        prediction_x = torch.cos(prediction[:, 1]) * torch.sin(prediction[:, 0])
        prediction_y = torch.sin(prediction[:, 1]) * torch.sin(prediction[:, 0])
        prediction_z = torch.cos(prediction[:, 0])
        prediction_xyz = torch.stack([prediction_x, prediction_y, prediction_z], dim=1)
        prediction_xyz = prediction_xyz / torch.norm(prediction_xyz, p=2, dim=1).unsqueeze(1)

        target_x = torch.cos(target[:, 1]) * torch.sin(target[:, 0])
        target_y = torch.sin(target[:, 1]) * torch.sin(target[:, 0])
        target_z = torch.cos(target[:, 0])
        target_xyz = torch.stack([target_x, target_y, target_z], dim=1)
        target_xyz = target_xyz / torch.norm(target_xyz, p=2, dim=1).unsqueeze(1)

        result = 1 - torch.sum(prediction_xyz * target_xyz, dim=1)

        return result


class CosineLoss3D(LossFunction):
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_xyz = prediction[:, :3]
        prediction_xyz = prediction_xyz / torch.norm(prediction_xyz, p=2, dim=1).unsqueeze(1)

        target_x = torch.cos(target[:, 1]) * torch.sin(target[:, 0])
        target_y = torch.sin(target[:, 1]) * torch.sin(target[:, 0])
        target_z = torch.cos(target[:, 0])
        target_xyz = torch.stack([target_x, target_y, target_z], dim=1)
        target_xyz = target_xyz / torch.norm(target_xyz, p=2, dim=1).unsqueeze(1)

        result = 1 - torch.sum(prediction_xyz * target_xyz, dim=1)

        return result


class VonMisesFisher2DLossSinCos(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 2D plane."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate von Mises-Fisher loss for an angle in the 2D plane.

        Args:
            prediction: Output of the model. Must have shape [N, 3] where 0, 1
                columns are sin and cos of `angle` and 2nd column is an estimate
                of `kappa`.
            target: Target tensor, extracted from graph object.

        Returns:
            loss: Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 3
        assert target.dim() == 2
        assert prediction.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target[:, 0]
        t = torch.stack(
            [
                torch.cos(angle_true),
                torch.sin(angle_true),
            ],
            dim=1,
        )

        # Formatting prediction
        sin = prediction[:, 0]
        cos = prediction[:, 1]
        kappa = prediction[:, 2]
        p = kappa.unsqueeze(1) * torch.stack(
            [
                cos,
                sin,
            ],
            dim=1,
        )

        return self._evaluate(p, t)


class EuclidianDistanceLossSinCos(LossFunction):
    """Mean squared error in three dimensions."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate 2D Euclidean distance between predicted and target
        (sin, cos) vectors.

        Args:
            prediction: Output of the model. Must have shape [N, 2]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Euclidian distance between (sin, cos) vectors 
            loss terms. Shape [N,]
        """
        sin_true = torch.sin(target[:, 0])
        cos_true = torch.cos(target[:, 0])

        sin_pred = prediction[:, 0]
        cos_pred = prediction[:, 1]

        return ((sin_true - sin_pred) ** 2 + (cos_true - cos_pred) ** 2) ** 0.5


class EuclidianDistanceLossCos(LossFunction):
    """Mean squared error in three dimensions."""

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate Euclidean distance between predicted and target
        (cos,) vectors.

        Args:
            prediction: Output of the model. Must have shape [N, 1]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise von Euclidian distance between (cos,) vectors 
            loss terms. Shape [N,]
        """
        cos_true = torch.cos(target[:, 0])
        cos_pred = prediction[:, 0]

        return (cos_true - cos_pred).abs()


class S2AbsCosineLoss(LossFunction):
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate cosine distance between abs of predicted and target
        spherical coordinates on unit sphere.

        Args:
            prediction: Output of the model. Must have shape [N, 3]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise cosine distance between vectors 
            loss terms. Shape [N,]
        """
        target = target.reshape(-1, 3)

        assert prediction.dim() == 2 and prediction.size()[1] == 3, \
            f"Prediction must have shape [N, 3], but has shape {prediction.shape}"
        assert target.dim() == 2 and target.size()[1] == 3, \
            f"Target must have shape [N, 3], but has shape {target.shape}"
        
        return 1 - cosine_similarity(prediction, torch.abs(target))


class S2AbsSoftCrossEntropyLoss(LossFunction):
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate cross entropy loss considering p^2 as predicted
        probabilities and y^2 as soft probabilistic labels.

        See paragraph 3.1 in https://arxiv.org/pdf/1904.05404.pdf 
        for details.

        Args:
            prediction: Output of the model. Must have shape [N, 3]
            target: Target tensor, extracted from graph object.

        Returns:
            Elementwise crossentropy loss terms between vectors. 
            Shape [N,]
        """
        target = target.reshape(-1, 3)

        assert prediction.dim() == 2 and prediction.size()[1] == 3, \
            f"Prediction must have shape [N, 3], but has shape {prediction.shape}"
        assert target.dim() == 2 and target.size()[1] == 3, \
            f"Target must have shape [N, 3], but has shape {target.shape}"
        
        return cross_entropy(prediction ** 2, target ** 2, reduction="none")


class S2SignCrossEntropyLoss(LossFunction):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculate absolute difference between predicted and target
        spherical coordinates.

        Args:
            prediction: X, Y, Z sign logits. Must have shape [N, 8]
            target: Target (zenith, azimuth) tensor, extracted from graph object.

        Returns:
            Cross-entropy loss of sign class. Shape [N,]
        """
        target = target.reshape(-1, 3)

        assert prediction.dim() == 2 and prediction.size()[1] == 8, \
            f"Prediction must have shape [N, 8], but has shape {prediction.shape}"
        assert target.dim() == 2 and target.size()[1] == 3, \
            f"Target must have shape [N, 3], but has shape {target.shape}"

        # Get sign, keep only 0 and 1, 0 is considered negative
        target_sign = torch.sign(target)
        target_sign[target_sign == -1] = 0

        # Convert to long
        target_x_sign = target_sign[:, 0].long()
        target_y_sign = target_sign[:, 1].long()
        target_z_sign = target_sign[:, 2].long()

        # Convert to decimal
        # x, y, z -> sign class
        # 0, 0, 0 -> 0	
        # 0, 0, 1 -> 1
        # 0, 1, 0 -> 2
        # 0, 1, 1 -> 3
        # 1, 0, 0 -> 4
        # 1, 0, 1 -> 5
        # 1, 1, 0 -> 6
        # 1, 1, 1 -> 7
        target_sign_class = 4 * target_x_sign + 2 * target_y_sign + 1 * target_z_sign

        return cross_entropy(prediction, target_sign_class, reduction="none")
