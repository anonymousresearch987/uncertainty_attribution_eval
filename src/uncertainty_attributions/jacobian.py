from abc import ABC, abstractmethod


class JacobianGenerator(ABC):
    """Class to generate Jacobian matrices"""

    @abstractmethod
    def get_perturbed_explanations(self, model, input, epsilon: float = 1e-1):
        pass

    @abstractmethod
    def compute_delta(self, model, input):
        """function to compute delta matrix based on the uq_strategy

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): input data

        Returns:
            np.ndarray: delta matrix
        """
        pass

    @abstractmethod
    def approximate_jacobian(self, model, input, explanation, epsilon: float = 1e-1):
        """function to approximate the Jacobian matrix

        Args:
            model (nn.Module): trained model
            input (torch.Tensor): input data
            explanation (torch.Tensor): feature attributions for the input
            epsilon (float, optional): small perturbation value. Defaults to 1e-1.

        Returns:
            np.ndarray: approximated Jacobian matrix
        """
        pass
