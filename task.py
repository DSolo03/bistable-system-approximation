from abc import ABC, abstractmethod
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

class Parameters(dict):
    pass

class Equation(Protocol):
    def __call__(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        ...
    @property
    def __name__(self) -> str: ...

class Eigen():
    value: float
    vector: NDArray
    def __init__(self, eigenvalue: float, eigenvector: NDArray):
        self.value = eigenvalue
        self.vector = eigenvector
    def __repr__(self):
        return f"Eigen({self.value}, {self.vector})"

class Task(ABC):
    equilibria_points_count: int
    dimensions: int

    @abstractmethod
    def system(self, z: NDArray, parameters: Parameters) -> NDArray:
        pass

    @abstractmethod
    def jacobian(self, z: NDArray, parameters: Parameters) -> NDArray:
        pass

    @abstractmethod
    def equilibria_seeds(self, parameters: Parameters) -> NDArray:
        pass

    def equilibria(self, parameters: Parameters) -> NDArray:
        points = []
        for x0 in self.equilibria_seeds(parameters):
            res = root(lambda z: self.system(z, parameters), x0)
            if res.success:
                point = np.round(res.x, 8)
                if all(point >= 0):
                    is_duplicate = any(np.all(point == p) for p in points)
                    if not is_duplicate:
                        points.append(point)
        return np.array(points)

    def get_saddle(self, parameters: Parameters) -> tuple[NDArray, list[Eigen]] | tuple[None, None]:
        eigens = []
        equilibria = self.equilibria(parameters)
        if len(equilibria) == self.equilibria_points_count:
            for point in equilibria:
                jac = np.array(self.jacobian(point, parameters))
                eigvals, eigvecs = np.linalg.eig(jac)
                real = np.real(eigvals)
                if np.any(real > 0) and np.any(real < 0):
                    for i in range(eigvecs.shape[1]):
                        vector = eigvecs[:, i]
                        value = eigvals[i]
                        norm = np.linalg.norm(vector)
                        vector = vector / norm if norm != 0 else vector
                        eigens.append(Eigen(value, vector))
                    return point, eigens
        return None, None

    @abstractmethod
    def multistable_condition(self, parameters: Parameters) -> bool:
        pass

    @abstractmethod
    def sample(self) -> tuple[NDArray, Parameters]:
        pass

    @abstractmethod
    def classify_point(self, point: NDArray) -> int:
        pass

    @abstractmethod
    def expansion(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        pass

class TwoDTask(Task):
    equilibria_points_count = 4
    dimensions = 2

    def system(self, z: NDArray, parameters: Parameters) -> NDArray:
        x, y = z
        rx, ry = parameters["r"]
        sx, sy = parameters["s"]
        cx, cy = parameters["c"]
        S = parameters["S"]

        dx = rx * x * (1 - (x/cx)) - sx*x*y
        dy = ry * y * (1 - (y/cy)) + (S*sx-sy)*x*y
        return np.array([dx,dy])

    def jacobian(self, z: NDArray, parameters: Parameters) -> NDArray:
        x, y = z
        rx, ry = parameters["r"]
        sx, sy = parameters["s"]
        cx, cy = parameters["c"]
        S = parameters["S"]

        return np.array([[rx*(1-((2*x)/(cx)))-sx*y,-sx*x],[(S*sx-sy)*y,ry*(1-((2*y)/(cy)))+(S*sx-sy)*x]])

    def equilibria_seeds(self, parameters: Parameters) -> NDArray:
        cx, cy = parameters["c"]
        
        seeds = [
            [0, 0],
            [cx, 0],
            [0, cy],
            [cx/2, cy/2]
        ]

        return np.array(seeds)

    def multistable_condition(self, parameters: Parameters) -> bool:
        rx, ry = parameters["r"]
        sx, sy = parameters["s"]
        cx, cy = parameters["c"]
        S = parameters["S"]

        return (rx<sx*cy) and ((ry*(S*sx-sy)*cx)<0)

    def sample(self) -> tuple[NDArray, Parameters]:
        x, y = np.random.uniform(0, 1, 2)
    
        mu = np.random.uniform(0, 0.035)
        rx = np.random.uniform(0.001, 0.022)
        ry = mu + np.random.uniform(0.015, 0.035)

        sx = np.random.uniform(0, 0.04)
        sy = np.random.uniform(0, 0.04)

        cx = np.random.uniform(0, 1)
        cy = np.random.uniform(0, 1)

        S = np.random.uniform(0, 0.3)

        return np.array([x,y]), Parameters({"r":[rx,ry], "s":[sx,sy], "c":[cx,cy], "S":S})

    def classify_point(self, point: NDArray) -> int:
        x, y = point
        if x < 1e-2:
            return -1
        elif y < 1e-2:
            return 1
        return 0

    def expansion(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        if weights is None:
            weights = np.ones(21)
            
        HG0, F0 = z
        ghg, gf = parameters["r"]
        shg, sf = parameters["s"]
        ehg, ef = 0, parameters["S"]
        khg, kf = parameters["c"]

        term1 = weights[0] * (HG0 - F0)
        term2 = weights[1] * (ghg * HG0 - gf * F0)
        term3 = weights[2] * (shg * HG0 - sf * F0)
        term4 = weights[3] * (ehg * HG0 - ef * F0)
        term5 = weights[4] * (khg * HG0 - kf * F0)
        term6 = weights[5] * (HG0**2 - F0**2)

        term7 = weights[6] * (ghg * shg * HG0 - gf * sf * F0)
        term8 = weights[7] * (ghg * ehg * HG0 - gf * ef * F0)
        term9 = weights[8] * (ghg * khg * HG0 - gf * kf * F0)
        term10 = weights[9] * (shg * ehg * HG0 - sf * ef * F0)
        term11 = weights[10] * (shg * khg * HG0 - sf * kf * F0)
        term12 = weights[11] * (ehg * khg * HG0 - ef * kf * F0)

        term13 = weights[12] * (ghg**2 * HG0 - gf**2 * F0)
        term14 = weights[13] * (shg**2 * HG0 - sf**2 * F0)
        term15 = weights[14] * (ehg**2 * HG0 - ef**2 * F0)
        term16 = weights[15] * (khg**2 * HG0 - kf**2 * F0)

        term17 = weights[16] * (ghg * HG0**2 - gf * F0**2)
        term18 = weights[17] * (shg * HG0**2 - sf * F0**2)
        term19 = weights[18] * (ehg * HG0**2 - ef * F0**2)
        term20 = weights[19] * (khg * HG0**2 - kf * F0**2)

        term21 = weights[20] * (HG0**3 - F0**3)

        return [np.array([term1, term2, term3, term4, term5,
                term6, term7, term8, term9, term10,
                term11, term12, term13, term14, term15,
                term16, term17, term18, term19, term20, term21])]
    
    def compromise_expansion(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        if weights is None:
            weights = np.ones(10)

        slx, sly = saddle_point   
        vx, vy = 0.0, 0.0
        for eigen in eigens:
            if eigen.value < 0:
                vx, vy = eigen.vector

        HG0, F0 = z
        HG0=HG0-slx
        F0=F0-sly
        
        ghg, gf = parameters["r"]
        shg, sf = parameters["s"]
        ehg, ef = 0, parameters["S"]
        khg, kf = parameters["c"]

        term1 = weights[0] * (vy*HG0-vx*F0)

        term2 = weights[1] * (HG0**2 - F0**2)
        term3 = weights[2] * (ghg**2 * HG0 - gf**2 * F0)
        term4 = weights[3] * (shg**2 * HG0 - sf**2 * F0)
        term5 = weights[4] * (ehg**2 * HG0 - ef**2 * F0)
        term6 = weights[5] * (khg**2 * HG0 - kf**2 * F0)

        term7 = weights[6] * (ghg * HG0**2 - gf * F0**2)
        term8 = weights[7] * (shg * HG0**2 - sf * F0**2)
        term9 = weights[8] * (ehg * HG0**2 - ef * F0**2)
        term10 = weights[9] * (khg * HG0**2 - kf * F0**2)

        return [np.array([term1, term2, term3, term4, term5,
                term6, term7, term8, term9, term10])]

    def saddle_omega_separatrix(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        x, y = z
        slx, sly = saddle_point
        for eigen in eigens:
            if eigen.value < 0:
                vx, vy = eigen.vector
        term1 = vy*(x-slx)
        term2 = -1 * vx*(y-sly)

        return [np.array([term1, term2])]
    
    def saddle_alpha_separatrix(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        x, y = z
        slx, sly = saddle_point
        for eigen in eigens:
            if eigen.value > 0:
                vx, vy = eigen.vector
        term1 = vy*(x-slx)
        term2 = -1 * vx*(y-sly)

        return [np.array([term1, term2])]

    def RGR(self, z: NDArray, parameters: Parameters, saddle_point: NDArray, eigens: list[Eigen], weights: Optional[NDArray] = None) -> list[NDArray]:
        values = self.system(z,parameters)
        term1 = values[0] / z[0]
        term2 = -values[1] / z[1]
        return [np.array([term1,term2])]