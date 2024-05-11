from abc import ABC

class Scheduler(ABC):
    def __init__(self) -> None:
        pass

    def step(self) -> float:
        pass

class Linear_scheduler(Scheduler):
    def __init__(self,to_min_step:int,init:float,minimum:float) -> None:
        self.param = init
        self.minimum = minimum
        self.k = (minimum-init)/to_min_step
    
    def step(self) -> float:
        if self.param <= self.minimum:
            return self.minimum
        
        self.param += self.k
        return self.param