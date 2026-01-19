import math

class LearningRateSchedule:
    def learning_rate_schedule(self, it: int, max_learning_rate: float, 
                               min_learning_rate: float, warmup_iters: int, 
                               cosine_cycle_iters: int):
        # Warmup phase: t < Tw
        if it < warmup_iters:
            current_learning_rate = (it / warmup_iters) * max_learning_rate
        
        # Post-annealing: t > Tc
        elif it > cosine_cycle_iters:
            current_learning_rate = min_learning_rate
        
        # Cosine annealing: Tw ≤ t ≤ Tc
        else:
            progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
            current_learning_rate = (
                min_learning_rate + 
                0.5 * (1 + math.cos(math.pi * progress)) * 
                (max_learning_rate - min_learning_rate)
            )
        
        return current_learning_rate