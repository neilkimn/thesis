from collections.abc import Sequence
from torch.multiprocessing import Array
from shared.util import MPSLogger, MaxVal, Counter, AtomicInt

class MPSWeights(Sequence):
    def __init__(self, num_processes, increment=2, update_interval=5, log_path=None):
        self.weights = Array('i', [10]*num_processes)
        self.num_processes = num_processes
        self.percentages = Array('i', [0]*num_processes)
        self.set_percentages()
        self.increment = increment
        self.iterations = Counter(0) # Number of iterations algorithm have run for
        self.update_interval = update_interval
        self.converged = AtomicInt(0) # Whether or not we've converged
        self.updates = Counter(0) # Amount of updates done in one iteration 
        self.biggest_queue = MaxVal(0) # Value of the biggest queue
        self.biggest_index = AtomicInt(-1) # Index of process with the biggest queue size
        self.equals = Counter(1) # Amount of queue sizes that are equal
        self.convergence_counter = Counter(0) # Number of times queues across training processes have been equal
        if log_path:
            self.logger = MPSLogger(log_path)
        super().__init__()

    def set_percentages(self):
        total = sum(self.weights)
        for idx, w in enumerate(self.weights):
            self.percentages[idx] = int(w/total*100)

    def increment_weight(self, i):
        self.weights[i] += self.increment
        self.set_percentages()

    def __getitem__(self, i):
        return self.percentages[i]
    def __len__(self):
        return len(self.percentages)
    
    def set_convergence(self):
        self.converged.set_value(1)
    
    def get_convergence(self):
        return self.converged.value()

    def update_weights(self, pid):

        # If queue sizes for all training processes have been updated in current iteration, we can update weights
        if self.updates.value() == self.num_processes:

            # Queue sizes across training processes were too similar, updating one over another would skew training speed
            if self.equals.value() == self.num_processes:

                self.convergence_counter.increment()
                self.equals.reset()
                self.biggest_queue.set_value(0)
                self.biggest_index.reset()

                # If we've seen similar queue sizes across training processes three times in a row, assume convergence
                if self.convergence_counter.value() == 3:
                    print("Weights were not updated for three iterations, setting convergence")
                    self.set_convergence()
            
            # Queue size of some training process was greater than others, increment weight for that process
            else:
                if self.biggest_index.value() > -1:
                    self.increment_weight(self.biggest_index.value())
                    self.biggest_queue.set_value(0)

                self.logger.write_line(str(pid),str(self.iterations.value()),"","","",
                                       str(self.biggest_index.value()),"","",str(self.convergence_counter.value()))
                
                if self.biggest_index.value() > -1:
                    self.biggest_index.reset()
                self.equals.reset()
                self.convergence_counter.reset()

            self.updates.reset()
            self.iterations.increment()
        

    def set_qsize(self, qsize, i, pid, model_name):

        self.logger.write_line(str(pid),str(self.iterations.value()),str(i),model_name,str(qsize),"",
                               str(self.weights[i]),str(self.percentages[i]),str(self.convergence_counter.value()))

        # Check if queue sizes were roughly the same and that both biggest queue size recorded and current queue size is not 0
        if abs(qsize - self.biggest_queue.value()) <= 2 and (self.biggest_queue.value() != 0 and qsize != 0):
            # If above is the case, we deem the queues equal and increment towards convergence
            self.equals.increment()
        elif qsize > self.biggest_queue.value():
            # Otherwise, set process with biggest queue size to have weight updated next
            self.biggest_queue.set_value(qsize)
            self.biggest_index.set_value(i)
        self.updates.increment()