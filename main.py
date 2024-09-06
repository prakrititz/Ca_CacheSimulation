import math 
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm 

class Cache:
    def __init__(self, cache_size_kb, block_size_bytes, associativity):
        """
        Initializes a cache object with the given cache size, block size, and associativity.
        
        The cache is organized as a set-associative cache, with the number of sets determined by the cache size, block size, and associativity. The offset bits, index bits, and tag bits are calculated based on these parameters.
        
        The cache is implemented as a list of OrderedDicts, where each OrderedDict represents a cache set. The OrderedDict is used to keep track of the least recently used (LRU) block in the set.
        
        The cache also keeps track of the number of hits and misses that occur during cache accesses.
        """
        self.size = cache_size_kb
        self.cache_size = cache_size_kb * 1024
        self.block_size = block_size_bytes
        self.associativity = associativity 
        self.num_sets = self.cache_size // (self.block_size * self.associativity)
        self.offset_bits = int(math.log2(self.block_size))
        self.index_bits = int(math.log2(self.num_sets))
        self.tag_bits = 32 - self.offset_bits - self.index_bits
        self.cache = [OrderedDict() for _ in range(self.num_sets)]
        self.hits = 0
        self.misses = 0
    
    def access(self,address):
        """Calculates the tag bits of the given memory address for the cache.
        The tag bits are calculated by shifting the address right by the sum of the offset bits and index bits, which gives the portion of the address that identifies the cache line."""
        tag = address>>(self.offset_bits + self.index_bits)


        """
        Calculates the index bits of the given memory address for the cache.
        The index bits are calculated by shifting the address right by the offset bits, and then masking off the lower bits to get the index portion of the address.
        """
        index = (address>>self.offset_bits)& ((1<<self.index_bits)-1)



        """
        Accesses the cache with the given memory address. 
        If the tag is found in the cache set, the cache hit count is incremented and the tag is moved to the end of the set (making it the most recently used). If the tag is not found, the cache miss count is incremented. 
        If the set is full, the least recently used tag is evicted to make room for the new tag.
        """
        
        cache_set = self.cache[index]
        if tag in cache_set:
            self.hits += 1
            cache_set.move_to_end(tag)
        else:
            self.misses += 1
            if len(cache_set) >= self.associativity:
                cache_set.popitem(last=True)
            cache_set[tag] = True


def read_file(filename):
    """
    Reads a trace file and yields the memory addresses accessed in the trace.
    
    The trace file is expected to have one address per line, with the address in hexadecimal format.
    
    Args:
        filename (str): The path to the trace file to read.
    
    Yields:
        int: The next memory address from the trace file, as an integer.
    """
    with open(filename, 'r') as f:
        for line in f:
            _, address, _ = line.strip().split()
            yield int(address, 16)



def run_cache_simulation(cache, trace_file):
    """
    Runs a cache simulation using the provided cache and trace file.
    
    Args:
        cache (Cache): The cache object to use for the simulation.
        trace_file (str): The path to the trace file containing the memory addresses to simulate.
    
    Returns:
        dict: A dictionary containing the results of the cache simulation, with the following keys:
            - "hits": The number of cache hits.
            - "misses": The number of cache misses.
            - "miss_rate": The cache miss rate as a percentage.
    """
    cache.hits = 0
    cache.misses = 0
    for address in read_file(trace_file):
        cache.access(address)
    total_accesses = cache.hits + cache.misses
    return {"hits": cache.hits, "misses":cache.misses, "miss_rate":(cache.misses / total_accesses)*100}


def print_results(part, results, cache):
    print(f"\n===== Results for {part}: A {cache.associativity} Way Cache Of Size {cache.size} And  Block Size {cache.block_size}======")
    
    headers = ["Trace Name", "Hits", "Misses", "Hit Rate", "Miss Rate", "Hit/Miss"]
    print(f"{headers[0]:<15} {headers[1]:>10} {headers[2]:>10} {headers[3]:>12} {headers[4]:>12} {headers[5]:>12}")
    
    print("-" * 75)
    
    for trace, stats in results.items():
        hits = stats['hits']
        misses = stats['misses']
        miss_rate = stats['miss_rate']
        hit_rate = 100 - miss_rate
        hit_miss_ratio = hit_rate / miss_rate if miss_rate != 0 else float('inf')
        
        print(f"{trace:<15} {hits:>10d} {misses:>10d} {hit_rate:>12.6f} {miss_rate:>12.6f} {hit_miss_ratio:>12.6f}")
    print()
    print()
    print()
    print()

class A:
    def __init__(self):
        self.cache_size_kb = 1024
        self.block_size_bytes = 4
        self.cache_size_kb = 1024
        self.block_size_bytes = 4
        self.associativity = 4
        self.trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
    
        self.results = {}
        self.hit_rates = []
        self.miss_rates = []
    
    def startA(self):
        for trace_file in self.trace_files:
            self.cache = Cache(self.cache_size_kb, self.block_size_bytes, self.associativity)
            
            data = run_cache_simulation(self.cache, os.path.join('./traces', trace_file))
            self.results[trace_file] = data
        self.printA()
        # self.plotA()

    def printA(self):
        print_results("Part A", self.results,self.cache)
    
    def plotA(self):
        plt.figure(figsize=(12, 8))
        x = range(len(self.trace_files))
        
        hit_rates = [100 - data['miss_rate'] for data in self.results.values()]
        miss_rates = [data['miss_rate'] for data in self.results.values()]
        
        plt.plot(x, hit_rates, marker='o', linestyle='-', label='Hit Rate', color='green')
        plt.plot(x, miss_rates, marker='s', linestyle=':', label='Miss Rate', color='red')
        
        plt.xlabel('Trace Files')
        plt.ylabel('Rate (%)')
        plt.title('Hit and Miss Rates for Different Trace Files')
        plt.xticks(x, [f.split('.')[0] for f in self.trace_files], rotation=45, ha='right')
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('part_a_plot.png', dpi=300)
        plt.close()

class B:
    def __init__(self):
        self.cache_sizes = [128, 256, 512, 1024, 2048, 4096]
        self.block_size = 4
        self.associativity = 4
        self.trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
        
    
    def startB(self):
        for cache_size in tqdm(self.cache_sizes, desc="Testing cache sizes"):
            self.results = {}
            for trace in self.trace_files:
                self.cache = Cache(cache_size, self.block_size,self. associativity)
                data = run_cache_simulation(self.cache, os.path.join('./traces', trace))
                self.results[trace] = data

            self.printB()
        # self.plotB()
    def plotB(self):
        plt.figure(figsize=(12, 8))
        colors = plt.get_cmap('tab10')
        for i, trace in enumerate(self.trace_files):
            color = colors(i / len(self.trace_files))
            hit_rates = []
            miss_rates = []
            for cache_size in self.cache_sizes:
                self.cache = Cache(cache_size, self.block_size, self.associativity)
                data = run_cache_simulation(self.cache, os.path.join('./traces', trace))
                hit_rates.append(100 - data['miss_rate'])
                miss_rates.append(data['miss_rate'])
            
            plt.plot(self.cache_sizes, hit_rates, marker='o', color=color, label=f'{trace} Hit Rate')
            plt.plot(self.cache_sizes, miss_rates, marker='s', linestyle='--', color=color, label=f'{trace} Miss Rate')
        
        plt.xlabel('Cache Size (KB)')
        plt.ylabel('Rate (%)')
        plt.title('Hit and Miss Rates vs Cache Size')
        plt.xscale('log', base=2)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('part_b_plot.png', dpi=300)
        plt.close()

    
    def printB(self):
        print_results("Part B", self.results, self.cache)

class C:
    def __init__(self):
        self.cache_size = 1024
        self.block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        self.associativity = 4
        self.trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
        

    
    def startC(self):
        for block_size in tqdm(self.block_sizes, desc="Testing block sizes"):
            self.results = {}
            for trace in self.trace_files:
                self.cache = Cache(self.cache_size, block_size, self.associativity)
                data = run_cache_simulation(self.cache, os.path.join('./traces', trace))
                self.results[trace] = data
            self.printC()

    def plotC(self):
        plt.figure(figsize=(12, 8))
        colors = plt.get_cmap('tab10')
        for i, trace in enumerate(self.trace_files):
            color = colors(i / len(self.trace_files))
            hit_rates = []
            miss_rates = []
            for block_size in self.block_sizes:
                self.cache = Cache(self.cache_size, block_size, self.associativity)
                data = run_cache_simulation(self.cache, os.path.join('./traces', trace))
                hit_rates.append(100 - data['miss_rate'])
                miss_rates.append(data['miss_rate'])
            
            plt.plot(self.block_sizes, hit_rates, marker='o', color=color, label=f'{trace} Hit Rate')
            plt.plot(self.block_sizes, miss_rates, marker='s', linestyle='--', color=color, label=f'{trace} Miss Rate')
        
        plt.xlabel('Block Size (bytes)')
        plt.ylabel('Rate (%)')
        plt.title('Hit and Miss Rates vs Block Size')
        plt.xscale('log', base=2)
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('part_c_plot.png', dpi=300)
        plt.close()

        
    def printC(self):
        print_results("Part C", self.results,self.cache)
class D:
    def __init__(self):
        self.cache_size = 1024
        self.block_size = 4
        self.associativities = [1, 2, 4, 8, 16, 32, 64]
        self.trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
        
        self.results = {trace: [] for trace in self.trace_files}
    
    def startD(self):
        for associativity in tqdm(self.associativities, desc="Testing associativities"):
            self.results = {}
            for trace in self.trace_files:
                self.cache = Cache(self.cache_size, self.block_size, associativity)
                data = run_cache_simulation(self.cache, os.path.join('./traces', trace))
                self.results[trace] =  data
            self.printD()



    def plotD(self):
            plt.figure(figsize=(12, 8))
            colors = plt.get_cmap('tab10')
            for i, trace in enumerate(self.trace_files):
                color = colors(i / len(self.trace_files))
                hit_rates = []
                miss_rates = []
                for associativity in self.associativities:
                    self.cache = Cache(self.cache_size, self.block_size, associativity)
                    data = run_cache_simulation(self.cache, os.path.join('./traces', trace))
                    hit_rates.append(100 - data['miss_rate'])
                    miss_rates.append(data['miss_rate'])
                
                plt.plot(self.associativities, hit_rates, marker='o', color=color, label=f'{trace} Hit Rate')
                plt.plot(self.associativities, miss_rates, marker='s', linestyle='--', color=color, label=f'{trace} Miss Rate')
            
            plt.xlabel('Associativity')
            plt.ylabel('Rate (%)')
            plt.title('Hit and Miss Rates vs Associativity')
            plt.xscale('log', base=2)
            plt.ylim(0, 100)
            plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
            plt.gca().xaxis.set_ticks(self.associativities)
            plt.legend()
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            plt.savefig('part_d_plot.png', dpi=300)
            plt.close()




    def printD(self):
        print_results("Part D", self.results,self.cache)

def main():
    print("Running Cache Simulation Experiments")

    # Part A
    print("\nRunning Part A: Fixed Cache Configuration")
    a = A()
    a.startA()
    a.plotA()

    # Part B
    print("\nRunning Part B: Varying Cache Size")
    b = B()
    b.startB()
    b.plotB()

    # Part C
    print("\nRunning Part C: Varying Block Size")
    c = C()
    c.startC()
    c.plotC()

    # Part D
    print("\nRunning Part D: Varying Associativity")
    d = D()
    d.startD()
    d.plotD()
    print("\nAll experiments completed. Check the generated plots and printed results.")

if __name__ == "__main__":
    main()
