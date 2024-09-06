import math
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

class CacheLine:
    def __init__(self):
        self.valid = True

class Cache:
    def __init__(self, cache_size_kb, block_size_bytes, associativity):
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

    def access(self, address):
        tag = address >> (self.offset_bits + self.index_bits)
        index = (address >> self.offset_bits) & ((1 << self.index_bits) - 1)
            
        cache_set = self.cache[index]
        if tag in cache_set:
            self.hits += 1
            cache_set.move_to_end(tag)
        else:
            self.misses += 1
            if len(cache_set) >= self.associativity:
                cache_set.popitem(last=True)
            cache_set[tag] = CacheLine()

def print_output(part, results, parameters):
    print(f"\n=== Results for {part} ===")
    for trace, rates in results.items():
        print(f"\nTrace: {trace}")
        print(f"{'Parameter':<10} {'Hit Rate':>10} {'Miss Rate':>10}")
        print("-" * 40)
        for param, (hit_rate, miss_rate) in zip(parameters, rates):
            print(f"{param:<10} {hit_rate:>10.4f} {miss_rate:>10.4f}")
        print()



def read_trace(filename):
    with open(filename, 'r') as f:
        for line in f:
            _, address, _ = line.strip().split()
            yield int(address, 16)

def run_cache_simulation(cache, trace_file):
    cache.hits = 0
    cache.misses = 0
    for address in read_trace(trace_file):
        cache.access(address)
    total_accesses = cache.hits + cache.misses
    return cache.misses / total_accesses

def print_results(part, results):
    print(f"\n=== Results for {part} ===")
    print(f"{'Trace File':<15} {'Hit Rate':>10} {'Miss Rate':>10}")
    print("-" * 40)
    for trace, (hit_rate, miss_rate) in results.items():
        print(f"{trace:<15} {hit_rate:>10.4f} {miss_rate:>10.4f}")

def part_a():
    cache_size_kb = 1024
    block_size_bytes = 4
    associativity = 4
    
    cache = Cache(cache_size_kb, block_size_bytes, associativity)
    trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
    
    results = {}
    hit_rates = []
    miss_rates = []
    
    for trace_file in trace_files:
        miss_rate = run_cache_simulation(cache, os.path.join('./traces', trace_file))
        hit_rate = 1 - miss_rate
        results[trace_file] = (hit_rate, miss_rate)
        hit_rates.append(hit_rate)
        miss_rates.append(miss_rate)
    
    print_results("Part A", results)
    
    plt.figure(figsize=(12, 8))
    x = range(len(trace_files))
    
    plt.plot(x, hit_rates, marker='o', linestyle='-', label='Hit Rate', color='green')
    plt.plot(x, miss_rates, marker='s', linestyle=':', label='Miss Rate', color='red')
    
    plt.xlabel('Trace Files')
    plt.ylabel('Rate')
    plt.title('Hit and Miss Rates for Different Trace Files')
    plt.xticks(x, [f.split('.')[0] for f in trace_files], rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('part_a_plot.png', dpi=300)
    plt.close()

def part_b():
    cache_sizes = [128, 256, 512, 1024, 2048, 4096]
    block_size = 4
    associativity = 4
    trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
    
    results = {trace: [] for trace in trace_files}
    
    for cache_size in tqdm(cache_sizes, desc="Testing cache sizes"):
        for trace in trace_files:
            cache = Cache(cache_size, block_size, associativity)
            miss_rate = run_cache_simulation(cache, os.path.join('./traces', trace))
            results[trace].append(miss_rate)
    
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap('tab10')
    for i, trace in enumerate(trace_files):
        color = colors(i / len(trace_files))
        plt.plot(cache_sizes, [1 - mr for mr in results[trace]], marker='o', color=color, label=f'{trace} Hit Rate')
        plt.plot(cache_sizes, results[trace], marker='s', linestyle='--', color=color, label=f'{trace} Miss Rate')
    
    plt.xlabel('Cache Size (KB)')
    plt.ylabel('Rate')
    plt.title('Hit and Miss Rates vs Cache Size')
    plt.xscale('log', base=2)
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.02))
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('part_b_plot.png', dpi=300)
    plt.close()
    print_results = {}
    for trace in trace_files:
        print_results[trace] = [(1 - miss_rate, miss_rate) for miss_rate in results[trace]]
    
    print_output("Part B - Cache Size Variation", print_results, cache_sizes)
    print("Cache sizes tested:", cache_sizes)

def part_c():
    cache_size = 1024
    block_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    associativity = 4
    trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
    
    results = {trace: [] for trace in trace_files}
    
    for block_size in tqdm(block_sizes, desc="Testing block sizes"):
        for trace in trace_files:
            cache = Cache(cache_size, block_size, associativity)
            miss_rate = run_cache_simulation(cache, os.path.join('./traces', trace))
            results[trace].append(miss_rate)
    
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap('tab10')
    for i, trace in enumerate(trace_files):
        color = colors(i / len(trace_files))
        plt.plot(block_sizes, [1 - mr for mr in results[trace]], marker='o', color=color, label=f'{trace} Hit Rate')
        plt.plot(block_sizes, results[trace], marker='s', linestyle='--', color=color, label=f'{trace} Miss Rate')
    
    plt.xlabel('Block Size (bytes)')
    plt.ylabel('Rate')
    plt.title('Hit and Miss Rates vs Block Size')
    plt.xscale('log', base=2)
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.02))
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('part_c_plot.png', dpi=300)
    plt.close()
    print_results = {}
    for trace in trace_files:
        print_results[trace] = [(1 - miss_rate, miss_rate) for miss_rate in results[trace]]
    
    print_output("Part C - Block Size Variation", print_results, block_sizes)
    print("Block sizes tested:", block_sizes)

def part_d():
    cache_size = 1024
    block_size = 4
    associativities = [1, 2, 4, 8, 16, 32, 64]
    trace_files = [f for f in os.listdir('./traces') if f.endswith('.trace')]
    
    results = {trace: [] for trace in trace_files}
    
    for associativity in tqdm(associativities, desc="Testing associativities"):
        for trace in trace_files:
            cache = Cache(cache_size, block_size, associativity)
            hit_rate = 1 - run_cache_simulation(cache, os.path.join('./traces', trace))
            results[trace].append(hit_rate)
    
    plt.figure(figsize=(12, 8))
    colors = plt.get_cmap('tab10')
    for i, trace in enumerate(trace_files):
        color = colors(i / len(trace_files))
        plt.plot(associativities, results[trace], marker='o', color=color, label=f'{trace} Hit Rate')
        plt.plot(associativities, [1 - hr for hr in results[trace]], marker='s', linestyle='--', color=color, label=f'{trace} Miss Rate')
    
    plt.xlabel('Associativity')
    plt.ylabel('Rate')
    plt.title('Hit and Miss Rates vs Associativity')
    plt.xscale('log', base=2)
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.02))
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('part_d_plot.png', dpi=300)
    plt.close()
    print_results = {}
    for trace in trace_files:
        print_results[trace] = [(hit_rate, 1 - hit_rate) for hit_rate in results[trace]]
    
    print_output("Part D - Associativity Variation", print_results, associativities)
    print("Associativities tested:", associativities)

def main():
    print("Starting Cache Simulation")
    print("=" * 40)
    part_a()
    part_b()
    part_c()
    part_d()
    print("\nSimulation Complete. Check the generated plots.")

if __name__ == "__main__":
    main()