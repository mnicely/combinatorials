# Combinatorials
Example using the combinatorial number system to produce all possible combinations of a complete graph and then finding all valid spanning trees.

## Prerequisites
The build process assumes compute capability (CC) 6.0 or greater.

An additional CC can easily be added by appending to ARCHES in the makefile.

## Built with 
This example utilizes the following toolsets:
- Thrust
- C++14
- CUB
- OpenMP

### Description
This example uses the [combinatorial number system](https://en.wikipedia.org/wiki/Combinatorial_number_system) to produce all possible trees in a complete graph in parallel. Trees are created in lexicograpical order using an independent index. Once created the tree is then analyzed to determine if it is a valid [spanning tree](https://en.wikipedia.org/wiki/Spanning_tree) i.e. no cycles or disconnects. The spanning tree is validated by using Kruskal's algorithm to find the [minimum spanning tree](https://en.wikipedia.org/wiki/Minimum_spanning_tree), which all edge weights set to a value of 1. If the cost of the minimum spanning tree is equal to the cost of the number of edges in the tree, the tree is said to be valid. The number of valid spanning trees returned to compared to [Cayley's formula](https://en.wikipedia.org/wiki/Cayley%27s_formula).

### Optimizations
1. The denominator of the binomial coefficient is precalculate on the CPU and copied to constant memory.
2. The number of valid spanning trees are summed using the [CUB BlockReduce](https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html) before using [CUB DeviceReduce](https://nvlabs.github.io/cub/structcub_1_1_device_reduce.html). This minimizes the number of results stored to global memory per block and minimizes the number of loads when DeviceReduce is called.
3. OpenMP is used to utilize multiple GPUs on a system.

### Results
Expected results
```bash
combos = 45; chains = 886163135
Number of GPUs = 2
Runtime = 2598.36 ms
Trees Found = 100000000: Cayley's formula = 100000000
Found correct number of trees!!
```

### Limitations
This example only works up to a 10 vertex complete graph because the number of possible combinations is less than the upper limit of an **unsigned int** or 4294967295.
