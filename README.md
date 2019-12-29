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

## Optimizations
1. The denominator of the binomial coefficient is precalculate on the CPU and copied to constant memory.
2. The number of valid spanning trees/forests are summed within each block using the [CUB BlockReduce](https://nvlabs.github.io/cub/classcub_1_1_block_reduce.html).
3. OpenMP is used to utilize multiple GPUs on a system.

## Description: combinatorials
This example uses the [combinatorial number system](https://en.wikipedia.org/wiki/Combinatorial_number_system) to produce all possible trees in a complete graph in parallel. Trees are created in lexicograpical order using an independent index. Once created, the tree is then analyzed to determine if it is a valid [spanning tree](https://en.wikipedia.org/wiki/Spanning_tree) i.e. no cycles or disconnects. The spanning tree is validated by using Kruskal's algorithm to find the [minimum spanning tree](https://en.wikipedia.org/wiki/Minimum_spanning_tree), which all edge weights set to a value of 1. If the cost of the minimum spanning tree is equal to the cost of the number of edges in the tree, the tree is said to be valid. 

If the number of edges, **k**, is less than the number of vertices - 1, **n - 1**, then Kruskal's algorithm will return a valid forest with no cycles. If the cost of the forest is equal to the cost of the number of edges in the forest, the forest is said to be valid. 

### Results
Expected results
```bash
Number of Vertices = 45
Max Trees Possible = 886163135
Number of GPUs = 2
Runtime = 2556.89 ms
100000000 trees found
```

### Limitations
1. This example can only analysis a maximum number of possible trees equal to the the upper limit of an **unsigned int** or 4294967295.

## Description: combosCheck
This example performs all the tasks as combinatorials include checking all trees/forest for cycles. If trees are produced, the number of valid spanning trees returned to compared to [Cayley's formula](https://en.wikipedia.org/wiki/Cayley%27s_formula).

### Results
Expected results
```bash
Number of Vertices = 36
Max Trees Possible = 30260340
Number of GPUs = 2
Runtime = 171.48 ms
4782969 trees found
Total trees equals Cayley's formula.
Found correct number of trees!!
0 cyclics found!
```

### Limitations
1. This example can only analysis a maximum number of possible trees equal to the the upper limit of an **unsigned int** or 4294967295.

2. If you receive the following error, your GPU doesn't have enough memory to store edges for all trees it is processing.
```bash
CUDA error at combosCheck.cu:301 code=2(cudaErrorMemoryAllocation) "cudaMalloc( reinterpret_cast<void **>( &comboStruct[ompId].d_treeCombos ), gpuWork[ompId].treesPerDevice * k_numEdges * sizeof( unsigned int ) )" 
```
