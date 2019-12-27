/*
 * tdoa.cu
 *
 *  Created on: Dec 16, 2019
 *      Author: mnicely
 */

#include <cmath>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <numeric>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

/*
 * Factorial required for combinatorial number system
 */
constexpr double factorial( const int &n ) {
    return ( n <= 1 ) ? 1 : ( n * factorial( n - 1 ) );
}

/*
 * The more we can do at compile time the better
 */
constexpr unsigned int k_tpb { 512 };
constexpr int          k_ept { 8 };
constexpr int          k_numNodes { 10 };
constexpr int          k_numEdges { k_numNodes - 1 };
constexpr int          k_numCombos { ( k_numNodes * ( k_numNodes - 1 ) ) / 2 };
constexpr int          k_treeStart { k_numCombos - 1 };

/*
 * Structure hold edge and angle information
 */
typedef struct fEdges_t {
    int a {};
    int b {};
} edgeData;

/*
 * Structure holds array information for CUB Device Radix Sort
 * It will hold arrays for each GPU in each CPU thread (OpenMP)
 */
typedef struct fCUB_t {
    unsigned int *d_totalTreesPerBlock {};
    unsigned int *d_totalTreesPerDevice {};
    void *        d_temp_storage     = NULL;
    size_t        temp_storage_bytes = 0;
} cubData;

/*
 * Constant memory holds read-only data in cached global memory
 */
__constant__ double c_denominator[k_numEdges];
__constant__ edgeData c_edges[k_numCombos];

/*
 * Calculate binomial coefficients
 * Care must be taken to ensure arithmetic doesn't
 * exceed what a data type can hold
 */
__host__ __device__ unsigned int nchoosek( const int &numerator, const double &denominator, const int &loops ) {

    unsigned long long n { static_cast<unsigned long long>( numerator ) };
    for ( int f = 1; f < loops; f++ )
        n *= static_cast<unsigned long long>( numerator - f );
    return ( static_cast<unsigned int>( static_cast<double>( n ) * denominator ) ); // Precalculate
}

/*
 * This function is find a combination based on a given id
 * It uses the combinatorial number system and produces
 * an answer in lexicographic order
 */
__host__ __device__ void
         getTree( const unsigned int &numChains, const unsigned int &id, const double *denominator, int *combo ) {

    unsigned int n {};
    unsigned int key { numChains - id - 1 };
#pragma unroll k_numEdges
    for ( int e = 0; e < k_numEdges; e++ ) {
        int numerator = k_treeStart;
        while ( true ) {
            // The denominator must start at the end of the array
            n = nchoosek( numerator, denominator[e], ( k_numEdges - e ) );
            if ( n <= key ) {
                combo[e] = k_treeStart - numerator;
                key -= n;
                break;
            }
            numerator--;
        }
    }
}

/*
 * This function is used by Kruskal's Minimum Spanning Tree algorithm
 */
__device__ int find( const int &x, int *parent ) {

    if ( parent[x] != x )
        parent[x] = find( parent[x], parent );
    return ( parent[x] );
}

/*
 * CUDA kernel to determine minimum angle diversity score for a given block.
 * Since we are using CUB BlockRadixSort and grid-stride looping
 * we need to pad the last block in the last grid
 */
__launch_bounds__( k_tpb ) __global__ void tdoa( const unsigned int offset,
                                                 const unsigned int numChains,
                                                 const unsigned int padding,
                                                 unsigned int *__restrict__ d_totalTreesPerBlock ) {

    const auto block = cooperative_groups::this_thread_block( );

    // Specialize BlockRadixSort for a 1D block of k_tpb threads of type int
    typedef cub::BlockReduce<unsigned int, k_tpb> BlockReduce;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockReduce::TempStorage temp_storage;

    unsigned int gid { blockIdx.x * blockDim.x + threadIdx.x }; // Change name
    unsigned int stride { blockDim.x * gridDim.x };
    unsigned int blockId { blockIdx.x };
    unsigned int newTid[k_ept] {};
    unsigned int score[k_ept] {};

    for ( unsigned int tid = gid; tid < padding; tid += stride ) {

        // To increase Instruction Level Parallelism (ILP)
        // Each thread will calculate multiple combination scores
        for ( int s = 0; s < k_ept; s++ ) {
            newTid[s] = offset + tid * k_ept + s;

            // Ensure only valid combinations are checked
            if ( newTid[s] < numChains ) {

                // Find tree
                int combo[k_numEdges] {};
                getTree( numChains, newTid[s], c_denominator, combo );

                // Determine is chain is valid spanning tree
                // Use Kruskal's algorithm with each edge weight set to 1
                // thrust::seq allows us to run Thrust functions
                // in individual threads
                int parent[k_numNodes] {};
                thrust::sequence( thrust::seq, parent, parent + k_numNodes, 0 );

                int cost {};
#pragma unroll k_numEdges
                for ( int e = 0; e < k_numEdges; e++ ) {
                    int findX = find( c_edges[combo[e]].a, parent );
                    int findY = find( c_edges[combo[e]].b, parent );
                    if ( findX == findY ) {

                        continue;
                    }
                    // printf("tid = %d: %d %d\n", newTid[s], findX, findY);
                    cost++;
                    parent[findX] = findY;
                }

                // If the minimum spanning tree has a cost
                // less than the number of edges, than it
                // is not a valid spanning tree
                if ( cost == k_numEdges ) {
                    score[s] = 1u;
                    // std::printf("tid %u = %d %d %d\n", newTid[s], combo[0], combo[1], combo[2]);
                } else
                    score[s] = 0u;

            } else
                score[s] = 0u; // For thread ids larger than the number of required combinations
        }

        unsigned int totalTrees = BlockReduce( temp_storage ).Sum( score );

        // Once BlockReduce is finished, the sum
        // is now stored in the first address of the score array
        // in the first thread in the block. That value is
        // then stored to global memory to the address pertaining
        // to that blockId
        if ( threadIdx.x == 0 )
            d_totalTreesPerBlock[blockId] = totalTrees;

        // Because we are using grid-stride looping, where we
        // reuse blocks in a grid, to minimize launch overhead,
        // we need to increment the blockId
        blockId += gridDim.x;

        // We need to sync the block again because we are using grid-stride looping.
        block.sync( ); // Sync block to reuse tempStorage for BlockReduce
    }
}

int main( int arg, char **argv ) {

    // Determine the number of combinations that will be evaluated
    unsigned int numChains { nchoosek( k_numCombos, ( 1 / factorial( k_numEdges ) ), k_numEdges ) };
    if ( numChains >= UINT_MAX ) { // numChains can't be larger than 4294967295
        std::printf( "combos = %d; chains = %u\n", k_numCombos, numChains );
        throw std::runtime_error( "The number is chains to test is larger than unsigned int can hold.\n" );
    }
    std::printf( "combos = %d; chains = %u\n", k_numCombos, numChains );

    // Precompute all possible denominators recipicals
    // Multiplication requires less operations than division
    double denominator[k_numEdges] {};
    for ( int i = k_numEdges; i > 0; i-- )
        denominator[k_numEdges - i] = 1 / factorial( i );

    // Generate all possible edges of graph in lexicographic order
    edgeData edges[k_numCombos] {};
    int      start = 1;
    int      idx   = 0;
    for ( int i = 0; i < k_numNodes; i++ ) {
        for ( int j = start; j < k_numNodes; j++ ) {
            edges[idx].a = i;
            edges[idx].b = j;
            idx++;
        }
        start++;
    }
    // for ( int i = 0; i < k_numCombos; i++ )
    //     std::printf("%d %d\n", edges[i].a, edges[i].b);

    // Get device attributes
    int numDevices {};
    int numSMs {};
    checkCudaErrors( cudaGetDeviceCount( &numDevices ) );
    checkCudaErrors( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );
    std::printf( "Number of GPUs = %d\n", numDevices );

    // numDevices =1;

    // Create streams
    const int    num_streams = numDevices;
    cudaStream_t streams[num_streams];

    // Padding blocks so we can use CUB BlockReduce in CUDA kernel
    unsigned int padding { static_cast<unsigned int>(
        std::ceil( static_cast<double>( numChains / k_ept / numDevices ) / static_cast<double>( k_tpb ) ) * k_tpb ) };

    // Will store final results for each GPU using pinned memory
    // Pinned memory is required for async copies
    thrust::host_vector<unsigned int, thrust::cuda::experimental::pinned_allocator<unsigned int>> h_totalTrees(
        numDevices, 0 );

    unsigned int numBlocksRequired { static_cast<unsigned int>( padding / k_tpb ) };

    cubData cubStruct[numDevices] {};

    // Launch one CPU thread per GPU
    omp_set_num_threads( numDevices );
#pragma omp parallel
    {
        int ompId = omp_get_thread_num( );

        // We must set the device in each thread
        // so the correct CUDA context is visible
        checkCudaErrors( cudaSetDevice( ompId ) );
        checkCudaErrors( cudaStreamCreate( &streams[ompId] ) );

        checkCudaErrors( cudaMalloc( reinterpret_cast<void **>( &cubStruct[ompId].d_totalTreesPerBlock ),
                                     numBlocksRequired * sizeof( unsigned int ) ) );
        checkCudaErrors( cudaMalloc( reinterpret_cast<void **>( &cubStruct[ompId].d_totalTreesPerDevice ),
                                     numBlocksRequired * sizeof( unsigned int ) ) );

        // Copy denominators to constant memory
        checkCudaErrors( cudaMemcpyToSymbolAsync(
            c_denominator, denominator, k_numEdges * sizeof( double ), 0, cudaMemcpyHostToDevice, streams[ompId] ) );

        // Copy angles to constant memory
        checkCudaErrors( cudaMemcpyToSymbolAsync(
            c_edges, edges, k_numCombos * sizeof( edgeData ), 0, cudaMemcpyHostToDevice, streams[ompId] ) );
    }

    // Start timer
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent  = nullptr;
    float       elapsed_gpu_ms {};

    cudaEventCreate( &startEvent, cudaEventBlockingSync );
    cudaEventCreate( &stopEvent, cudaEventBlockingSync );

    cudaEventRecord( startEvent );

#pragma omp parallel
    {
        int ompId = omp_get_thread_num( );
        checkCudaErrors( cudaSetDevice( ompId ) );

        // The number of blocks launched is based on the number of
        // Streaming Multiprocessor available on the GPU
        dim3 threadPerBlock { k_tpb };
        dim3 blocksPerGrid { static_cast<uint>( 20 * numSMs ) };

        // Each GPU will work on a portion of the possible combinations
        unsigned int offset { ompId * padding * k_ept };

        void *args[] { &offset, &numChains, &padding, &cubStruct[ompId].d_totalTreesPerBlock };

        checkCudaErrors( cudaLaunchKernel(
            reinterpret_cast<void *>( &tdoa ), blocksPerGrid, threadPerBlock, args, 0, streams[ompId] ) );

        checkCudaErrors( cub::DeviceReduce::Sum( cubStruct[ompId].d_temp_storage,
                                                 cubStruct[ompId].temp_storage_bytes,
                                                 cubStruct[ompId].d_totalTreesPerBlock,
                                                 cubStruct[ompId].d_totalTreesPerDevice,
                                                 numBlocksRequired,
                                                 streams[ompId],
                                                 false ) );

        // Allocate temporary storage
        checkCudaErrors( cudaMalloc( &cubStruct[ompId].d_temp_storage, cubStruct[ompId].temp_storage_bytes ) );

        // Run sorting operation
        checkCudaErrors( cub::DeviceReduce::Sum( cubStruct[ompId].d_temp_storage,
                                                 cubStruct[ompId].temp_storage_bytes,
                                                 cubStruct[ompId].d_totalTreesPerBlock,
                                                 cubStruct[ompId].d_totalTreesPerDevice,
                                                 numBlocksRequired,
                                                 streams[ompId],
                                                 false ) );

        // Scores and ids are copied back to the CPU in parallel
        checkCudaErrors( cudaMemcpyAsync( &h_totalTrees[ompId],
                                          cubStruct[ompId].d_totalTreesPerDevice,
                                          sizeof( unsigned int ),
                                          cudaMemcpyDeviceToHost,
                                          streams[ompId] ) );

        // Sync each stream to ensure data copy is complete
        checkCudaErrors( cudaStreamSynchronize( streams[ompId] ) );
    }

    // Stop timer
    cudaEventRecord( stopEvent );
    cudaEventSynchronize( stopEvent );

    cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent );
    std::printf( "Runtime = %0.2f ms\n", elapsed_gpu_ms );

    unsigned int h_total { std::accumulate( h_totalTrees.begin( ), h_totalTrees.end( ), 0u ) };

    unsigned int cayley =
        static_cast<unsigned int>( pow( static_cast<double>( k_numNodes ), static_cast<double>( k_numNodes - 2 ) ) );
    std::printf( "Trees Found = %u: Cayley's formula = %u\n", h_total, cayley );

    if ( h_total == cayley )
        std::printf( "Found correct number of trees!!\n" );
    else
        std::printf( "Error!!\n" );

        // Data clean up
#pragma omp parallel
    {
        int ompId = omp_get_thread_num( );
        checkCudaErrors( cudaSetDevice( ompId ) );
        checkCudaErrors( cudaFree( cubStruct[ompId].d_temp_storage ) );
        checkCudaErrors( cudaFree( cubStruct[ompId].d_totalTreesPerBlock ) );
        checkCudaErrors( cudaFree( cubStruct[ompId].d_totalTreesPerDevice ) );
        checkCudaErrors( cudaStreamDestroy( streams[ompId] ) );
    }

    return ( EXIT_SUCCESS );
}
