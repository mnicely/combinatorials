/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#include <cmath>
#include <cooperative_groups.h>
#include <numeric>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                    "with "                                                                                           \
                    "%s (%d).\n",                                                                                     \
                    #call,                                                                                            \
                    __LINE__,                                                                                         \
                    __FILE__,                                                                                         \
                    cudaGetErrorString( status ),                                                                     \
                    status );                                                                                         \
    }
#endif  // CUDA_RT_CALL
// *************** FOR ERROR CHECKING *******************

/*
* Factorial required for combinatorial number system
*/
constexpr size_t factorial( const int &n ) {
    return ( n <= 1 ) ? 1 : ( n * factorial( n - 1 ) );
}

/*
* The more we can do at compile time the better
*/
constexpr uint k_tpb { 512 };
constexpr uint k_ept { 1 };
constexpr uint k_numNodes { 10 };
constexpr uint k_numEdges { 9 };
constexpr uint k_numVertices { ( k_numNodes * ( k_numNodes - 1 ) ) >> 1 };
constexpr uint k_treeStart { k_numVertices - 1 };

/*
* Structure hold edge and angle information
*/
typedef struct fEdgeData_t {
    int a {};
    int b {};
} edgeData;

/*
* Structure to hold all combos and scores for testing
*/
typedef struct fgpuData_t {
    uint         offset {};
    uint         treesMaxDevice {};
    cudaStream_t streams {};
} gpuData;

/*
* Constant memory holds read-only data in cached global memory
*/
__constant__ double c_denominator[k_numEdges];
__constant__ edgeData c_edges[k_numVertices];

/*
* Device memory to hold total trees per device
*/
__device__ unsigned long long int d_totalTreesPerBlock {};

/*
* Calculate binomial coefficients
* Care must be taken to ensure arithmetic doesn't
* exceed what a data type can hold
*/
__host__ __device__ size_t nchoosek( const uint &numerator, const double &denominator, const int &loops ) {

    size_t n { static_cast<size_t>( numerator ) };
    for ( int f = 1; f < loops; f++ )
        n *= ( numerator - f );
    return ( static_cast<size_t>( static_cast<double>( n ) * denominator ) );  // Precalculate
}

/*
* This function is find a combination based on a given id
* It uses the combinatorial number system and produces
* an answer in lexicographic order
*/
__host__ __device__ void getTree( const uint &maxTrees, const uint &id, uint *combo ) {

    uint n {};
    uint key { maxTrees - id - 1 };
#pragma unroll k_numEdges
    for ( int e = 0; e < k_numEdges; e++ ) {
        uint numerator { k_treeStart };
        while ( true ) {
            // The denominator must start at the end of the array
            n = nchoosek( numerator, c_denominator[e], ( k_numEdges - e ) );
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
__device__ uint find( const uint &x, uint *parent ) {

    if ( parent[x] != x )
        parent[x] = find( parent[x], parent );
    return ( parent[x] );
}

/*
* This function is used to generate a sequence on the device
*/
__device__ void gen_seq( uint *parent ) {
#pragma unroll  k_numNodes
    for ( int i = 0; i < k_numNodes; i++ ) {
        parent[i] = i;
    }
}

/*
* CUDA kernel to determine minimum angle diversity score for a given block.
* Since we are using CUB BlockRadixSort and grid-stride looping
* we need to pad the last block in the last grid
*/
__launch_bounds__( k_tpb ) __global__
    void tdoa( const uint offset, const uint treesPerDevice, const uint maxTrees, const uint padding ) {

    const auto block = cooperative_groups::this_thread_block( );

    // Specialize BlockRadixSort for a 1D block of k_tpb threads of type int
    typedef cub::BlockReduce<uint, k_tpb> BlockReduce;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockReduce::TempStorage temp_storage;

    uint gid { blockIdx.x * blockDim.x + threadIdx.x };  // Change name
    uint stride { blockDim.x * gridDim.x };
    uint newTid[k_ept] {};

    for ( uint tid = gid; tid < padding; tid += stride ) {

        uint score[k_ept] {};

        // To increase Instruction Level Parallelism (ILP)
        // Each thread will calculate multiple combination scores
        for ( int s = 0; s < k_ept; s++ ) {
            newTid[s] = offset + tid * k_ept + s;

            // Ensure only valid combinations are checked
            if ( newTid[s] < treesPerDevice ) {

                // Find tree
                uint combo[k_numEdges] {};
                getTree( maxTrees, newTid[s], combo );

                // Determine is chain is valid spanning tree
                // Use Kruskal's algorithm with each edge weight set to 1
                uint parent[k_numNodes] {};
                gen_seq( parent );

                uint cost {};
#pragma unroll k_numEdges
                for ( int e = 0; e < k_numEdges; e++ ) {
                    uint findX = find( c_edges[combo[e]].a, parent );
                    uint findY = find( c_edges[combo[e]].b, parent );
                    if ( findX == findY )
                        continue;
                    cost++;
                    parent[findX] = findY;
                }

                // If the minimum spanning tree has a cost
                // less than the number of edges, than it
                // is not a valid spanning tree
                if ( cost == k_numEdges )
                    score[s] = 1u;
            }
        }

        uint totalTrees { BlockReduce( temp_storage ).Sum( score ) };

        // Once BlockReduce is finished, the sum
        // is now stored in the first address of the score array
        // in the first thread in the block. That value is
        // then stored to global memory to the address pertaining
        // to that blockId
        if ( !block.thread_rank( ) )
            atomicAdd( &d_totalTreesPerBlock, totalTrees );

        // We need to sync the block again because we are using grid-stride looping.
        block.sync( );  // Sync block to reuse tempStorage for BlockReduce
    }
}

int main( int arg, char **argv ) {

    size_t maxTrees { nchoosek( k_numVertices, ( 1.0 / factorial( k_numEdges ) ), k_numEdges ) };
    if ( maxTrees >= UINT_MAX ) {  // maxTrees can't be larger than 4294967295
        std::printf( "combos = %d; chains = %lu\n", k_numVertices, maxTrees );
        throw std::runtime_error( "The number is chains to test is larger than uint can hold.\n" );
    }
    std::printf( "Number of Vertices = %d\n", k_numVertices );
    std::printf( "Max Trees Possible = %lu\n", maxTrees );

    // Precompute all possible denominators recipicals
    // Multiplication requires less operations than division
    double denominator[k_numEdges] {};
    for ( int i = k_numEdges; i > 0; i-- )
        denominator[k_numEdges - i] = 1.0 / factorial( i );

    // Generate all possible edges of graph in lexicographic order
    edgeData edges[k_numVertices] {};
    int      start { 1 };
    int      idx {};
    for ( int i = 0; i < k_numNodes; i++ ) {
        for ( int j = start; j < k_numNodes; j++ ) {
            edges[idx].a = i;
            edges[idx].b = j;
            idx++;
        }
        start++;
    }

    // Get device attributes
    int numDevices {};
    int numSMs {};
    CUDA_RT_CALL( cudaGetDeviceCount( &numDevices ) );
    CUDA_RT_CALL( cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 ) );
    std::printf( "Number of GPUs = %d\n", numDevices );

    // Padding blocks so we can use CUB BlockReduce in CUDA kernel
    uint padding { static_cast<uint>( std::ceil( static_cast<double>( maxTrees ) / k_ept / numDevices / k_tpb ) *
                                    k_tpb ) };

    // Will store final results for each GPU using pinned memory
    // Pinned memory is required for async copies
    thrust::host_vector<uint, thrust::cuda::experimental::pinned_allocator<uint>> h_totalTrees( numDevices, 0 );

    // Divide up work between all GPUs
    gpuData gpuWork[numDevices] {};
    uint    chunk { static_cast<uint>( maxTrees / numDevices ) };

    for ( int d = 0; d < numDevices; d++ ) {
        if ( d < ( numDevices - 1 ) ) {
            gpuWork[d].offset         = d * chunk;
            gpuWork[d].treesMaxDevice = ( d + 1 ) * chunk;
        } else {
            gpuWork[d].offset         = d * chunk;
            gpuWork[d].treesMaxDevice = maxTrees;
        }
    }

    // Launch one CPU thread per GPU
    omp_set_num_threads( numDevices );
#pragma omp parallel
    {
        int ompId { omp_get_thread_num( ) };

        // We must set the device in each thread
        // so the correct CUDA context is visible
        CUDA_RT_CALL( cudaSetDevice( ompId ) );
        CUDA_RT_CALL( cudaStreamCreate( &gpuWork[ompId].streams ) );

        // Copy denominators to constant memory
        CUDA_RT_CALL( cudaMemcpyToSymbolAsync( c_denominator,
                                            denominator,
                                            sizeof( double ) * k_numEdges,
                                            0,
                                            cudaMemcpyHostToDevice,
                                            gpuWork[ompId].streams ) );

        // Copy angles to constant memory
        CUDA_RT_CALL( cudaMemcpyToSymbolAsync(
            c_edges, edges, sizeof( edgeData ) * k_numVertices, 0, cudaMemcpyHostToDevice, gpuWork[ompId].streams ) );
    }

    // Start timer
    cudaEvent_t startEvent { nullptr };
    cudaEvent_t stopEvent { nullptr };
    float       elapsed_gpu_ms {};

    CUDA_RT_CALL( cudaEventCreate( &startEvent, cudaEventBlockingSync ) );
    CUDA_RT_CALL( cudaEventCreate( &stopEvent, cudaEventBlockingSync ) );

    CUDA_RT_CALL( cudaEventRecord( startEvent ) );

#pragma omp parallel
    {
        int ompId { omp_get_thread_num( ) };
        CUDA_RT_CALL( cudaSetDevice( ompId ) );

        // The number of blocks launched is based on the number of
        // Streaming Multiprocessor available on the GPU
        int threadPerBlock { k_tpb };
        int blocksPerGrid { numSMs * 32 };

        void *args[] { &gpuWork[ompId].offset, &gpuWork[ompId].treesMaxDevice, &maxTrees, &padding };

        CUDA_RT_CALL( cudaLaunchKernel(
            reinterpret_cast<void *>( &tdoa ), blocksPerGrid, threadPerBlock, args, 0, gpuWork[ompId].streams ) );

        CUDA_RT_CALL( cudaMemcpyFromSymbolAsync( &h_totalTrees[ompId],
                                                d_totalTreesPerBlock,
                                                sizeof( uint ),
                                                0,
                                                cudaMemcpyDefault,
                                                gpuWork[ompId].streams ) );

        // Sync each stream to ensure data copy is complete
        CUDA_RT_CALL( cudaStreamSynchronize( gpuWork[ompId].streams ) );
    }

    // Stop timer
    CUDA_RT_CALL( cudaEventRecord( stopEvent ) );
    CUDA_RT_CALL( cudaEventSynchronize( stopEvent ) );

    CUDA_RT_CALL( cudaEventElapsedTime( &elapsed_gpu_ms, startEvent, stopEvent ) );
    std::printf( "Runtime = %0.2f ms\n", elapsed_gpu_ms );

    uint h_total { std::accumulate( h_totalTrees.begin( ), h_totalTrees.end( ), 0u ) };

    uint cayley =
        static_cast<uint>( std::pow( static_cast<double>( k_numNodes ), static_cast<double>( k_numNodes - 2 ) ) );

    if ( ( k_numNodes - 1 ) == k_numEdges )
        std::printf( "%d trees found = (%d)\n", h_total, cayley );
    else
        std::printf( "%d forests found\n", h_total );

        // Data clean up
#pragma omp parallel
    {
        int ompId { omp_get_thread_num( ) };
        CUDA_RT_CALL( cudaSetDevice( ompId ) );
        CUDA_RT_CALL( cudaStreamDestroy( gpuWork[ompId].streams ) );
    }

    return ( EXIT_SUCCESS );
}
