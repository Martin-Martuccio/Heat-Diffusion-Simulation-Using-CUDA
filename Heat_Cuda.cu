#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Simple define to index into a 1D array from 2D space
#define I2D(num, c, r) ((r)*(num)+(c))

/*
  Cuda permette di sfruttare la GPU per lanciare diversi threads su una medesima operazione, ma su dati differenti.
  Per far questo bisogna definire il numero di blocchi e il numero di threads per ogni blocco.
  I threads sono raggrupati in batch chiamati warp e sono 32.

  Per N elementi da calcolare nella CPU, necessito di lanciare N threads.

  Solitamente bisogna gestire la copia e lo spostamento dei dati da una memoria all'altra (CPU - GPU).
  Se usassimo CudaMallocManaged, tutto questo viene gestito in maniera automatica.

  Gerarchia:
  - I thread sono raggruppati in blocchi.
  - I blocchi possono essere  1D, 2D o 3D.
  - I blocchi sono  organizzati in una griglia, che può essere 1D, 2D o 3D.

  Ogni thread e blocco ha un identificatore unico:
  - I thread all'interno di un blocco sono indicizzati usando la variabile threadIdx, che contiene le componenti x, y e z per i blocchi multidimensionali.
  - I blocchi all'interno di una griglia sono indicizzati usando la variabile blockIdx, con componenti x, y e z per le griglie multidimensionali.

  La dimensione di ciascuna dimensione è accessibile tramite:
  - blockDim, che fornisce le dimensioni di un blocco in termini di thread.
  - gridDim, che fornisce le dimensioni della griglia in termini di blocchi.

    Griglia (2D) [composta da blocchi]

    | Block (0,0) | Block (1,0) | Block (2,0) |
    |-------------|-------------|-------------|
    | Block (0,1) | Block (1,1) | Block (2,1) |
    |-------------|-------------|-------------|
    | Block (0,2) | Block (1,2) | Block (2,2) |

    Blocco (2,1) [composta da Thread]
    |-Thread (0,0) Thread (1,0) Thread (2,0) Thread (3,0)|
    |-Thread (0,1) Thread (1,1) Thread (2,1) Thread (3,1)|
    |-Thread (0,2) Thread (1,2) Thread (2,2) Thread (3,2)|
    |-Thread (0,3) Thread (1,3) Thread (2,3) Thread (3,3)|

*/

__global__ void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (nj-2)*(ni-2)){

      int i = idx % (ni-2) + 1;
      int j = idx / (ni-2) + 1;

      if (i < ni-1 && j < nj-1){
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
      }
    }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  int istep;
  int nstep = 200; // number of time steps

  // Specify our 2D dimensions (MODIFICARE CON 10 000 X 10 000 E CON 30 000 X 30 000)
  const int ni = 1000;
  const int nj = 1000;
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const int size = ni * nj * sizeof(float);

  temp1_ref = (float*)malloc(size);
  temp2_ref = (float*)malloc(size);

//  GESTIONE DELLE VARIABILI CONDIVISE CPU-GPU

  cudaMallocManaged(&temp1, size);
  cudaMallocManaged(&temp2, size);
  
/*  
    DEFINITION OF THE BLOCKS AND THREADS
    Threads must be multiple of 32 (Cuda GuideLine) 
 */

  // Total number of internal points in the domain
  int numPoints = (ni-2) * (nj-2); 
  dim3 threads(64);  
  dim3 dimblock((numPoints + threads.x - 1) / threads.x);  // Dimension of the block proportional to the number of points

  // print some info
  printf("threads: %u \n", threads.x);
  printf("block: %u \n", dimblock.x);
  printf("total num of threads: %u\n", threads.x*dimblock.x);
  printf("total num of pixels: %d\n", ni*nj);
  printf("total num of internal pixels: %d\n", (ni-2) * (nj-2));
 
  //Identify some events to measure the time
  cudaEvent_t start_ref, stop_ref; 
  cudaEvent_t start_mod, stop_mod;
  cudaEventCreate(&start_ref);
  cudaEventCreate(&stop_ref);
  cudaEventCreate(&start_mod);
  cudaEventCreate(&stop_mod);

  // Initialize with random data
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }

  cudaEventRecord(start_ref);

  // Execute the CPU-only reference version
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    temp_tmp = temp1_ref; // swap the temperature pointers
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }

  cudaEventRecord(stop_ref);  
  cudaEventSynchronize(stop_ref);
  float elapsed_ref = 0;
  // Compute the elapsed time
  cudaEventElapsedTime(&elapsed_ref, start_ref, stop_ref);
  // Print the elapsed time
  printf("Elapsed ref time: %f microseconds\n", elapsed_ref) ; 

  cudaEventRecord(start_mod);

  // Execute the modified version using same data
  for (istep=0; istep < nstep; istep++) {

    // original problem without blocks and threads
    //step_kernel_mod(ni, nj, tfac, temp1, temp2);
    
    // i have to specify the number of threads e dimblock
    step_kernel_mod<<< dimblock , threads >>>(ni, nj, tfac, temp1, temp2);
    cudaDeviceSynchronize();

    // swap the temperature pointers
    temp_tmp = temp1;
    temp1 = temp2;
    temp2= temp_tmp;
  }

  cudaEventRecord(stop_mod); 
  cudaEventSynchronize(stop_mod);
  float elapsed_mod = 0;
  cudaEventElapsedTime(&elapsed_mod, start_mod, stop_mod);
  printf("Elapsed mod time: %f microseconds\n", elapsed_mod) ; 

  // Check for errors (all CUDA API calls return an error code)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

  float maxError = 0;
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  
  
  // se omettimo il calcolo dell'errore possiamo non deallocare temp1_ref e temp2_ref
  cudaFree(temp1);
  cudaFree(temp2);
  
  free( temp1_ref );
  free( temp2_ref );

  return 0;
}
