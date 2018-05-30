
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <CL/cl.h>

#define SEPARATOR ("----------------------------------------------------------------------\n")

#define min(A,B) ((A) < (B) ? (A) : (B))

int GROUP_SIZE = 128;

static char *
slurpFile(const char *filename)
{
   struct stat statbuf;
   FILE        *fh;
   char        *source;

   fh = fopen(filename, "r");
   if (fh == 0)
      return 0;

   stat(filename, &statbuf);
   source = (char *) malloc(statbuf.st_size + 1);
   fread(source, statbuf.st_size, 1, fh);
   source[statbuf.st_size] = '\0';

   return source;
}

#define N_BINS 256
#define N_X_TILES 4
#define N_Y_TILES 4
#define N_TILES (N_X_TILES * N_Y_TILES)
#define WIDTH 320
#define HEIGHT 320

uint32_t nextPowerOfTwo(uint32_t val)
{
   val--;
   val |= val >> 1;
   val |= val >> 2;
   val |= val >> 4;
   val |= val >> 8;
   val |= val >> 16;
   val++;

   return val;
}

cl_int prefixSum(cl_command_queue queue, cl_kernel k_prefixSum,
                 uint8_t *d_data, uint8_t *d_working, int iHist, int size)
{
   cl_int err = CL_SUCCESS;

   uint8_t *d_input = d_working;
   uint8_t *d_output = d_data;

   int start = N_BINS * iHist;
   for (int offset = 1;
        // Ensure that we always end with the d_output in the d_data array.
        d_output == d_working || offset < size;
        offset <<= 1)
   {
      /* printf("offset: %d, d_output == d_working: %d\n", offset, d_output == d_working); */
      // Swap d_input and d_output each iteration to avoid a copy
      uint8_t *temp = d_input;
      d_input = d_output;
      d_output = temp;

      err |= clSetKernelArg(k_prefixSum, 0, sizeof(cl_mem), &d_input);
      err |= clSetKernelArg(k_prefixSum, 1, sizeof(cl_int), &start);
      err |= clSetKernelArg(k_prefixSum, 2, sizeof(cl_int), &size);
      err |= clSetKernelArg(k_prefixSum, 3, sizeof(cl_int), &offset);
      err |= clSetKernelArg(k_prefixSum, 4, sizeof(cl_mem), &d_output);

      const size_t gws[] = {size};
      const size_t lws[] = {32};
      /* const size_t gws[] = {size}; */
      err |= clEnqueueNDRangeKernel(queue, k_prefixSum, 1, NULL, gws, lws, 0, NULL, NULL);
   }
   return err;
}

/* Exclusive version.  Just do an inclusive sum and then shift over the elements
 * inserting the defined operator's zero into the first element. */
cl_int prefixSumEx(cl_command_queue queue, cl_kernel k_prefixSum, cl_kernel k_prefixSumIncToExc,
                   uint8_t *d_input, uint8_t *d_output, int iHist, int size)
{
   prefixSum(queue, k_prefixSum, d_input, d_output, iHist, size);

   int start = N_BINS * iHist;
   cl_int err = CL_SUCCESS;
   err |= clSetKernelArg(k_prefixSumIncToExc, 0, sizeof(cl_mem), &d_input);
   err |= clSetKernelArg(k_prefixSumIncToExc, 1, sizeof(cl_int), &start);
   err |= clSetKernelArg(k_prefixSumIncToExc, 2, sizeof(cl_int), &size);
   err |= clSetKernelArg(k_prefixSumIncToExc, 3, sizeof(cl_mem), &d_output);

   const size_t gws[] = {size};
   const size_t lws[] = {32};
   err |= clEnqueueNDRangeKernel(queue, k_prefixSumIncToExc, 1, NULL, gws, NULL, 0, NULL, NULL);

   return err;
}

void histogram(cl_command_queue queue,
               cl_kernel k_histogram,
               cl_mem d_img,
               cl_int nBins,
               cl_int iTile,
               cl_int jTile,
               cl_mem d_hist,
               cl_int minVal,
               cl_int maxVal,
               cl_int width,
               cl_int height,
               cl_int tileWidth,
               cl_int tileHeight,
               cl_int ldaNumer,
               cl_int ldaDenom)
{
   cl_int iHist = jTile * N_X_TILES + iTile;

   cl_int err = CL_SUCCESS;
   err |= clSetKernelArg(k_histogram, 0, sizeof(cl_mem), &d_img);
   err |= clSetKernelArg(k_histogram, 1, sizeof(cl_int), &width);
   err |= clSetKernelArg(k_histogram, 2, sizeof(cl_int), &height);
   err |= clSetKernelArg(k_histogram, 3, sizeof(cl_int), &ldaNumer);
   err |= clSetKernelArg(k_histogram, 4, sizeof(cl_int), &ldaDenom);
   err |= clSetKernelArg(k_histogram, 5, sizeof(cl_mem), &d_hist);
   err |= clSetKernelArg(k_histogram, 6, sizeof(cl_int), &iHist);
   err |= clSetKernelArg(k_histogram, 7, sizeof(cl_int), &nBins);
   err |= clSetKernelArg(k_histogram, 8, sizeof(cl_int), &minVal);
   err |= clSetKernelArg(k_histogram, 9, sizeof(cl_int), &maxVal);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set arguments!\n");
      return EXIT_FAILURE;
   }
   else
   {
      size_t gws[] = {tileWidth, tileHeight};
      size_t goff[] ={iTile * WIDTH/N_X_TILES, jTile * HEIGHT/N_Y_TILES};
      printf("Histogram %d %d\n", iTile, jTile);
      err |= clEnqueueNDRangeKernel(queue, k_histogram, 2, goff, gws, NULL, 0, NULL, NULL);
      if (err != CL_SUCCESS)
      {
         printf("Error: Failed to enqueue kernel!\n");
         return EXIT_FAILURE;
      }
   }
}

void equalize(cl_command_queue queue,
              cl_kernel k_equalize,
              cl_mem d_img,
              cl_int width,
              cl_int height,
              cl_int iTile,
              cl_int jTile,
              cl_int tileWidth,
              cl_int tileHeight,
              cl_int ldaNumer,
              cl_int ldaDenom,
              cl_int minVal,
              cl_int maxVal,
              cl_mem d_cdf,
              cl_int nBins,
              cl_mem d_imgOut)
{
   cl_float normalizationFactor = 255.0 / (tileWidth * tileHeight);
   cl_int iCdf = jTile * N_X_TILES + iTile;

   cl_int err = CL_SUCCESS;
   err |= clSetKernelArg(k_equalize, 0, sizeof(cl_mem), &d_img);
   err |= clSetKernelArg(k_equalize, 1, sizeof(cl_int), &width);
   err |= clSetKernelArg(k_equalize, 2, sizeof(cl_int), &height);
   err |= clSetKernelArg(k_equalize, 3, sizeof(cl_int), &ldaNumer);
   err |= clSetKernelArg(k_equalize, 4, sizeof(cl_int), &ldaDenom);
   err |= clSetKernelArg(k_equalize, 5, sizeof(cl_int), &minVal);
   err |= clSetKernelArg(k_equalize, 6, sizeof(cl_int), &maxVal);
   err |= clSetKernelArg(k_equalize, 7, sizeof(cl_mem), &d_cdf);
   err |= clSetKernelArg(k_equalize, 8, sizeof(cl_int), &iCdf);
   err |= clSetKernelArg(k_equalize, 9, sizeof(cl_int), &nBins);
   err |= clSetKernelArg(k_equalize, 10, sizeof(cl_mem), &d_imgOut);
   err |= clSetKernelArg(k_equalize, 11, sizeof(cl_float), &normalizationFactor);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set arguments!\n");
      return EXIT_FAILURE;
   }
   {
      size_t goff[] ={iTile * WIDTH/N_X_TILES, jTile * HEIGHT/N_Y_TILES};
      size_t gws[] = {tileWidth, tileHeight};
      printf("Equalize %d %d\n", iTile, jTile);
      clEnqueueNDRangeKernel(queue, k_equalize, 2, goff, gws, NULL, 0, NULL, NULL);
   }
}

void limitContrast(cl_command_queue queue,
                   cl_kernel k_clipHistogram, cl_kernel k_addExcess,
                   cl_mem d_hist, cl_int iHist, cl_int nBins,
                   cl_int tileWidth, cl_int tileHeight,
                   cl_float contrastLimit, cl_mem d_excess,
                   cl_int minBin, cl_int maxBin)
{
   // contrastLimit >> 1 means no contrast equalization; ~1 means strong
   // contrast limiting and thus weak histogram equalization; 0 means the
   // original image is returned (full contrast limitation).
   int maxCount = contrastLimit * tileHeight * tileWidth / nBins;

   cl_int err = CL_SUCCESS;

   err |= clSetKernelArg(k_clipHistogram, 0, sizeof(cl_mem), &d_hist);
   err |= clSetKernelArg(k_clipHistogram, 1, sizeof(cl_int), &iHist);
   err |= clSetKernelArg(k_clipHistogram, 2, sizeof(cl_int), &nBins);
   err |= clSetKernelArg(k_clipHistogram, 3, sizeof(cl_int), &maxCount);
   err |= clSetKernelArg(k_clipHistogram, 4, sizeof(cl_mem), &d_excess);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set arguments!\n");
      return EXIT_FAILURE;
   }
   size_t histGws[] = {nBins};
   clEnqueueNDRangeKernel(queue, k_clipHistogram, 1, NULL, histGws, NULL, 0, NULL, NULL);


   err |= clSetKernelArg(k_addExcess, 0, sizeof(cl_mem), &d_hist);
   err |= clSetKernelArg(k_addExcess, 1, sizeof(cl_int), &iHist);
   err |= clSetKernelArg(k_addExcess, 2, sizeof(cl_int), &nBins);
   err |= clSetKernelArg(k_addExcess, 3, sizeof(cl_int), &maxCount);
   err |= clSetKernelArg(k_addExcess, 4, sizeof(cl_mem), &d_excess);
   err |= clSetKernelArg(k_addExcess, 5, sizeof(cl_int), &minBin);
   err |= clSetKernelArg(k_addExcess, 6, sizeof(cl_int), &maxBin);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set arguments!\n");
      return EXIT_FAILURE;
   }
   clEnqueueNDRangeKernel(queue, k_addExcess, 1, NULL, histGws, NULL, 0, NULL, NULL);
}

int main(int argc, char **argv)
{
   cl_int              err = CL_SUCCESS;

   if (argc != 4)
   {
      fprintf(stderr, "Usage: %s input-pgm-image output-pgm-image\n", argv[0]);
      exit(1);
   }

   float contrastLimit = atof(argv[3]);

   cl_platform_id platIds[10] = {0};
   cl_uint nPlatforms;
   err = clGetPlatformIDs(10, platIds, &nPlatforms);
   printf("Number of platforms: %d\n", nPlatforms);

   // Connect to a GPU compute device
   //
   cl_device_id devIds[10] = {0};
   cl_uint nDevices;
   err = clGetDeviceIDs(platIds[0], CL_DEVICE_TYPE_ALL, 10, devIds, &nDevices);
   printf("Number of devices: %d\n", nDevices);
   cl_device_id  computeDeviceId = devIds[0];
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to locate a compute device!\n");
      return EXIT_FAILURE;
   }

   size_t returned_size = 0;
   size_t max_workgroup_size = 0;
   err = clGetDeviceInfo(computeDeviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, &returned_size);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to retrieve device info!\n");
      return EXIT_FAILURE;
   }

	GROUP_SIZE = min(GROUP_SIZE, max_workgroup_size);

   cl_char vendor_name[1024] = {0};
   cl_char device_name[1024] = {0};
   err = clGetDeviceInfo(computeDeviceId, CL_DEVICE_VENDOR, sizeof(vendor_name), vendor_name, &returned_size);
   err|= clGetDeviceInfo(computeDeviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, &returned_size);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to retrieve device info!\n");
      return EXIT_FAILURE;
   }

   printf(SEPARATOR);
   printf("Connecting to %s %s...\n", vendor_name, device_name);

   // Load the compute program from disk into a cstring buffer
   //
   printf(SEPARATOR);
   const char* filename = "../clahe_kernels.cl";
   printf("Loading program '%s'...\n", filename);
   printf(SEPARATOR);

   char *source = slurpFile(filename);
   if(!source)
   {
      printf("Error: Failed to load compute program from file!\n");
      return EXIT_FAILURE;
   }

   // Create a compute ctx
   //
   cl_context ctx = clCreateContext(0, 1, &computeDeviceId, NULL, NULL, &err);
   if (!ctx)
   {
      printf("Error: Failed to create a compute ctx!\n");
      return EXIT_FAILURE;
   }

   // Create a command queue
   //
   cl_command_queue queue = clCreateCommandQueue(ctx, computeDeviceId, 0, &err);
   if (!queue)
   {
      printf("Error: Failed to create a command queue!\n");
      return EXIT_FAILURE;
   }

   // Create the compute program from the source buffer
   //
   cl_program program = clCreateProgramWithSource(ctx, 1, (const char **) & source, NULL, &err);
   if (!program || err != CL_SUCCESS)
   {
      printf("%s\n", source);
      printf("Error: Failed to create compute program!\n");
      return EXIT_FAILURE;
   }

   // Build the program executable
   //
   err = clBuildProgram(program, 0, NULL,
                        "-DINFIX_PREFIX_SUM_OP=1 -DPREFIX_SUM_OP=+ -DPREFIX_SUM_DATA_TYPE=int",
                        NULL, NULL);
   if (err != CL_SUCCESS)
   {
      size_t length;
      char build_log[2048];
      printf("%s\n", source);
      printf("Error: Failed to build program executable!\n");
      clGetProgramBuildInfo(program, computeDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, &length);
      printf("%s\n", build_log);
      return EXIT_FAILURE;
   }

   cl_program maxScan = clCreateProgramWithSource(ctx, 1, (const char **) &source, NULL, &err);
   err = clBuildProgram(maxScan, 0, NULL,
                        "-DINFIX_PREFIX_SUM_OP=0 -DPREFIX_SUM_OP=max -DPREFIX_SUM_DATA_TYPE=uchar",
                        NULL, NULL);

   cl_program minScan = clCreateProgramWithSource(ctx, 1, (const char **) &source, NULL, &err);
   err = clBuildProgram(minScan, 0, NULL,
                        "-DINFIX_PREFIX_SUM_OP=0 -DPREFIX_SUM_OP=min -DPREFIX_SUM_DATA_TYPE=uchar",
                        NULL, NULL);

   free(source);

   // MY CODE HERE

   int nPixels = WIDTH * HEIGHT;

   cl_kernel k_histogram = clCreateKernel(program, "histogram", &err);
   cl_kernel k_prefixSum = clCreateKernel(program, "prefixSum", &err);
   cl_kernel k_prefixSumIncToExc = clCreateKernel(program, "prefixSumIncToExc", &err);
   cl_kernel k_equalize = clCreateKernel(program, "equalize", &err);
   cl_kernel k_localEqualize = clCreateKernel(program, "localEqualize", &err);
   cl_kernel k_clipHistogram = clCreateKernel(program, "clipHistogram", &err);
   cl_kernel k_addExcess = clCreateKernel(program, "addExcess", &err);
   cl_kernel k_localHistogram = clCreateKernel(program, "localHistogram", &err);
   cl_kernel k_min = clCreateKernel(minScan, "prefixSum", &err);
   cl_kernel k_max = clCreateKernel(maxScan, "prefixSum", &err);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to create kernel!\n");
      return EXIT_FAILURE;
   }
   cl_mem d_hist = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N_BINS * N_TILES * sizeof(cl_int), NULL, NULL);
   cl_mem d_img = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nPixels * sizeof(cl_uchar), NULL, NULL);
   cl_mem d_imgOut = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nPixels * sizeof(cl_uchar), NULL, NULL);
   cl_mem d_excess = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N_TILES * sizeof(cl_int), NULL, NULL);

   uint8_t srcImg[WIDTH][HEIGHT];
   uint8_t dstImg[WIDTH][HEIGHT];
   int hist[N_TILES][N_BINS] = {0};

   err |= clEnqueueWriteBuffer(queue, d_hist, CL_TRUE, 0, sizeof(int) * N_TILES * N_BINS, hist, 0, NULL, NULL);

   FILE *in = fopen(argv[1], "rb");
#define MAX_HEADER_LENGTH 8192
   char header[MAX_HEADER_LENGTH] = {0};
   fgets(&header[strlen(header)], MAX_HEADER_LENGTH - 1, in);
   fgets(&header[strlen(header)], MAX_HEADER_LENGTH - 1, in);
   fgets(&header[strlen(header)], MAX_HEADER_LENGTH - 1, in);
   fgets(&header[strlen(header)], MAX_HEADER_LENGTH - 1, in);
#undef MAX_HEADER_LENGTH
   fread(srcImg, sizeof(uint8_t), nPixels, in);
   fclose(in);

   memcpy(dstImg, srcImg, nPixels * sizeof(uint8_t));

   err |= clEnqueueWriteBuffer(queue, d_img, CL_TRUE, 0, sizeof(cl_uchar) * nPixels, srcImg, 0, NULL, NULL);
   if (err != CL_SUCCESS)
   {
      printf("Error(%d): Failed to copy image!\n", err);
      return EXIT_FAILURE;
   }

   int nBins = N_BINS;
   int iTile = 0;
   int jTile = 0;
   int iHist = jTile * N_X_TILES + iTile;
   int iCdf = iHist;
   int minVal = 0;
   int maxVal = 255;
   int width = WIDTH;
   int height = HEIGHT;
   int tileWidth = 80;
   int tileHeight = 80;
   int ldaNumer = WIDTH;
   int ldaDenom = 1;
   // A ratio that is close to an 60 degrees
   /* int ldaNumer = 7*WIDTH+4; */
   /* int ldaDenom = 7; */

   /* int i = 0; */
   for (int j = 0; j < N_X_TILES; j++)
      for (int i = 0; i < N_Y_TILES; i++)
         histogram(queue, k_histogram,
                   d_img, nBins,
                   i, j,
                   d_hist,
                   minVal, maxVal,
                   width, height,
                   tileWidth, tileHeight,
                   ldaNumer, ldaDenom);

   int h_excess[N_TILES] = {0};
   clEnqueueWriteBuffer(queue, d_excess, CL_TRUE, 0, N_TILES * sizeof(cl_int), &h_excess, 0, NULL, NULL);

   // Compute min and max
   cl_mem d_imgWorking1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                         nPixels * sizeof(cl_uchar),
                                         NULL, NULL);
   cl_mem d_imgWorking2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
                                         nPixels * sizeof(cl_uchar),
                                         NULL, NULL);
   clEnqueueCopyBuffer(queue, d_img, d_imgWorking1, 0, 0, nPixels * sizeof(cl_uchar),
                       0, NULL, NULL);

   /* printf("Before read\n"); */
   /* fflush(stdout); */
   prefixSum(queue, k_max, d_imgWorking1, d_imgWorking2, 0, nPixels);

   uint8_t maxImgVal;
   clEnqueueReadBuffer(queue, d_imgWorking1, CL_TRUE, nPixels - 1, 1, &maxImgVal, 0, NULL, NULL);

   clEnqueueCopyBuffer(queue, d_img, d_imgWorking1, 0, 0, nPixels * sizeof(cl_uchar),
                       0, NULL, NULL);
   prefixSum(queue, k_min, d_imgWorking1, d_imgWorking2, 0, nPixels);

   uint8_t minImgVal;
   clEnqueueReadBuffer(queue, d_imgWorking1, CL_TRUE, nPixels - 1, 1, &minImgVal, 0, NULL, NULL);

   printf("Max: %d, Min: %d\n", maxImgVal, minImgVal);

   int minBin = (float) (minImgVal - minVal) * nBins / (maxVal - minVal);
   int maxBin = (float) (maxImgVal - minVal) * nBins / (maxVal - minVal);
   for (int i = 0; i < N_TILES; i++)
      limitContrast(queue,
                    k_clipHistogram, k_addExcess,
                    d_hist, i, nBins, tileWidth, tileHeight,
                    contrastLimit, d_excess, minImgVal, maxImgVal);

   // Save the histogram
   clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0, sizeof(cl_int) * N_TILES * N_BINS, hist, 0, NULL, NULL);

   cl_mem d_working = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N_BINS * N_TILES * sizeof(cl_int), NULL, NULL);
   for (int i = 0; i < N_TILES; i++)
      // d_hist now holds the inclusive cdf and d_working holds the exclusive cdf
      prefixSumEx(queue, k_prefixSum, k_prefixSumIncToExc, d_hist, d_working, i, N_BINS);


   int cdf[N_TILES][N_BINS] = {0};
   clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0, sizeof(cl_int) * N_TILES * N_BINS, cdf, 0, NULL, NULL);
   int cdfLo[N_TILES][N_BINS] = {0};
   clEnqueueReadBuffer(queue, d_working, CL_TRUE, 0, sizeof(cl_int) * N_TILES * N_BINS, cdfLo, 0, NULL, NULL);

   cl_float normalizationFactor = 255.0 / (tileWidth * tileHeight);

   err = CL_SUCCESS;
   cl_int nWidthTiles = N_X_TILES;
   cl_int nHeightTiles = N_Y_TILES;
   err |= clSetKernelArg(k_localEqualize, 0, sizeof(cl_mem), &d_img);
   err |= clSetKernelArg(k_localEqualize, 1, sizeof(cl_int), &width);
   err |= clSetKernelArg(k_localEqualize, 2, sizeof(cl_int), &height);
   err |= clSetKernelArg(k_localEqualize, 3, sizeof(cl_int), &nWidthTiles);
   err |= clSetKernelArg(k_localEqualize, 4, sizeof(cl_int), &nHeightTiles);
   err |= clSetKernelArg(k_localEqualize, 5, sizeof(cl_int), &minVal);
   err |= clSetKernelArg(k_localEqualize, 6, sizeof(cl_int), &maxVal);
   err |= clSetKernelArg(k_localEqualize, 7, sizeof(cl_mem), &d_working);
   err |= clSetKernelArg(k_localEqualize, 8, sizeof(cl_mem), &d_hist);
   err |= clSetKernelArg(k_localEqualize, 9, sizeof(cl_int), &nBins);
   err |= clSetKernelArg(k_localEqualize, 10, sizeof(cl_mem), &d_imgOut);
   err |= clSetKernelArg(k_localEqualize, 11, sizeof(cl_float), &normalizationFactor);
   if (err != CL_SUCCESS)
   {
      printf("Error: Failed to set arguments!\n");
      return EXIT_FAILURE;
   }
   {
      size_t gws[] = {width, height};
      size_t lws[] = {32, 1};
      clEnqueueNDRangeKernel(queue, k_localEqualize, 2, NULL, gws, lws, 0, NULL, NULL);
   }
   /* for (int j = 0; j < 4; j++) */
   /*    for (int i = 0; i < 4; i++) */
   /*       equalize(queue, k_equalize, d_img, width, height, i, j, */
   /*                tileWidth, tileHeight, */
   /*                ldaNumer, ldaDenom, minVal, maxVal, d_hist, nBins, */
   /*                d_imgOut); */

   clEnqueueReadBuffer(queue, d_imgOut, CL_TRUE, 0, sizeof(cl_uchar) * nPixels, dstImg, 0, NULL, NULL);

   for (int i = 0; i < N_TILES; i++)
   {
      for (int j = 0; j < N_BINS; j++)
         printf("(%d %d %d %d) ", j, hist[i][j], cdfLo[i][j], cdf[i][j]);
      printf("\n");
   }

   for (int i = 0; i < (N_BINS < WIDTH? N_BINS : WIDTH); i++)
   {
      /* printf("(%d, %d), ", hist[iHist][i], cdf[iCdf][i]); */
      float scale = (float) (width * height) / (tileWidth * tileHeight);
      float cdfScale = scale/4000;
      /* for (int j = 0; j < (cdf[iCdf][i]*cdfScale < HEIGHT? cdf[iCdf][i]*cdfScale : HEIGHT-1); j++) */
      /* { */
      /*    dstImg[HEIGHT-1 - j][i] = 255; */
      /* } */
      float histScale = scale / 40;
      /* for (int j = 0; j < (hist[iHist][i]*histScale < HEIGHT? hist[iHist][i]*histScale : HEIGHT-1); j++) */
      /* { */
      /*    dstImg[j][i] = 255; */
      /* } */
   }
   /* printf("\n"); */

   FILE *out = fopen(argv[2], "wb");
   fprintf(out, header);
   fwrite(dstImg, sizeof(uint8_t), nPixels, out);
   fclose(out);

   clReleaseProgram(program);
   clReleaseCommandQueue(queue);
   clReleaseContext(ctx);

   return 0;
}
